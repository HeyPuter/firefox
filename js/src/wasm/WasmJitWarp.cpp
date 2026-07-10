/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*-
 * vim: set ts=8 sts=2 et sw=2 tw=80:
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

// WasmJitWarp.cpp -- front-end + compile orchestrator of the re-architected
// JS->wasm JIT. Reuses SpiderMonkey's WarpBuilder by feeding it a real JSScript
// (=> compilingWasm()==false => full typed JS-tier MIR), runs OptimizeMIR, then
// drives the MIR->wasm back-end (WasmJitBackend.cpp), assembles a wasm module,
// and host-compiles + instantiates it. Returns a host handle (or -1).

#include "mozilla/ScopeExit.h"
#include "mozilla/TimeStamp.h"  // GECKO_WJ_COMPILESTAT split compile timing

#include <stdio.h>
#include <stdlib.h>

#include "gc/GC.h"
#include "jit/BaselineIC.h"
#include "jit/CompileInfo.h"
#include "jit/CompileWrappers.h"
#include "jit/Ion.h"
#include "jit/IonOptimizationLevels.h"
#include "jit/JitAllocPolicy.h"
#include "jit/JitContext.h"
#include "jit/JitScript.h"
#include "jit/MIR.h"
#include "jit/MIRGenerator.h"
#include "jit/MIRGraph.h"
#include "jit/TrialInlining.h"
#include "jit/WarpBuilder.h"
#include "jit/WarpOracle.h"
#include "jit/WarpSnapshot.h"
#include "vm/JSContext.h"
#include "vm/JSScript.h"
#include "wasm/WasmBinary.h"
#include "wasm/WasmConstants.h"
#include "wasm/WasmJitBackend.h"

#include "jit/InlineScriptTree-inl.h"
#include "jit/JitScript-inl.h"
#include "vm/Realm-inl.h"

using namespace js;
using namespace js::jit;
using namespace js::wasm;

extern "C" {
int wasmhost_compile(const void* bytes, int len);
int wasmhost_instantiate(int handle, const int* callbackIds, int importCount);
int wasmhost_guest_mem_objid();
int wasmhost_guest_mem_shared();
int wasmhost_jit_table();
int wasmhost_jit_table_set(int handle, int idx);
}

namespace js {
namespace wasm {

static void DumpMIR(JSScript* script, MIRGraph& graph) {
  for (ReversePostorderIterator b = graph.rpoBegin(); b != graph.rpoEnd(); b++) {
    fprintf(stderr, "[wb-dump] block%u (preds=%u succs=%u):\n", b->id(),
            unsigned(b->numPredecessors()), unsigned(b->numSuccessors()));
    for (MPhiIterator p = b->phisBegin(); p != b->phisEnd(); p++) {
      fprintf(stderr, "    phi%u ty=%s nops=%u\n", p->id(),
              StringFromMIRType(p->type()), unsigned(p->numOperands()));
    }
    for (MInstructionIterator it = b->begin(); it != b->end(); it++) {
      const char* tag = "";
      if (it->isNewObject()) tag = " <<NEWOBJECT>>";
      else if (it->isNewArray()) tag = " <<NEWARRAY>>";
      else if (it->isNewArrayObject()) tag = " <<NEWARRAYOBJECT>>";
      else if (it->isCreateThis()) tag = " <<CREATETHIS>>";
      else if (it->isCall()) tag = " <<CALL>>";
      else if (it->isBox()) tag = " <<BOX>>";
      else if (it->isUnbox()) tag = " <<UNBOX>>";
      fprintf(stderr, "    op#%u id%u ty=%s nops=%u%s\n", unsigned(it->op()),
              it->id(), StringFromMIRType(it->type()),
              unsigned(it->numOperands()), tag);
    }
  }
}

// Assemble the wasm module wrapping `e`'s emitted body and host-compile +
// instantiate it. Module shape (mirrors the proven jitref layout):
//   type0 (f64)->f64 [body], type1 (f64,f64)->f64 [wjhelp]
//   import m.help(func type1), m.mem(memory)
//   func[0]: type0 ; export "f" = func index 1 (wjhelp import is index 0)
// Dense allocator for shared-table slots. The host's funcref table has a fixed
// size (kWJTableSize); module handles are a process-global counter that easily
// exceeds it, so the table slot must NOT be the handle (call_indirect would trap
// out-of-bounds). Slots are handed out densely; once exhausted, a function still
// runs (via wasmhost_call) but can't be reached by the fast call_indirect path.
static constexpr int kWJTableSize = 4096;
static int gWJNextTblSlot = 0;

static int AssembleAndInstall(MIRGenerator& mirGen, MIRGraph& graph,
                              uint32_t nargs, WarpSnapshot* snapshot, int* tblSlotOut,
                              bool dump) {
  Bytes out;
  Encoder e(out);
  const uint8_t kF64 = uint8_t(TypeCode::F64);
  const uint8_t kI64 = uint8_t(TypeCode::I64);
  if (!e.writeFixedU32(MagicNumber) || !e.writeFixedU32(EncodingVersionModule)) {
    return -1;
  }
  size_t s;
  if constexpr (kWJEHABI) {
    // EHABI: 4 types. type0 = main (f64 sb, i64 callee, i64 this, i64 args...) -> i64
    // (no flag; exceptions via wasm EH). type1 = wjhelp (f64,f64)->f64. type2 =
    // trampoline (f64)->f64. type3 = $wjexn tag ()->() (payload-less; the JS exception
    // value stays on cx).
    if (!e.startSection(SectionId::Type, &s) || !e.writeVarU32(4)) return -1;
    if (!e.writeFixedU8(0x60) || !e.writeVarU32(3 + kWJMaxArgs) ||
        !e.writeFixedU8(kF64)) {
      return -1;
    }
    for (uint32_t i = 0; i < 2 + kWJMaxArgs; i++) {
      if (!e.writeFixedU8(kI64)) return -1;  // callee + this + args
    }
    if (!e.writeVarU32(1) || !e.writeFixedU8(kI64)) return -1;  // result: i64 only
    if (!e.writeFixedU8(0x60) || !e.writeVarU32(2) || !e.writeFixedU8(kF64) ||
        !e.writeFixedU8(kF64) || !e.writeVarU32(1) || !e.writeFixedU8(kF64))
      return -1;  // type1 wjhelp
    if (!e.writeFixedU8(0x60) || !e.writeVarU32(1) || !e.writeFixedU8(kF64) ||
        !e.writeVarU32(1) || !e.writeFixedU8(kF64))
      return -1;  // type2 trampoline
    if (!e.writeFixedU8(0x60) || !e.writeVarU32(0) || !e.writeVarU32(0))
      return -1;  // type3 tag ()->()
    e.finishSection(s);
  } else {
    // 3 types: type0 = main (f64 scratchbase, i64 this, i64 arg0..arg[kWJMaxArgs-1])
    // -> f64; type1 = wjhelp (f64,f64)->f64; type2 = trampoline (f64)->f64.
    if (!e.startSection(SectionId::Type, &s) || !e.writeVarU32(3)) return -1;
    if (!e.writeFixedU8(0x60) || !e.writeVarU32(2 + kWJMaxArgs) ||
        !e.writeFixedU8(kF64)) {
      return -1;
    }
    for (uint32_t i = 0; i < 1 + kWJMaxArgs; i++) {
      if (!e.writeFixedU8(kI64)) return -1;  // this + args
    }
    if (!e.writeVarU32(2) || !e.writeFixedU8(kF64) || !e.writeFixedU8(kI64))
      return -1;  // main results: [f64 deopt-flag, i64 boxed-result]
    if (!e.writeFixedU8(0x60) || !e.writeVarU32(2) || !e.writeFixedU8(kF64) ||
        !e.writeFixedU8(kF64) || !e.writeVarU32(1) || !e.writeFixedU8(kF64)) {
      return -1;  // type1 wjhelp
    }
    if (!e.writeFixedU8(0x60) || !e.writeVarU32(1) || !e.writeFixedU8(kF64) ||
        !e.writeVarU32(1) || !e.writeFixedU8(kF64)) {
      return -1;  // type2 trampoline
    }
    e.finishSection(s);
  }

  int memId = wasmhost_guest_mem_objid();
  if (memId < 0) return -1;
  int tblId = wasmhost_jit_table();  // shared funcref table for wasm->wasm calls
  if (tblId < 0) return -1;
  bool sharedMem = wasmhost_guest_mem_shared() != 0;
  // 3 imports: m.help (func type1), m.mem (memory), m.tbl (funcref table).
  if (!e.startSection(SectionId::Import, &s) || !e.writeVarU32(3) ||
      !e.writeBytes("m", 1) || !e.writeBytes("help", 4) || !e.writeFixedU8(0x00) ||
      !e.writeVarU32(1) ||  // wjhelp: func import of type 1
      !e.writeBytes("m", 1) || !e.writeBytes("mem", 3) || !e.writeFixedU8(0x02)) {
    return -1;
  }
  if (sharedMem) {
    if (!e.writeFixedU8(0x03) || !e.writeVarU32(1) || !e.writeVarU32(65536)) return -1;
  } else {
    if (!e.writeFixedU8(0x00) || !e.writeVarU32(0)) return -1;
  }
  // m.tbl: table import, elemtype funcref (0x70), limits min=4096 (matches host).
  if (!e.writeBytes("m", 1) || !e.writeBytes("tbl", 3) || !e.writeFixedU8(0x01) ||
      !e.writeFixedU8(0x70) || !e.writeFixedU8(0x00) || !e.writeVarU32(4096)) {
    return -1;
  }
  e.finishSection(s);

  // 2 local funcs: main (type0, func idx 1), trampoline (type2, func idx 2).
  // (wjhelp import is func idx 0.)
  if (!e.startSection(SectionId::Function, &s) || !e.writeVarU32(2) ||
      !e.writeVarU32(0) || !e.writeVarU32(2)) {
    return -1;
  }
  e.finishSection(s);

  if constexpr (kWJEHABI) {
    // Tag section (id 13): one local tag $wjexn (index 0), attribute 0x00 (exception),
    // type index 3 (()->()). `throw 0` raises it; caught by try/catch_all. The JS
    // exception value rides on cx, not the tag payload.
    if (!e.startSection(SectionId::Tag, &s) || !e.writeVarU32(1) ||
        !e.writeFixedU8(0x00) || !e.writeVarU32(3)) {
      return -1;
    }
    e.finishSection(s);
  }

  // Exports: "f" = trampoline (host entry, func idx 2) FIRST so it is export 0;
  // "m" = main (func idx 1) for shared-table registration.
  if (!e.startSection(SectionId::Export, &s) || !e.writeVarU32(2) ||
      !e.writeBytes("f", 1) || !e.writeFixedU8(0x00) || !e.writeVarU32(2) ||
      !e.writeBytes("m", 1) || !e.writeFixedU8(0x00) || !e.writeVarU32(1)) {
    return -1;
  }
  e.finishSection(s);

  // 2 code bodies: main (func idx 1) then trampoline (func idx 2).
  if (!e.startSection(SectionId::Code, &s) || !e.writeVarU32(2)) return -1;
  size_t bodyOff;
  if (!e.writePatchableVarU32(&bodyOff)) return -1;
  size_t bodyStart = e.currentOffset();
  if (!WJEmitBody(mirGen, graph, nargs, /*useThis=*/true, snapshot, e)) {
    if (dump) fprintf(stderr, "[wb-compile] WJEmitBody bailed (unsupported node)\n");
    static int cdbg = getenv("GECKO_WJ_CDBG") ? 1 : 0;
    if (cdbg)
      fprintf(stderr, "[wj-cdbg] BAIL line=%u reason=%s\n",
              js::wasm::gWJBailLine, js::wasm::gWJBailReason);
    return -1;
  }
  e.patchVarU32(bodyOff, uint32_t(e.currentOffset() - bodyStart));
  size_t trampOff;
  if (!e.writePatchableVarU32(&trampOff)) return -1;
  size_t trampStart = e.currentOffset();
  if (!WJEmitTrampoline(e)) return -1;
  e.patchVarU32(trampOff, uint32_t(e.currentOffset() - trampStart));
  e.finishSection(s);

  // jitdis debugger: GECKO_WJ_WASMDUMP dumps emitted wasm to /tmp/wbjit_<lineno>.wasm
  // (disassemble with `wasm-dis`). A numeric value >1 dumps ONLY that lineno (targeted);
  // any other set value dumps all compiled functions.
  bool wasmDump = getenv("GECKO_WJWARP_DUMP") || getenv("GECKO_WJ_WASMDUMP") ||
                  (js::wasm::gWJForceMega && getenv("GECKO_WJ_MEGADUMP"));
  if (wasmDump) {
    if (const char* wd = getenv("GECKO_WJ_WASMDUMP")) {
      int want = atoi(wd);
      JSScript* ds = mirGen.outerInfo().script();
      if (want > 1 && ds && int(ds->lineno()) != want) wasmDump = false;
    }
  }
  if (wasmDump) {
    JSScript* dscript = mirGen.outerInfo().script();
    char path[128];
    // Megamorphic recompiles dump to a SEPARATE file so they're not overwritten
    // by interleaved non-mega compiles of the same function (deltablue 414 diag).
    snprintf(path, sizeof(path), js::wasm::gWJForceMega ? "/tmp/wbjit_mega_%u.wasm"
                                              : "/tmp/wbjit_%u.wasm",
             dscript ? dscript->lineno() : 0);
    if (FILE* f = fopen(path, "wb")) {
      fwrite(out.begin(), 1, out.length(), f);
      fclose(f);
      fprintf(stderr, "[wb-compile] wrote %s (%zu bytes)\n", path,
              size_t(out.length()));
    }
  }

  static int compileStat = getenv("GECKO_WJ_COMPILESTAT") ? 1 : 0;
  mozilla::TimeStamp hostT0;
  if (compileStat) hostT0 = mozilla::TimeStamp::Now();
  int handle = wasmhost_compile(out.begin(), int(out.length()));
  if (compileStat) {
    js::wasm::gWJHostCompileMs +=
        (mozilla::TimeStamp::Now() - hostT0).ToMilliseconds();
    js::wasm::gWJEmitBytes += uint64_t(out.length());
  }
  if (handle < 0) {
    // The host (V8) REJECTED the emitted wasm module -- our codegen produced
    // invalid/oversized wasm for this function. This is NOT an unsupported-node
    // bail; the stale gWJBailReason (last emitted op) would mislead. Record it +
    // the body size so LOGBAIL points here. GECKO_WJ_DUMPBADWASM writes the bytes.
    js::wasm::gWJBailReason = "host-compile-reject";
    if (getenv("GECKO_WJ_DUMPBADWASM")) {
      fprintf(stderr, "[wb-badwasm] host rejected module: %zu bytes (fn line %u)\n",
              size_t(out.length()),
              mirGen.outerInfo().script() ? mirGen.outerInfo().script()->lineno() : 0);
    }
    return -1;
  }
  const int importIds[3] = {-3, memId, tblId};  // help (shim), mem, tbl
  mozilla::TimeStamp instT0;
  if (compileStat) instT0 = mozilla::TimeStamp::Now();
  int instRes = wasmhost_instantiate(handle, importIds, 3);
  if (compileStat) {
    js::wasm::gWJHostInstMs +=
        (mozilla::TimeStamp::Now() - instT0).ToMilliseconds();
  }
  if (instRes != 0) {
    js::wasm::gWJBailReason = "host-instantiate-fail";
    return -1;
  }
  // Register this function in the shared table at a dense slot, so other JIT'd
  // functions can call_indirect it. -1 if the table is full (slow path only).
  // REUSE an existing slot (*tblSlotOut >= 0 on a deopt-storm RECOMPILE): callers'
  // polymorphic call ICs cache (funPtr -> tblSlot), and the funPtr is unchanged
  // across a recompile, so the new module MUST take over the same slot -- else the
  // cached ICs call_indirect the stale slot (old/garbage module -> deltablue wrong
  // results). On the first compile *tblSlotOut is -1 -> allocate a fresh slot.
  int slot = (*tblSlotOut >= 0)
                 ? *tblSlotOut
                 : ((gWJNextTblSlot < kWJTableSize) ? gWJNextTblSlot++ : -1);
  if (slot >= 0) (void)wasmhost_jit_table_set(handle, slot);
  *tblSlotOut = slot;
  return handle;
}

// Build Warp MIR for `script`, optimize, lower to wasm, host-compile. Returns a
// host handle (>=0) on success and writes *nargsOut, or -1 to stay in PBL.
int WJWarpCompile(JSContext* cx, JSScript* script, uint32_t* nargsOut,
                  uint32_t* nlocalsOut, int* tblSlotOut) {
  bool dump = getenv("GECKO_WJWARP_DUMP");
  // Functions with try/catch/finally (try-notes): our JIT propagates a thrown
  // exception OUT of a function via a return flag, but has NO in-function CATCH
  // landing pad. So a throw inside a try region unwinds PAST its own catch instead
  // of landing in it -- e.g. gbemu initLCD's `try { new GameBoyCanvas() } catch`
  // (the headless canvas op throws; the catch must run) hard-failed. Bail such
  // functions to PBL, which handles try/catch correctly. They are typically cold
  // (setup / feature-detection), so the perf cost is ~nil.
  // 2026-06-27: NOW DEFAULT-OFF (try/catch fns COMPILE). The catch runs in PBL via
  // deopt-in-error: an in-try call/helper that throws sets gWJResumeInError + deopts;
  // PBL's HandleException walks the trynotes and runs the catch (Warp builds no
  // catch-block MIR). try-body runs in JIT (the win: ubo compileToFilter et al.).
  // Validated: tc_repro + octane no-regression + ubo correct. GECKO_WJ_TRYBAIL restores
  // the old whole-function bail.
  if (getenv("GECKO_WJ_TRYBAIL")) {
    // Only Catch/Finally need an exception LANDING pad (they swallow/handle a
    // throw). ForIn/ForOf/Destructuring/Loop try-notes are normal control-flow
    // cleanup -- bailing on those too is far too broad (it tanked richards 5x:
    // for-of/destructuring are everywhere). Bail only when a real catch/finally
    // is present.
    for (const js::TryNote& tn : script->trynotes()) {
      if (tn.kind() == js::TryNoteKind::Catch ||
          tn.kind() == js::TryNoteKind::Finally) {
        js::wasm::gWJBailReason = "trycatch";
        return -1;
      }
    }
  }
  // Enter the SCRIPT's realm for the whole compile. The inline tier-up path is
  // already in it (the fn is executing), but the DEFERRED-compile drain
  // (WasmJitDrainDeferred, called from the embed's idle loop with no JS on the
  // stack) is NOT -- and the rest of this function reads cx->realm()/cx->global()
  // (CompileRealm, global lexical env, nursery alloc site). Compiling in the wrong
  // realm built the snapshot/MIR against the wrong global -> browser crash after an
  // idle drain (defer-on correctness sweep died mid-run). Re-entering the same realm
  // on the inline path is cheap + safe. (Nested AutoRealms below are harmless.)
  AutoRealm ar(cx, script);
  // Bake the zone's needs-marking-barrier flag address for the emitted pre-write
  // barrier fast path (single-zone shell -> one stable address).
  js::wasm::gWJMarkBarrierAddr =
      uintptr_t(script->zone()->addressOfNeedsMarkingBarrier());
  js::wasm::gWJWholeCellLastAddr =
      uintptr_t(cx->runtime()->gc.storeBuffer().addressOfLastBufferedWholeCell());
  js::wasm::gWJGlobalLexEnvVal =
      JS::ObjectValue(cx->global()->lexicalEnvironment()).asRawBits();
  // Inline nursery bump-allocation params (for the GECKO_WJ_INLINEALLOC path):
  // the zone's nursery position_ address (loaded/bumped at runtime) and the
  // catch-all-alloc-site object cell header word. Stable per zone.
  {
    jit::CompileZone* cz = jit::CompileRealm::get(cx->realm())->zone();
    js::wasm::gWJNurseryPosAddr = uintptr_t(cz->addressOfNurseryPosition());
    js::gc::AllocSite* site = cz->catchAllAllocSite(
        JS::TraceKind::Object, js::gc::CatchAllAllocSite::Optimized);
    js::wasm::gWJObjHeaderWord =
        js::gc::NurseryCellHeader::MakeValue(site, JS::TraceKind::Object);
    // String nursery cell header (for inline rope-concat bump-alloc). Gated on the
    // nursery actually allocating strings -- if it tenures strings, leave 0 so the
    // inline rope path falls back to the ConcatStrings helper (correct either way).
    if (cx->nursery().canAllocateStrings()) {
      js::gc::AllocSite* ssite = cz->catchAllAllocSite(
          JS::TraceKind::String, js::gc::CatchAllAllocSite::Optimized);
      js::wasm::gWJStringHeaderWord =
          js::gc::NurseryCellHeader::MakeValue(ssite, JS::TraceKind::String);
    } else {
      js::wasm::gWJStringHeaderWord = 0;
    }
  }
  // Bisection: GECKO_WJ_SKIPLINE=N[,M,...] bails the functions at those lines.
  if (const char* sl = getenv("GECKO_WJ_SKIPLINE")) {
    uint32_t ln = uint32_t(script->lineno());
    const char* p = sl;
    while (*p) {
      uint32_t v = uint32_t(atoi(p));
      if (v == ln) return -1;
      while (*p && *p != ',') p++;
      if (*p == ',') p++;
    }
  }
  RootedScript scriptRoot(cx, script);

  if (!script->hasJitScript()) {
    if (!cx->zone()->ensureJitZoneExists(cx)) return -1;
    AutoRealm ar(cx, script);
    AutoKeepJitScripts keep(cx);
    if (!script->ensureHasJitScript(cx, keep)) return -1;
  }

  // Drive trial inlining RIGHT BEFORE building the snapshot, so WarpBuilder
  // inlines callee dispatch into this script. Production does this from the
  // baseline warmup hook; under PBL the timing never lines up (the hot caller is
  // already compiled+in-wasm before its LoopHead hook fires), so we invoke the
  // inliner directly here on the base ICScript. Bounded to MaxICScriptDepth
  // internally; best-effort (ignore failure -> just no inlining).
  if (getenv("GECKO_WJ_COLDCALL")) {
    AutoRealm ar(cx, script);
    // Crank the inliner toward Ion-like aggressiveness: inline call sites with far
    // fewer warmup entries and much larger callee bodies. raytrace's hot path is
    // tiny Vector methods (dot/add/...) whose call overhead dwarfs their body --
    // inlining them is the win. GECKO_WJ_INLINEAGGR tunes; default cranks hard.
    if (!getenv("GECKO_WJ_NOINLINEAGGR")) {
      jit::JitOptions.inliningEntryThreshold = 2;
      jit::JitOptions.smallFunctionMaxBytecodeLength = 2000;
    }
    // GECKO_WJ_INLINEMONO: force BODY-inlining of monomorphic callees (override
    // ShouldUseMonomorphicInlining, which otherwise picks the no-body-splice "Mono"
    // strategy for already-monomorphic targets -- e.g. crypto-sha1's hot pure-int32
    // leaves safe_add/rol/sha1_ft/kt all DECIDE=Mono, so their CALL overhead stays in
    // the inner loop). Mono skips specialization-wins but ALSO skips removing the call.
    if (getenv("GECKO_WJ_INLINEMONO"))
      jit::JitOptions.monomorphicInlining = jit::UseMonomorphicInlining::Never;
    jit::ICScript* ic = scriptRoot->jitScript()->icScript();
    jit::TrialInliner inliner(cx, scriptRoot, ic);
    if (!inliner.tryInlining()) {
      cx->clearPendingException();
    }
  }

  LifoAlloc lifo(TempAllocator::PreferredLifoChunkSize, js::MallocArena);
  TempAllocator temp(&lifo);
  JitContext jctx(cx);
  MIRGraph graph(&temp);

  InlineScriptTree* inlineScriptTree =
      InlineScriptTree::New(&temp, nullptr, nullptr, script);
  if (!inlineScriptTree) return -1;
  CompileInfo* info = temp.lifoAlloc()->new_<CompileInfo>(
      CompileRuntime::get(cx->runtime()), script, /*osrPc=*/nullptr,
      script->needsArgsObj(), inlineScriptTree);
  if (!info || info->compilingWasm()) return -1;

  const OptimizationInfo* optimizationInfo =
      IonOptimizations.get(OptimizationLevel::Normal);
  const JitCompileOptions options(cx);
  MIRGenerator mirGen(CompileRealm::get(cx->realm()), options, &temp, &graph,
                      info, optimizationInfo);

  static int compileStatP = getenv("GECKO_WJ_COMPILESTAT") ? 1 : 0;
  WarpSnapshot* snapshot = nullptr;
  {
    gc::AutoSuppressGC suppressGC(cx);
    mozilla::TimeStamp t0;
    if (compileStatP) t0 = mozilla::TimeStamp::Now();
    WarpOracle oracle(cx, mirGen, scriptRoot);
    AbortReasonOr<WarpSnapshot*> result = oracle.createSnapshot();
    if (result.isErr()) return -1;
    snapshot = result.unwrap();
    if (compileStatP)
      js::wasm::gWJSnapshotMs += (mozilla::TimeStamp::Now() - t0).ToMilliseconds();
  }
  {
    gc::AutoSuppressGC suppressGC(cx);
    mozilla::TimeStamp t0;
    if (compileStatP) t0 = mozilla::TimeStamp::Now();
    WarpCompilation comp(mirGen.alloc());
    WarpBuilder builder(*snapshot, mirGen, &comp);
    if (!builder.build()) return -1;
    if (compileStatP)
      js::wasm::gWJBuildMs += (mozilla::TimeStamp::Now() - t0).ToMilliseconds();
  }
  {
    gc::AutoSuppressGC suppressGC(cx);
    mozilla::TimeStamp t0;
    if (compileStatP) t0 = mozilla::TimeStamp::Now();
    // NB: disabling GVN/LICM/RangeAnalysis here to cut OptimizeMIR time (~19% of
    // tier-up compile) is a NET LOSS -- measured: the un-optimized graph has more
    // instructions, so emit + host-compile (which scale with graph/output size) grow
    // MORE than OptimizeMIR shrinks (per-compile 1.96ms -> 2.27ms). The optimizer
    // pays for itself by shrinking the graph. So run the full Normal pipeline.
    if (!OptimizeMIR(&mirGen)) return -1;
    if (compileStatP)
      js::wasm::gWJOptimizeMs += (mozilla::TimeStamp::Now() - t0).ToMilliseconds();
  }
  if (dump) {
    fprintf(stderr, "[wb-compile] %s:%u optimized, %u blocks\n",
            script->filename() ? script->filename() : "?",
            uint32_t(script->lineno()), unsigned(graph.numBlocks()));
    DumpMIR(script, graph);
  }

  uint32_t nargs = script->function() ? script->function()->nargs() : 0;
  // NB: *tblSlotOut is IN/OUT -- a >=0 value on entry is a preferred slot to REUSE
  // (deopt-storm recompile keeps the same slot so cached call ICs stay valid).
  int handle = AssembleAndInstall(mirGen, graph, nargs, snapshot, tblSlotOut, dump);
  if (handle < 0) {
    if (dump) fprintf(stderr, "[wb-compile] assemble/host-compile failed\n");
    return -1;
  }
  *nargsOut = nargs;
  *nlocalsOut = info->nlocals();
  static int instlog = getenv("GECKO_WJ_INSTLOG") ? 1 : 0;
  if (dump || instlog)
    fprintf(stderr, "[wb-compile] installed %s:%u handle=%d nargs=%u\n",
            script->filename() ? script->filename() : "?",
            uint32_t(script->lineno()), handle, nargs);
  return handle;
}

}  // namespace wasm
}  // namespace js
