/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*-
 * vim: set ts=8 sts=2 et sw=2 tw=80:
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

// WasmJitBackend.cpp -- MIR->wasm-bytecode back-end. Walks an OptimizeMIR'd Warp
// graph and emits a wasm function body in "value-per-local" form: every used
// MDefinition gets its own wasm local; each node computes its value (pushing
// operands via local.get) then local.set's its own local. The host engine
// re-runs real register allocation on the result. Control flow lowers to a
// single br_table dispatch loop. JS::Values are NUNBOX32 (low32 payload, high32
// tag).

#include "wasm/WasmJitBackend.h"

#include <algorithm>
#include <stdio.h>
#include <limits>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <string>
#include <tuple>
#include <vector>

#include "jit/CompileInfo.h"
#include "jit/InlineScriptTree.h"  // BytecodeSite (precise bail op location)
#include "jit/IonTypes.h"  // IsResumeAfter
#include "jit/MIR.h"
#include "jit/MIRGenerator.h"
#include "jit/MIRGraph.h"
#include "jit/WarpSnapshot.h"
#include "js/Value.h"
#include "jsfriendapi.h"  // js::IdToValue
#include "js/shadow/Object.h"

// OSR cheap-resume: backend appends each OSR-able loop head's (resume pcOff ->
// dispatch block index) to a runtime-side sink (defined in WasmJitRuntime.cpp).
extern "C" void WJAddOsrTarget(uint32_t pcOff, uint32_t blockIdx);
#include "vm/BytecodeUtil.h"  // GetNextPc
#include "vm/JSScript.h"  // PCToLineNumber (precise bail op location)
#include "vm/JSFunction.h"
#include "vm/GetterSetter.h"  // LoadGetterSetterFunction offsets
#include "vm/JSObject.h"
#include "vm/NativeObject.h"
#include "vm/Caches.h"        // js::MegamorphicCache (inline megamorphic prop cache)
#include "vm/PropertyKey.h"   // HashPropertyKeyThreadSafe (megamorphic cache hash)
#include "vm/Iteration.h"  // NativeIterator/PropertyIndex layout (LoadSlotByIteratorIndex)
#include "vm/BoundFunctionObject.h"     // offsetOfFlagsSlot/NumBoundArgsShift (BoundFunctionNumArgs)
#include "vm/ProxyObject.h"             // offsetOfPrivateSlot (LoadWrapperTarget)
#include "vm/ArrayObject.h"             // ArrayObject::class_ (element-store IC clasp guard)
#include "vm/ArrayBufferViewObject.h"  // dataOffset (typed-array element access)
#include "vm/ArgumentsObject.h"        // arguments-object slot offsets (CreateArgumentsObject family)
#include "vm/ArrayBufferObject.h"      // offsetOfByteLengthSlot (ArrayBufferByteLength)
#include "vm/TypedArrayObject.h"       // fixedLengthClasses (typed-array store IC)
#include "js/ScalarType.h"             // Scalar::Type/byteSize (typed arrays)
#include "vm/JSAtomUtils-inl.h"  // js::AtomToId (GetPropertyCache inline IC key)
#include "gc/Heap.h"      // gc::Arena::thingSize, NurseryCellHeader (inline alloc)
#include "gc/Nursery.h"   // Nursery::nurseryCellHeaderSize/offsetOfCurrentEndFromPosition
#include "gc/ObjectKind-inl.h"  // gc::GetGCObjectKind (object-literal inline alloc)
#include "vm/GlobalObject.h"     // getArrayShapeWithDefaultProto (array inline alloc)
#include "vm/JSObject-inl.h"     // GuessArrayGCKind (array inline alloc)
#include "vm/StringType.h"       // JSString/JSRope/JSFatInlineString (inline rope concat)
#include "vm/StringFlags.h"      // StringFlags::ropeFlags / CharEncoding (inline rope concat)
#include "vm/EnvironmentObject.h"  // BlockLexicalEnvironmentObject (NewLexicalEnvironmentObject)
#include "vm/Scope.h"            // LexicalScope (NewLexicalEnvironmentObject)
#include "jit/InlinableNatives.h"  // InlinableNative (UnsafeGet/SetReservedSlot inlining)
#include "wasm/WasmBinary.h"

#include <cstdlib>  // std::getenv (WJGETENV)

// Per-call-site cached getenv. The whole file's GECKO_WJ_* gating/debug flags are
// process-lifetime constants (the environment never changes mid-run), but many are
// read in the PER-MIR-NODE lowering switch. Under emscripten getenv() is a linear
// environ scan, so an uncached read per op made WJEmitBody (the wasm-emit backend)
// the dominant tier-up compile cost (~72% of a real-site load's synchronous compile
// -- GECKO_WJ_COMPILESTAT). Each expansion gets its own function-local static, so
// the getenv runs once per call site for the whole process. Behavior-identical.
#define WJGETENV(nm) \
  ([]() -> char* { static char* wjEnvV_ = std::getenv(nm); return wjEnvV_; }())

using namespace js;
const uint32_t js::wasm::gWJOpGuardGlobalGeneration =
    uint32_t(js::jit::MDefinition::Opcode::GuardGlobalGeneration);
using namespace js::jit;
using namespace js::wasm;

namespace {

static int gWJDbg = -1;
static inline bool WJDbg() {
  if (gWJDbg < 0) gWJDbg = WJGETENV("GECKO_WJWARP_DUMP") ? 1 : 0;
  return gWJDbg;
}
#define WJBAIL(...) (WJDbg() ? (fprintf(stderr, "[wb-bail] " __VA_ARGS__), false) : false)

// GECKO_WJ_SITEHIST: definitive per-deopt-site attribution. Each EmitDeopt gets a
// unique global id whose metadata (MIR curOp, emit __LINE__, JS script:line) is
// recorded at compile time; the emitted deopt path bumps gWJSiteHits[id] at
// runtime. atexit dumps the hottest sites with full attribution -- unlike the
// gWJDeoptByOp histogram this is never confused by inlined/rematerialized guards.
static constexpr uint32_t kWJMaxDSites = 8192;
uint32_t gWJSiteHits[kWJMaxDSites] = {0};
static uint16_t gWJSiteOp[kWJMaxDSites] = {0};
static uint32_t gWJSiteLine[kWJMaxDSites] = {0};
static const char* gWJSiteFile[kWJMaxDSites] = {nullptr};
static uint32_t gWJSiteJsLine[kWJMaxDSites] = {0};
static uint32_t gWJSiteCounter = 0;
static int gWJSiteHistEnabled = -1;
// GECKO_WJ_BCRT: idx/len/hits at a taken bounds-check deopt (store-only, no
// helper call -- safe to emit). Dumped by WJDumpSiteHist.
int32_t gWJBCDbgIdx = -1;
int32_t gWJBCDbgLen = -1;
uint32_t gWJBCDbgHits = 0;
// GECKO_WJ_GSRT: actual obj->shape_ vs expected (pool) at a taken GuardShape deopt.
uint32_t gWJGSActual = 0;
uint32_t gWJGSExpect = 0;
uint32_t gWJGSHits = 0;
// GECKO_WJ_UNBOXDBG: at a taken Unbox tag-guard deopt, histogram the ACTUAL value tag
// (gWJUnboxTagHist[actualTag&15]) and record the most-recent expected tag. Tells us
// WHAT type is arriving when an Unbox bails (the ubo charCodeAt 90k-deopt mystery).
uint32_t gWJUnboxTagHist[16] = {0};
uint32_t gWJUnboxExpectTag = 0;
// Current function's script start-line, set per-instruction; used by GECKO_WJ_HYBSTORELINE
// to scope the WIP store-hybrid to ONE function while bisecting the correctness bug.
uint32_t gWJCurScriptLine = 0;
void WJDumpSiteHist() {
  // Top sites by runtime hit count.
  uint32_t n = gWJSiteCounter < kWJMaxDSites ? gWJSiteCounter : kWJMaxDSites;
  for (uint32_t rank = 0; rank < 30; rank++) {
    uint32_t best = 0, bestIdx = n;
    for (uint32_t i = 0; i < n; i++) {
      if (gWJSiteHits[i] > best) { best = gWJSiteHits[i]; bestIdx = i; }
    }
    if (bestIdx == n || best == 0) break;
    fprintf(stderr,
            "[wj-sitehist] hits=%u op#%u line=%u %s:%u\n", best,
            gWJSiteOp[bestIdx], gWJSiteLine[bestIdx],
            gWJSiteFile[bestIdx] ? gWJSiteFile[bestIdx] : "?",
            gWJSiteJsLine[bestIdx]);
    gWJSiteHits[bestIdx] = 0;  // consume so next rank finds the next-highest
  }
  fprintf(stderr, "[wj-bcdbg] hits=%u lastIdx=%d lastLen=%d\n", gWJBCDbgHits,
          gWJBCDbgIdx, gWJBCDbgLen);
  fprintf(stderr, "[wj-gsdbg] hits=%u actualShape=%#x expectShape=%#x\n",
          gWJGSHits, gWJGSActual, gWJGSExpect);
  {
    // Unbox tag-guard deopt distribution (GECKO_WJ_UNBOXDBG). Tag names per NUNBOX.
    static const char* tn[16] = {
        "DOUBLE", "INT32", "BOOL", "UNDEF", "NULL", "MAGIC", "STRING", "SYMBOL",
        "PRIVGC", "BIGINT", "?10", "?11", "OBJECT", "?13", "?14", "?15"};
    fprintf(stderr, "[wj-unboxdbg] lastExpectTag=%#x hist:", gWJUnboxExpectTag);
    for (int i = 0; i < 16; i++)
      if (gWJUnboxTagHist[i])
        fprintf(stderr, " %s(%d)=%u", tn[i], i, gWJUnboxTagHist[i]);
    fprintf(stderr, "\n");
  }
}

// Bail-reason tally: under GECKO_WJ_BAILDBG, count how often each MIR opcode
// forces a function to stay in PBL, and dump the histogram at process exit.
static const char* WJOpName(jit::MDefinition::Opcode op) {
  switch (op) {
#define WJ_OPNAME(o) \
  case jit::MDefinition::Opcode::o: \
    return #o;
    MIR_OPCODE_LIST(WJ_OPNAME)
#undef WJ_OPNAME
  }
  return "?";
}
static void WJTallyBail(jit::MDefinition::Opcode op) {
  js::wasm::gWJBailReason = WJOpName(op);  // record for GECKO_WJ_CDBG
  static int on = -1;
  if (on < 0) on = WJGETENV("GECKO_WJ_BAILDBG") ? 1 : 0;
  if (!on) return;
  fprintf(stderr, "[wb-baildbg] %s\n", WJOpName(op));
}

// wasm valtype byte for an MIR type, or 0 if this node carries no wasm value.
static uint8_t WJValType(MIRType t) {
  switch (t) {
    case MIRType::Int32:
    case MIRType::Boolean:
    case MIRType::IntPtr:  // wasm32: a native pointer-sized int == i32
      return uint8_t(TypeCode::I32);
    case MIRType::Int64:
    case MIRType::Value:  // boxed JS::Value (NUNBOX32 -> 64 bits)
      return uint8_t(TypeCode::I64);
    case MIRType::Double:
    case MIRType::Float32:
      // Float32 values are held in f64 locals (JS numbers are f64; only typed-array
      // STORAGE is 32-bit). This matches EmitSpillValue/arith which already treat
      // Float32 as f64, and avoids f32/f64 local.set mismatches (gbemu).
      return uint8_t(TypeCode::F64);
    case MIRType::Object:
    case MIRType::String:
    case MIRType::Symbol:
    case MIRType::BigInt:
    case MIRType::Slots:
    case MIRType::Elements:
      return uint8_t(TypeCode::I32);  // wasm32 pointer (low 32 bits)
    default:
      return 0;
  }
}

// Whether a typed slot-load's unbox must stay fallible (emit the per-access tag
// check + deopt). GECKO_WJ_INFALLIBLESLOT makes Int32-typed slot loads INFALLIBLE
// (branch-free i32.wrap) -- sound when the slot is monomorphically int32 (e.g.
// gbemu's CPU-register object properties this.registerA/B/...). The per-access
// unbox branch is the dominant per-op cost on register-heavy code (it blocks the
// host from keeping the value in a register); dropping it is the real codegen win.
static bool WJSlotFallible(jit::MIRType type, jit::MUnbox::Mode mode) {
  // NOTE: forcing Int32/Boolean slot-unboxes always-fallible (tag-guard + deopt,
  // matching Ion's type-barrier) is now SOUND (deopt-resume IC-entry is aligned) and
  // no longer catastrophically regresses -- but it fixed no bench (box2d's truncation
  // is in arith, not slot-loads) while adding a tag-check to every Int32 slot load,
  // so it stays opt-in via GECKO_WJ_SLOTGUARD. The demonstrated pdfjs/box2d-correctness
  // case is already covered by the mega-load tag-guard (EmitHelperResultAsType guardTag).
  if (WJGETENV("GECKO_WJ_SLOTGUARD") &&
      (type == jit::MIRType::Int32 || type == jit::MIRType::Boolean)) {
    return true;
  }
  bool fallible = mode == jit::MUnbox::Fallible;
  if (!fallible) return false;
  static int infSlot = WJGETENV("GECKO_WJ_INFALLIBLESLOT") ? 1 : 0;
  if (infSlot && type == jit::MIRType::Int32) return false;
  return true;
}

// A value of this MIR type may be (or may box) a GC pointer that a post-write
// barrier must record when stored into a tenured container.
static bool WJTypeMaybeGC(jit::MIRType t) {
  switch (t) {
    case jit::MIRType::Value:
    case jit::MIRType::Object:
    case jit::MIRType::String:
    case jit::MIRType::Symbol:
    case jit::MIRType::BigInt:
      return true;
    default:
      return false;
  }
}

struct WJBackend {
  std::vector<int32_t> localOf;  // def id -> wasm local index (-1 = none)
  std::vector<uint8_t> localTy;  // valtype per declared local (index order)
  uint32_t paramCount = 1;       // wasm param 0 = scratch base address (f64)
  uint32_t scratchBase = 0;      // local holding the scratch base as i32
  uint32_t unboxScratch = 0;     // i64 local for unboxing fused slot loads
  uint32_t instofProto = 0;      // i32 DEDICATED: current proto in the multi-hop instanceof walk
  uint32_t instofCount = 0;      // i32 DEDICATED: hop counter for the multi-hop instanceof walk
  uint32_t instofResult = 0;     // i32 DEDICATED: multi-hop instanceof result (0/1)
  uint32_t iterIdx = 0;          // i32 DEDICATED: LoadSlotByIteratorIndex extracted PropertyIndex index
  uint32_t iterKind = 0;         // i32 DEDICATED: LoadSlotByIteratorIndex extracted PropertyIndex kind
  uint32_t iterAddr = 0;         // i32 DEDICATED: StoreSlotByIteratorIndex computed store address
  uint32_t dynArgAcc = 0;        // i64 DEDICATED accumulator for dynamic GetFrameArgument's
  uint32_t dynArgIdx = 0;        // i32 DEDICATED index  -- must NOT reuse unboxScratch/
                                 // propShapeLocal: a dyn arguments[i] can be a sub-operand
                                 // of a parent op that holds those scratch locals live.
  int32_t argCopyLocal[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
                                 // per-arg-index GC-ROOTED copy local (or -1). GetFrameArgument
                                 // MUST read this, not the raw wasm param -- the raw param goes
                                 // stale when a GC moves an object arg (the copy is reloaded by
                                 // WJCollectRoots). Populated at the prologue param-copy spill.
  bool needsArgRoots = false;    // has a DYNAMIC GetFrameArgument -> pure-variadic fns have no
                                 // formal Parameter copy locals, so root all kWJMaxArgs ACTUALS
                                 // in gWJCallRoots shadow slots (after env/callee) at prologue;
                                 // frame-arg reads them from the traced slots (GC-current).
                                 // Forces useEnvRoot (the arg slots sit above envRootIdx and are
                                 // popped by EmitEnvRootPop). Only live when the fn compiles
                                 // (dyn GetFrameArgument bails to PBL unless GECKO_WJ_DYNFRAMEARG).
  uint32_t callScratch = 0;      // i32 local: callee ptr for the call-site IC
  uint32_t callArgBase = 0;      // first of (1+kWJMaxArgs) i64 locals: boxed this+args
  uint32_t callFlagLocal = 0;    // f64 local: deopt flag returned by a call
  uint32_t callResultLocal = 0;  // i64 local: boxed result returned by a call
  uint32_t envLocal = 0;         // i32 local: this invocation's environment object,
                                 // saved from gWJCurrentEnv in the prologue (sound
                                 // even across nested calls that clobber the global)
  bool useEnvRoot = false;       // true if any FunctionEnvironment node exists -> the
                                 // env object must be GC-rooted for the function's
                                 // whole lifetime (it's a movable nursery CallObject
                                 // for closures; a stale raw ptr reads garbage ->
                                 // navier/earley/splay "X is not a function"/OOB)
  uint32_t envRootIdx = 0;       // i32 local: index of the PERSISTENT GC-shadow-stack
                                 // slot (gWJCallRoots) holding the boxed env for the
                                 // whole invocation. FunctionEnvironment rematerializes
                                 // from it on every use (traced every GC -> always
                                 // current, even across non-call GC points). Pushed in
                                 // the prologue, popped at every exit.
  const CompileInfo* info = nullptr;  // slot layout for resume points
  MResumePoint* curRp = nullptr;      // resume point for the instruction being emitted
  std::unordered_map<MBasicBlock*, MResumePoint*> blockExitRp;  // each block's exit rp (rp-fallback for null entryResumePoint)
  std::unordered_map<uint32_t, uintptr_t> nurseryObjSlot;  // NurseryObject node id -> traced const-pool slot (rematerialize, never cache: a minor GC moves the object, the slot is traced/current, a cached local goes stale -- the deltablue-under-inlining corruption)
  uint32_t curOp = 0;                 // op() of the instruction being emitted (deopt histogram)
  uint32_t deoptCallLine = 0;         // __LINE__ of the EmitDeopt call (deopt-line histogram)
  uint32_t propReadIdx = 0;           // per-compile mega-read counter (stable prop-site key)
  WarpSnapshot* snapshot = nullptr;   // for resolving NurseryObject references

  // Precise-rooting dataflow liveness cache (GECKO_WJ_NOROOTALL path only; default
  // rootAll=1 never touches this). wjLiveOut[B] = the local-backed Object/Value defs
  // (the GetOp-root-set elements) live at B's exit. Computed once per function by a
  // proper backward fixpoint (handles loops, unlike the prior ad-hoc per-safepoint
  // use-checks). See WJComputeLiveness.
  bool wjLiveDone = false;
  std::unordered_map<const jit::MBasicBlock*,
                     std::unordered_set<const jit::MDefinition*>>
      wjLiveOut;

  // Out-of-line deopt (the dispatch-loop path). Each guard miss inside a hot
  // block body branches to a single out-of-line dispatcher (the `block $D` that
  // wraps the dispatch loop) instead of inlining the big spill+resume sequence --
  // keeping the hot loop body tiny so the host VM register-allocates/optimizes it
  // (inline deopt bloat made navier's inner loop ~90% never-taken spill code and
  // ~3.6x slower). The dispatcher, emitted after all bodies, switches on
  // deoptSiteLocal and runs the matching resume. Only active on the dispatch path.
  bool oolDeopt = false;          // out-of-line mode enabled
  bool hasCatchTrynote = false;   // try/catch: the script has a Catch/Finally trynote
                                  // (only compiled when GECKO_WJ_NOTRYBAIL removes the
                                  // bail). Routes call/helper exceptions through
                                  // deopt-in-error so PBL's HandleException runs the
                                  // catch (Warp skips catch-block MIR).
  bool hasInlinedFrames = false;  // this compile contains Warp-inlined callees (a
                                  // multi-frame resume point exists). The fast
                                  // wasm->wasm call_indirect path corrupts state on a
                                  // deopt nested under an inlined caller (no C++
                                  // boundary for the resume); route this function's
                                  // calls through the slow (wjhelp, clean-boundary)
                                  // path, which is deopt-correct. Targeted vs global
                                  // FORCESLOWCALL: only inlining-containing functions.
  bool forceMega = false;         // megamorphic recompile: GuardShape-guarded
                                  // property reads -> multi-shape EmitPropIC (no
                                  // deopt) instead of a monomorphic GuardShape
                                  // that storms on a polymorphic receiver.
  bool inDispatchBody = false;    // currently emitting a hot dispatch block body
  bool inReloopBody = false;      // currently emitting a relooper (WJEmitStructured) block
                                  // body -- deopt-resume into a reloop'd switch is the
                                  // unfixed #16 miscompile, so ops that would add a NEW
                                  // deopt site here must bail the whole fn to PBL.
  uint32_t bodyLoopIdx = 0;       // br-index of loop $L from the current body base
  bool reloopHasCalls = false;    // caller hint: the function makes calls (-> use OOL
                                  // deopt in the relooper to keep call-heavy bodies lean)
  bool reloopOol = false;         // OOL deopt is active inside the relooper (block
                                  // $D wraps the structured body). When set, a guard
                                  // miss brs to $D at depth reloopScopeDepth+1+nest
                                  // instead of the dispatch-loop's bodyLoopIdx+2.
  uint32_t reloopScopeDepth = 0;  // # of open relooper scopes (loops/blocks) enclosing
                                  // the block body currently being emitted (excludes $D
                                  // and the guard's own `if`).
  uint32_t deoptExtraNest = 0;    // extra `if`/block levels enclosing a deopt vs the
                                  // usual single guard-`if` (so the OOL br reaches $D)
  uint32_t deoptSiteLocal = 0;    // i32 local: which deopt site fired
  uint32_t propObjLocal = 0;      // i32 local: prop-IC receiver pointer
  uint32_t propShapeLocal = 0;    // i32 local: prop-IC receiver shape
  uint32_t propTaggedLocal = 0;   // i32 local: prop-IC tagged slot offset
  uint32_t propBVKeyLocal = 0;    // i32 local: by-value mega-probe runtime key (atom ptr)
  uint32_t propKeyLocal = 0;      // i64 local: element-store IC boxed key value
  uint32_t allocPosLocal = 0;     // i32 local: inline-alloc old nursery position
  uint32_t allocObjLocal = 0;     // i32 local: inline-alloc result object pointer
  uint32_t rootBaseLocal = 0;     // i32 local: GC-root spill slot base (sp*8+base), computed once per call
  uint32_t savedDepthLocal = 0;   // i32 local: gWJJitDepth value at fn entry (restored at every exit)
  uint32_t actualArgcLocal = 0;   // i32 local: this invocation's actual arg count, snapshotted from
                                  // gWJCallArgc at entry (for ArgumentsLength etc.); valid iff usesArgc
  bool usesArgc = false;          // graph reads the actual arg count (ArgumentsLength) -> snapshot gWJCallArgc at entry
  uint32_t gggAddr = 0;           // GuardGlobalGeneration: generation counter addr (0 = none)
  uint32_t gggExpected = 0;       // ... and the compile-time expected generation
  bool needsArgcGuard = false;    // graph has a variadic op (CreateArgumentsObject/Rest) whose
                                  // staging only covers kWJMaxArgs actuals -> emit an entry
                                  // `if (actualArgc > kWJMaxArgs) return flag-2 (did-not-run)`
                                  // so the body always sees argc<=8 and the op is correct.
  bool usesCallee = false;        // OUTER frame has MCallee -> push the RUNTIME callee (gWJCallCallee)
                                  // into the root slot after the env (envRootIdx+1); MCallee reads it.
                                  // Baking the script's canonical function is WRONG for closures
                                  // sharing a script (acorn-walk NFE self-recursion read a garbage env).
  bool depthChecked = false;      // did the entry block emit the recursion save/check? (gates the restore)
  bool hasCalls = false;          // graph contains a call (can recurse) -> needs the recursion guard; leaves skip it
  uint32_t frameBytesEst = 128;   // estimated V8 wasm frame size (bytes) for the recursion BYTE-guard: (params+valuelocals+dedicated)*8 + overhead. Approximates real stack usage so the guard fires by BYTES (like a real SP check), not by a flat frame count that ignores frame size.
  // (rp, curOp) per out-of-line deopt site.
  std::vector<std::pair<MResumePoint*, uint32_t>> deoptSites;

  // Rematerialization cache (GECKO_WJ_REMATCACHE): GetOp re-derives Unbox/GuardShape
  // pointers on EVERY use (for GC-currency). Within a region with no MOVING GC point
  // (calls/allocs -- barriers don't move), the re-derived value is stable, so cache
  // it in a local and reuse. Invalidated at every moving GC point + block boundary.
  // Ion keeps such values live in registers across the loop; this approximates that.
  std::unordered_map<uint32_t, uint32_t> rematCacheLocal;  // def id -> cache local
  std::unordered_set<uint32_t> rematValid;                 // def ids currently cached
  bool rematCacheOn = false;
  void rematInvalidate() { rematValid.clear(); }

  int32_t local(const MDefinition* d) const {
    uint32_t id = d->id();
    return id < localOf.size() ? localOf[id] : -1;
  }
  void ensureSize(uint32_t id) {
    if (id >= localOf.size()) localOf.resize(id + 1, -1);
  }
  int32_t reserve(uint8_t ty) {
    int32_t idx = int32_t(paramCount + localTy.size());
    localTy.push_back(ty);
    return idx;
  }
  int32_t assign(const MDefinition* d) {
    ensureSize(d->id());
    if (localOf[d->id()] >= 0) return localOf[d->id()];
    uint8_t ty = WJValType(d->type());
    int32_t idx = reserve(ty);
    localOf[d->id()] = idx;
    return idx;
  }
};

static bool GetOpInner(Encoder& e, WJBackend& be, const MDefinition* d);
static bool WJCanGetOp(WJBackend& be, const MDefinition* d);
static bool EmitInlinedFEEnv(Encoder& e, WJBackend& be, const MDefinition* feNode);

// Remat cache wrapper (GECKO_WJ_REMATCACHE). For cacheable re-derived pointer defs
// (Unbox-to-ptr, GuardShape/List passthroughs), reuse a cached local when still
// valid (no moving GC since materialization); otherwise derive via GetOpInner and
// tee the result into the cache local. Invalidated at moving GC points (calls/
// allocs) + block boundaries -- barriers are non-moving so the value survives them
// (which TurboFan can't assume across the opaque call). All callers use GetOp; the
// recursive derivations inside GetOpInner also route here, so caching composes.
static bool GetOp(Encoder& e, WJBackend& be, const MDefinition* d) {
  if (!be.rematCacheOn) return GetOpInner(e, be, d);
  auto op = d->op();
  bool cacheable =
      (op == MDefinition::Opcode::Unbox &&
       (d->type() == MIRType::Object || d->type() == MIRType::String ||
        d->type() == MIRType::Symbol || d->type() == MIRType::BigInt)) ||
      op == MDefinition::Opcode::GuardShape ||
      op == MDefinition::Opcode::GuardShapeList;
  if (!cacheable) return GetOpInner(e, be, d);
  uint32_t id = d->id();
  if (be.rematValid.count(id)) {
    return e.writeOp(Op::LocalGet) && e.writeVarU32(be.rematCacheLocal[id]);
  }
  if (!GetOpInner(e, be, d)) return false;
  uint32_t loc;
  auto it = be.rematCacheLocal.find(id);
  if (it != be.rematCacheLocal.end()) {
    loc = it->second;
  } else {
    loc = uint32_t(be.reserve(WJValType(d->type())));
    be.rematCacheLocal[id] = loc;
    if (WJGETENV("GECKO_WJ_REMATDBG"))
      fprintf(stderr, "[rematdbg] mid-emit reserve id=%u op=%u ty=%u loc=%u\n",
              id, unsigned(d->op()), unsigned(d->type()), loc);
  }
  be.rematValid.insert(id);
  return e.writeOp(Op::LocalTee) && e.writeVarU32(loc);
}

static bool GetOpInner(Encoder& e, WJBackend& be, const MDefinition* d) {
  // Slots/Elements are RAW INTERIOR pointers into an object's heap storage. A
  // minor GC that moves the owning object invalidates them, so they must NEVER
  // be read from a cached local that may predate a GC-causing call -- that stale
  // pointer reads garbage (the raytrace `ray`-goes-garbage crash). Rematerialize
  // them on every use from the owning object's local (an Object, GC-rooted and
  // reloaded across calls, hence always current). This mirrors Ion, which treats
  // MElements/MSlots as rematerializable rather than keeping them live in regs.
  // Slots/Elements are RAW INTERIOR pointers; a minor GC moving the owning object
  // invalidates a cached copy. Rematerialize from the (GC-rooted) owning object's
  // local on every use -- sound, and measured no slower than caching (the array
  // overhead is bounds-checks/unboxing, not this load).
  if (d->type() == MIRType::Slots) {
    // Read the owning object via GetOp too (it may be an Unbox/GuardShape whose
    // cached raw ptr is stale across a GC-ing call) -- rematerialize the whole
    // chain from the rooted source.
    if (!GetOp(e, be, d->toSlots()->object())) return false;
    return e.writeOp(Op::I32Load) && e.writeVarU32(2) &&
           e.writeVarU32(uint32_t(js::NativeObject::offsetOfSlots()));
  }
  if (d->type() == MIRType::Elements) {
    // Typed-array / DataView data pointer: a PrivateValue in fixed slot DATA_SLOT.
    // On NUNBOX32 a PrivateValue stores the raw pointer as asBits_ (no tag/shift),
    // so the wasm32 data pointer is the low 32 bits at dataOffset() -- a plain
    // 32-bit load. Rematerialized (not cached) so it stays current if a minor GC
    // moves the owning view (same hazard as MElements).
    if (d->isArrayBufferViewElements()) {
      if (!GetOp(e, be, d->toArrayBufferViewElements()->object())) return false;
      return e.writeOp(Op::I32Load) && e.writeVarU32(2) &&
             e.writeVarU32(uint32_t(js::ArrayBufferViewObject::dataOffset()));
    }
    if (!GetOp(e, be, d->toElements()->object())) return false;
    return e.writeOp(Op::I32Load) && e.writeVarU32(2) &&
           e.writeVarU32(uint32_t(js::NativeObject::offsetOfElements()));
  }
  // Callee (outer frame): rematerialize the RUNTIME callee from its persistent
  // root slot (gWJCallRoots[envRootIdx+1], traced) on every use -- a cached raw
  // function pointer would go stale across a moving GC.
  if (d->op() == MDefinition::Opcode::Callee && be.usesCallee && be.useEnvRoot &&
      d->block()->info().script() == be.info->script()) {
    uintptr_t rootsBase = uintptr_t(static_cast<void*>(&gWJCallRoots[0]));
    return e.writeOp(Op::LocalGet) && e.writeVarU32(be.envRootIdx) &&
           e.writeOp(Op::I32Const) && e.writeVarS32(8) && e.writeOp(Op::I32Mul) &&
           e.writeOp(Op::I32Const) && e.writeVarS32(int32_t(rootsBase)) &&
           e.writeOp(Op::I32Add) && e.writeOp(Op::I64Load) && e.writeVarU32(3) &&
           e.writeVarU32(8) && e.writeOp(Op::I32WrapI64);
  }
  // FunctionEnvironment: rematerialize the env from its PERSISTENT GC-shadow-stack
  // slot (gWJCallRoots[envRootIdx]) on every use -- the slot is traced on every GC
  // so this is always the GC-current env, even across non-call GC points (where a
  // cached raw ptr would go stale: navier/earley closure bugs). Unbox the boxed
  // ObjectValue (low 32 bits = object payload, NUNBOX32).
  if (d->op() == MDefinition::Opcode::FunctionEnvironment) {
    // Discriminate by the FE's FUNCTION operand, like Ion's LFunctionEnvironment:
    //  - operand == MCallee (the running outer function): its env is the runtime env
    //    snapshotted at entry (envRootIdx / envLocal).
    //  - any OTHER function (an inlined callee, or an outer-frame GuardSpecificFunction
    //    referencing a specific closure -- navier): load THAT function's env slot
    //    GC-fresh (EmitInlinedFEEnv). The guard/identity fixes which function it is.
    // (Was keyed on the `inlined` flag, which missed outer-frame guarded-fn FEs.)
    const MDefinition* feFn = d->toFunctionEnvironment()->function();
    if (!feFn->isCallee()) return EmitInlinedFEEnv(e, be, d);
    if (be.useEnvRoot) {
      uintptr_t rootsBase = uintptr_t(static_cast<void*>(&gWJCallRoots[0]));
      return e.writeOp(Op::LocalGet) && e.writeVarU32(be.envRootIdx) &&
             e.writeOp(Op::I32Const) && e.writeVarS32(8) && e.writeOp(Op::I32Mul) &&
             e.writeOp(Op::I32Const) && e.writeVarS32(int32_t(rootsBase)) &&
             e.writeOp(Op::I32Add) && e.writeOp(Op::I64Load) && e.writeVarU32(3) &&
             e.writeVarU32(0) && e.writeOp(Op::I32WrapI64);
    }
  }
  // Parameter: read the GC-ROOTED COPY local, NOT the raw wasm param. The raw
  // param holds a boxed object payload that goes STALE when a minor GC moves the
  // arg object (e.g. earley/nboyer pass freshly-consed nursery lists, then `new
  // sc_Pair` triggers a minor GC, then read `.car` on the now-moved arg -> memory
  // out of bounds). The raw param is never reloaded, but the copy local IS
  // reloaded across every call/GC point by WJCollectRoots, and it's initialized
  // at the top of the prologue (before any block-entry spill or deopt), so it is
  // always GC-current. Fall back to the raw param only if there's no copy local
  // (an unused param, never actually read here).
  if (d->op() == MDefinition::Opcode::Parameter) {
    const MParameter* p = d->toParameter();
    if (p->index() != MParameter::THIS_SLOT &&
        uint32_t(p->index()) >= kWJMaxArgs)
      return false;
    int32_t cl = be.local(d);
    if (cl >= 0) return e.writeOp(Op::LocalGet) && e.writeVarU32(uint32_t(cl));
    uint32_t pidx = (p->index() == MParameter::THIS_SLOT)
                        ? kWJThisParam
                        : (kWJArg0Param + uint32_t(p->index()));
    return e.writeOp(Op::LocalGet) && e.writeVarU32(pidx);
  }
  // Unbox to a GC-POINTER type (Object/String/Symbol/BigInt): the unboxed value
  // is a RAW heap pointer. Caching it in a local makes it go STALE across a
  // GC-causing call (the boxed source is rooted+reloaded, but the cached raw
  // pointer is not) -- e.g. earley sc_reverse: `res=sc_cons(l1.car,res); l1=l1.cdr`
  // re-used the pre-call Unbox(l1) pointer after sc_cons's nursery alloc moved
  // l1, reading l1.cdr from freed memory -> "car of undefined". So REMATERIALIZE
  // by re-unboxing the (rooted, GC-current) source on every use, mirroring how
  // Slots/Elements are rematerialized. The fallible type guard already ran when
  // the Unbox node was first emitted; here we only need the (current) pointer.
  if (d->op() == MDefinition::Opcode::Unbox &&
      (d->type() == MIRType::Object || d->type() == MIRType::String ||
       d->type() == MIRType::Symbol || d->type() == MIRType::BigInt)) {
    MDefinition* src = d->toUnbox()->input();
    if (src->type() == MIRType::Value || src->type() == MIRType::Int64) {
      return GetOp(e, be, src) && e.writeOp(Op::I32WrapI64);  // boxed -> ptr
    }
    if (src->type() == MIRType::Object || src->type() == MIRType::String ||
        src->type() == MIRType::Symbol || src->type() == MIRType::BigInt) {
      return GetOp(e, be, src);  // src already a (rooted/reloaded) raw ptr
    }
    // other src types: fall through to the cached local. (DEAD for uglify -- measured
    // 0 hits: all Object-Unbox there have a Value src -> the remat branch above.)
  }
  // GuardShape (and friends) are PASSTHROUGHs: their value is the guarded object.
  // Rematerialize via GetOp(object) so a post-call use re-derives from the rooted
  // source (same stale-pointer hazard as Unbox -- earley sc_reverse l1.cdr). The
  // guard itself already ran when the node was emitted.
  if (d->op() == MDefinition::Opcode::GuardShape)
    return GetOp(e, be, d->toGuardShape()->object());
  if (d->op() == MDefinition::Opcode::GuardShapeList)
    return GetOp(e, be, d->toGuardShapeList()->object());
  // Object passthroughs (return their operand 0 unchanged after a guard): also
  // rematerialize from the rooted source so a post-call use isn't a stale ptr.
  if (d->op() == MDefinition::Opcode::GuardSpecificFunction ||
      d->op() == MDefinition::Opcode::GuardToFunction ||
      d->op() == MDefinition::Opcode::GuardFunctionScript ||
      d->op() == MDefinition::Opcode::ConstantProto)
    return GetOp(e, be, d->getOperand(0));
  if (d->op() == MDefinition::Opcode::GuardObjectIdentity)
    return GetOp(e, be, d->toGuardObjectIdentity()->object());
  // NurseryObject: a snapshot-held (movable) object reference. Its boxed pointer is
  // kept current in a TRACED const-pool slot; reload it from that slot on EVERY use
  // (never the cached local, which goes stale when a minor GC moves the object --
  // the deltablue-under-inlining corruption: inlined callees hold NurseryObject refs
  // like `planner`/`Direction`/prototypes live across allocations).
  if (d->op() == MDefinition::Opcode::NurseryObject) {
    auto it = be.nurseryObjSlot.find(d->id());
    if (it != be.nurseryObjSlot.end()) {
      return e.writeOp(Op::I32Const) && e.writeVarS32(int32_t(it->second)) &&
             e.writeOp(Op::I64Load) && e.writeVarU32(3) && e.writeVarU32(0) &&
             e.writeOp(Op::I32WrapI64);
    }
  }
  // Constant Object: bake-as-immediate is STALE across a moving GC (the object
  // relocates; the cached local / raw immediate is not updated). Load from the
  // GC-TRACED intern pool slot on every use, mirroring EmitObjPtr/EmitSpillValue.
  // (richards' `this.addTask` proto read: callee = LoadDynamicSlotAndUnbox <- Slots
  // <- GuardShape <- Constant(Scheduler.prototype); a major GC moved the prototype,
  // the cached-local proto base went stale, so the slot read returned the FREED old
  // addTask pointer -> "null function". This was the forceMega-off keystone crash.)
  if (d->isConstant() && d->type() == MIRType::Object) {
    if (WJGETENV("GECKO_WJ_CONSTGETOP") && gWJCurScriptLine ==
        uint32_t(atoi(WJGETENV("GECKO_WJ_CONSTGETOP"))))
      fprintf(stderr, "[constgetop] line=%u GetOp(Constant Object) hit\n", gWJCurScriptLine);
    JSObject* obj = d->toConstant()->toObjectOrNull();
    if (!obj) return e.writeOp(Op::I32Const) && e.writeVarS32(0);  // null object
    uintptr_t slot = WJInternConstant(JS::ObjectValue(*obj).asRawBits());
    if (!slot) return false;
    return e.writeOp(Op::I32Const) && e.writeVarS32(int32_t(slot)) &&
           e.writeOp(Op::I64Load) && e.writeVarU32(3) && e.writeVarU32(0) &&
           e.writeOp(Op::I32WrapI64);
  }
  int32_t l = be.local(d);
  if (l < 0) return false;
  return e.writeOp(Op::LocalGet) && e.writeVarU32(uint32_t(l));
}

// Faithful mirror of GetOp's SUCCESS predicate (no emission). Used to decide
// whether an OPTIONAL GetOp (e.g. the valNursery post-barrier refinement) is safe
// to emit -- if not, the caller falls back rather than bailing the whole function.
// Keep in lockstep with GetOp above.
static bool WJCanGetOp(WJBackend& be, const MDefinition* d) {
  using Op_ = MDefinition::Opcode;
  if (d->type() == MIRType::Slots) return WJCanGetOp(be, d->toSlots()->object());
  if (d->type() == MIRType::Elements) {
    if (d->isArrayBufferViewElements())
      return WJCanGetOp(be, d->toArrayBufferViewElements()->object());
    return WJCanGetOp(be, d->toElements()->object());
  }
  if (d->op() == Op_::FunctionEnvironment && be.useEnvRoot) return true;
  if (d->op() == Op_::Parameter) {
    const MParameter* p = d->toParameter();
    return p->index() == MParameter::THIS_SLOT ||
           uint32_t(p->index()) < kWJMaxArgs;
  }
  if (d->op() == Op_::Unbox &&
      (d->type() == MIRType::Object || d->type() == MIRType::String ||
       d->type() == MIRType::Symbol || d->type() == MIRType::BigInt)) {
    MDefinition* src = d->toUnbox()->input();
    if (src->type() == MIRType::Value || src->type() == MIRType::Int64 ||
        src->type() == MIRType::Object || src->type() == MIRType::String ||
        src->type() == MIRType::Symbol || src->type() == MIRType::BigInt)
      return WJCanGetOp(be, src);
    // other src types fall through to the cached local.
  }
  if (d->op() == Op_::GuardShape)
    return WJCanGetOp(be, d->toGuardShape()->object());
  if (d->op() == Op_::GuardShapeList)
    return WJCanGetOp(be, d->toGuardShapeList()->object());
  if (d->op() == Op_::GuardSpecificFunction || d->op() == Op_::GuardToFunction ||
      d->op() == Op_::GuardFunctionScript || d->op() == Op_::ConstantProto)
    return WJCanGetOp(be, d->getOperand(0));
  if (d->op() == Op_::GuardObjectIdentity)
    return WJCanGetOp(be, d->toGuardObjectIdentity()->object());
  if (d->op() == Op_::NurseryObject)
    return be.nurseryObjSlot.find(d->id()) != be.nurseryObjSlot.end();
  return be.local(d) >= 0;
}

// Env for an INLINED FunctionEnvironment (COLDCALL inlining). The FE's function operand
// is the inlined callee, reached via GuardSpecificFunction/GuardFunctionScript (a
// passthrough of the runtime callee that the guard proved == a specific function). Load
// THAT function's environment fixed slot (a boxed Value) + unbox -> the inlined closure's
// env. Sound: the guard fixed the function identity; env is its own field, re-loaded
// GC-fresh on every use (GetOp rematerializes the funcptr). Returns false (bail) if the
// function operand isn't a rematerializable pointer (then the whole fn bails to PBL, as
// before). This is what made COLDCALL regress ubo: 242 inlined-FE bails -> PBL.
static bool EmitInlinedFEEnv(Encoder& e, WJBackend& be, const MDefinition* feNode) {
  const MDefinition* fn = feNode->toFunctionEnvironment()->function();
  if (fn->type() != MIRType::Object && fn->type() != MIRType::Value) return false;
  if (!WJCanGetOp(be, fn)) return false;
  uint32_t envOff = uint32_t(JSFunction::offsetOfEnvironment());
  if (!GetOp(e, be, fn)) return false;
  if (fn->type() == MIRType::Value && !e.writeOp(Op::I32WrapI64)) return false;  // fn ptr
  // env slot holds a boxed ObjectValue (or null) -> load i64, unbox to object ptr.
  return e.writeOp(Op::I64Load) && e.writeVarU32(3) && e.writeVarU32(envOff) &&
         e.writeOp(Op::I32WrapI64);
}

// Sound deopt: spill the current resume point's live state and continue in PBL
// (defined below). Returns false if it can't be emitted (caller bails the fn).
static bool EmitDeoptResume(Encoder& e, WJBackend& be);
static bool EmitValueTruthy(Encoder& e, WJBackend& be, uint32_t v);
static bool EmitEnvRootPop(Encoder& e, WJBackend& be);
static bool EmitThrowWjexn(Encoder& e);  // EHABI: throw the $wjexn tag
static bool EmitDepthPop(Encoder& e, WJBackend& be);  // restore gWJJitDepth (recursion byte-guard)

// Emit a deopt at the current instruction's resume point. All guard misses route
// here; if there's no usable resume point the whole function bails at compile.
static bool EmitDeoptImpl(Encoder& e, WJBackend& be) { return EmitDeoptResume(e, be); }
#define EmitDeopt(e, be) (((be).deoptCallLine = __LINE__), EmitDeoptImpl((e), (be)))

// NUNBOX32 tag word (high 32 bits) for a non-double value type.
static uint32_t TagWord(JSValueType t) {
  return uint32_t(JSVAL_TAG_CLEAR) | uint32_t(t);
}

// Box a typed value already on the stack into an i64 JS::Value.
static bool EmitBoxFromStack(Encoder& e, MIRType from) {
  if (from == MIRType::Double) {
    return e.writeOp(Op::I64ReinterpretF64);
  }
  if (from == MIRType::Int32 || from == MIRType::Boolean ||
      from == MIRType::Object || from == MIRType::String ||
      from == MIRType::Symbol || from == MIRType::BigInt) {
    JSValueType vt = from == MIRType::Int32     ? JSVAL_TYPE_INT32
                     : from == MIRType::Boolean ? JSVAL_TYPE_BOOLEAN
                     : from == MIRType::String  ? JSVAL_TYPE_STRING
                     : from == MIRType::Symbol  ? JSVAL_TYPE_SYMBOL
                     : from == MIRType::BigInt  ? JSVAL_TYPE_BIGINT
                                                : JSVAL_TYPE_OBJECT;
    // payload(i32) -> i64 zero-extended, OR'd with (tag << 32).
    if (!e.writeOp(Op::I64ExtendI32U)) return false;
    if (!e.writeOp(Op::I64Const) ||
        !e.writeVarS64(int64_t(uint64_t(TagWord(vt)) << 32))) {
      return false;
    }
    return e.writeOp(Op::I64Or);
  }
  return false;
}

// Push the i64 value held in wasm local `localIdx`.
static bool GetLocal(Encoder& e, uint32_t localIdx) {
  return e.writeOp(Op::LocalGet) && e.writeVarU32(localIdx);
}

// Guard that the i64 value in wasm local `localIdx` has tag word `tag`; deopt if not.
static bool EmitTagGuardLocal(Encoder& e, WJBackend& be, uint32_t localIdx,
                              uint32_t tag) {
  if (!GetLocal(e, localIdx)) return false;
  if (!e.writeOp(Op::I64Const) || !e.writeVarS64(32)) return false;
  if (!e.writeOp(Op::I64ShrU)) return false;
  if (!e.writeOp(Op::I32WrapI64)) return false;
  if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(tag))) return false;
  if (!e.writeOp(Op::I32Ne)) return false;
  static int noBranch = WJGETENV("GECKO_WJ_GUARD_NOBRANCH") ? 1 : 0;
  if (noBranch) return e.writeOp(Op::Drop);  // PROBE: compute cond, no exit
  if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
  static int unboxDbg = WJGETENV("GECKO_WJ_UNBOXDBG") ? 1 : 0;
  if (unboxDbg) {
    // gWJUnboxTagHist[(local>>32)&15]++  ; gWJUnboxExpectTag = tag
    uintptr_t hb = uintptr_t(static_cast<void*>(&gWJUnboxTagHist[0]));
    uintptr_t eb = uintptr_t(static_cast<void*>(&gWJUnboxExpectTag));
    auto pushAddr = [&]() -> bool {  // base + ((local>>32)&15)<<2
      return GetLocal(e, localIdx) && e.writeOp(Op::I64Const) && e.writeVarS64(32) &&
             e.writeOp(Op::I64ShrU) && e.writeOp(Op::I32WrapI64) &&
             e.writeOp(Op::I32Const) && e.writeVarS32(15) && e.writeOp(Op::I32And) &&
             e.writeOp(Op::I32Const) && e.writeVarS32(2) && e.writeOp(Op::I32Shl) &&
             e.writeOp(Op::I32Const) && e.writeVarS32(int32_t(hb)) && e.writeOp(Op::I32Add);
    };
    if (!pushAddr() || !pushAddr() || !e.writeOp(Op::I32Load) || !e.writeVarU32(2) ||
        !e.writeVarU32(0) || !e.writeOp(Op::I32Const) || !e.writeVarS32(1) ||
        !e.writeOp(Op::I32Add) || !e.writeOp(Op::I32Store) || !e.writeVarU32(2) ||
        !e.writeVarU32(0))
      return false;
    if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(eb)) ||
        !e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(tag)) ||
        !e.writeOp(Op::I32Store) || !e.writeVarU32(2) || !e.writeVarU32(0))
      return false;
  }
  if (!EmitDeopt(e, be)) return false;
  return e.writeOp(Op::End);
}

// Guard that the i64 value in local `localIdx` is a double; deopt if not.
static bool EmitDoubleGuardLocal(Encoder& e, WJBackend& be, uint32_t localIdx) {
  if (!GetLocal(e, localIdx)) return false;
  if (!e.writeOp(Op::I64Const) || !e.writeVarS64(32)) return false;
  if (!e.writeOp(Op::I64ShrU)) return false;
  if (!e.writeOp(Op::I32WrapI64)) return false;
  if (!e.writeOp(Op::I32Const) ||
      !e.writeVarS32(int32_t(uint32_t(JSVAL_TAG_CLEAR)))) {
    return false;
  }
  if (!e.writeOp(Op::I32GtU)) return false;  // tag > CLEAR => not a double
  static int noBranch = WJGETENV("GECKO_WJ_GUARD_NOBRANCH") ? 1 : 0;
  if (noBranch) return e.writeOp(Op::Drop);  // PROBE: compute cond, no exit
  if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
  if (!EmitDeopt(e, be)) return false;
  return e.writeOp(Op::End);
}

// Relaxed fallible Int32 unbox (GECKO_WJ_NORELAXINT32 disables). Leaves the i32
// on the stack. Fast path: tag==INT32 -> payload. Else an integer-valued double
// in int32 range converts inline -- Ion parity for int32/double-unstable fields
// (e.g. gbemu flag/counter values stored as 1.0): without this, every such read
// tag-guard-deopts to PBL (the gbemu deopt storm). Non-number, non-integer, NaN,
// or out-of-range values still deopt (correct: a genuine Int32 type violation).
// localIdx is NOT mutated (it may be a live operand local, not scratch).
static bool EmitRelaxedInt32Unbox(Encoder& e, WJBackend& be, uint32_t localIdx) {
  const uint8_t kI32bt = 0x7F;
  auto tagOf = [&]() -> bool {
    return GetLocal(e, localIdx) && e.writeOp(Op::I64Const) && e.writeVarS64(32) &&
           e.writeOp(Op::I64ShrU) && e.writeOp(Op::I32WrapI64);
  };
  // tag == INT32 ?
  if (!tagOf() || !e.writeOp(Op::I32Const) ||
      !e.writeVarS32(int32_t(TagWord(JSVAL_TYPE_INT32))) || !e.writeOp(Op::I32Eq))
    return false;
  if (!e.writeOp(Op::If) || !e.writeFixedU8(kI32bt)) return false;
  if (!GetLocal(e, localIdx) || !e.writeOp(Op::I32WrapI64)) return false;  // payload
  if (!e.writeOp(Op::Else)) return false;
  // not int32: deopt if not a double (tag > CLEAR => some other non-number type)
  if (!tagOf() || !e.writeOp(Op::I32Const) ||
      !e.writeVarS32(int32_t(uint32_t(JSVAL_TAG_CLEAR))) || !e.writeOp(Op::I32GtU))
    return false;
  if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
  {
    uint32_t saved = be.deoptExtraNest;
    be.deoptExtraNest = saved + 1;  // deopt sits 2 scopes deep (else-arm + this if)
    bool ok = EmitDeopt(e, be);
    be.deoptExtraNest = saved;
    if (!ok) return false;
  }
  if (!e.writeOp(Op::End)) return false;
  // it's a double: deopt unless convert(truncsat(d)) == d (exact int in int32 range)
  if (!GetLocal(e, localIdx) || !e.writeOp(Op::F64ReinterpretI64) ||
      !e.writeOp(MiscOp::I32TruncSatF64S) || !e.writeOp(Op::F64ConvertI32S))
    return false;
  if (!GetLocal(e, localIdx) || !e.writeOp(Op::F64ReinterpretI64)) return false;
  if (!e.writeOp(Op::F64Ne)) return false;
  if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
  {
    uint32_t saved = be.deoptExtraNest;
    be.deoptExtraNest = saved + 1;
    bool ok = EmitDeopt(e, be);
    be.deoptExtraNest = saved;
    if (!ok) return false;
  }
  if (!e.writeOp(Op::End)) return false;
  // exact integer double: result = truncsat(d)
  if (!GetLocal(e, localIdx) || !e.writeOp(Op::F64ReinterpretI64) ||
      !e.writeOp(MiscOp::I32TruncSatF64S))
    return false;
  return e.writeOp(Op::End);  // end if/else -> leaves i32
}

// Unbox the i64 JS::Value held in `localIdx` to `out`, leaving the unboxed value
// on the stack. Fallible mode emits a tag guard (deopt on mismatch).
static bool EmitUnboxLocal(Encoder& e, WJBackend& be, uint32_t localIdx,
                           MIRType out, bool fallible) {
  switch (out) {
    case MIRType::Int32: {
      // Opt-in (GECKO_WJ_RELAXINT32): sound Ion-parity (integer-double -> int32
      // inline vs deopt) but measured neutral/slightly-negative so far (gbemu's
      // storm is NOT int32 tag guards -- see [[gbemu-deopt-storm]]). Kept for the
      // workload with genuine int32/double instability on a hot int32 unbox.
      static int relaxInt32 = WJGETENV("GECKO_WJ_RELAXINT32") ? 1 : 0;
      if (fallible && relaxInt32) return EmitRelaxedInt32Unbox(e, be, localIdx);
      if (fallible &&
          !EmitTagGuardLocal(e, be, localIdx, TagWord(JSVAL_TYPE_INT32)))
        return false;
      return GetLocal(e, localIdx) && e.writeOp(Op::I32WrapI64);
    }
    case MIRType::Boolean:
      if (fallible &&
          !EmitTagGuardLocal(e, be, localIdx, TagWord(JSVAL_TYPE_BOOLEAN)))
        return false;
      return GetLocal(e, localIdx) && e.writeOp(Op::I32WrapI64);
    case MIRType::Object:
      if (fallible &&
          !EmitTagGuardLocal(e, be, localIdx, TagWord(JSVAL_TYPE_OBJECT)))
        return false;
      return GetLocal(e, localIdx) && e.writeOp(Op::I32WrapI64);
    case MIRType::String:
      if (fallible &&
          !EmitTagGuardLocal(e, be, localIdx, TagWord(JSVAL_TYPE_STRING)))
        return false;
      return GetLocal(e, localIdx) && e.writeOp(Op::I32WrapI64);
    case MIRType::Symbol:
      if (fallible &&
          !EmitTagGuardLocal(e, be, localIdx, TagWord(JSVAL_TYPE_SYMBOL)))
        return false;
      return GetLocal(e, localIdx) && e.writeOp(Op::I32WrapI64);
    case MIRType::BigInt:
      if (fallible &&
          !EmitTagGuardLocal(e, be, localIdx, TagWord(JSVAL_TYPE_BIGINT)))
        return false;
      return GetLocal(e, localIdx) && e.writeOp(Op::I32WrapI64);
    case MIRType::Double: {
      static int infDbl = WJGETENV("GECKO_WJ_INFALLIBLE_DBL") ? 1 : 0;
      if (!fallible || infDbl) {
        return GetLocal(e, localIdx) && e.writeOp(Op::F64ReinterpretI64);
      }
      static int selUnbox = WJGETENV("GECKO_WJ_SELUNBOX") ? 1 : 0;
      if (selUnbox) {
        // BRANCHLESS number-unbox. One side-exit (deopt on NON-number only); the
        // double-vs-int32 split is a data-flow `select`, not control flow -- so the
        // hot loop has no nested branches and V8 can keep the FP value in a reg.
        // Sound: a number is either double (tag<=CLEAR, reinterpret) or int32
        // (convert); only genuinely non-number values deopt.
        // guard: deopt if tag > INT32 (skipped under SELNOGUARD probe -> zero exits)
        static int selNoGuard = WJGETENV("GECKO_WJ_SELNOGUARD") ? 1 : 0;
        if (!selNoGuard) {
          if (!GetLocal(e, localIdx) || !e.writeOp(Op::I64Const) ||
              !e.writeVarS64(32) || !e.writeOp(Op::I64ShrU) ||
              !e.writeOp(Op::I32WrapI64) || !e.writeOp(Op::I32Const) ||
              !e.writeVarS32(int32_t(TagWord(JSVAL_TYPE_INT32))) ||
              !e.writeOp(Op::I32GtU))
            return false;
          if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
          if (!EmitDeopt(e, be)) return false;
          if (!e.writeOp(Op::End)) return false;
        }
        // dblIfDouble = reinterpret(bits)
        if (!GetLocal(e, localIdx) || !e.writeOp(Op::F64ReinterpretI64))
          return false;
        // dblIfInt = convert(wrap(bits))
        if (!GetLocal(e, localIdx) || !e.writeOp(Op::I32WrapI64) ||
            !e.writeOp(Op::F64ConvertI32S))
          return false;
        // cond = tag <= CLEAR (isDouble)
        if (!GetLocal(e, localIdx) || !e.writeOp(Op::I64Const) ||
            !e.writeVarS64(32) || !e.writeOp(Op::I64ShrU) ||
            !e.writeOp(Op::I32WrapI64) || !e.writeOp(Op::I32Const) ||
            !e.writeVarS32(int32_t(uint32_t(JSVAL_TAG_CLEAR))) ||
            !e.writeOp(Op::I32LeU))
          return false;
        return e.writeOp(Op::SelectNumeric);
      }
      static int noNum = WJGETENV("GECKO_WJ_NONUMUNBOX") ? 1 : 0;
      if (noNum) {  // old: double-only guard, deopt on int32 (-> PBL fallback)
        if (!EmitDoubleGuardLocal(e, be, localIdx)) return false;
        return GetLocal(e, localIdx) && e.writeOp(Op::F64ReinterpretI64);
      }
      // NUMBER unbox, SINGLE-FAST-BRANCH. A Double-typed value that's actually an
      // Int32 is still a number -- convert it (no deopt) so navier's int32-0
      // boundary elements don't storm. The OLD code computed the tag 3x with a
      // separate guard + select (~20 ops/load) -- navier does 5 loads/iter x 1.35M
      // iter, and that tag-machinery is its DOMINANT cost (infallible unbox measured
      // 3.6x faster). New shape: HOT path = double -> reinterpret (one tag compare);
      // COLD else = int32 -> convert / non-number -> deopt, out-of-line so V8 keeps
      // the hot reinterpret path tight. NUNBOX tags: double has tag u<= CLEAR, int32
      // == INT32, everything else (non-number) > INT32. GECKO_WJ_NOFASTDBL reverts.
      const uint8_t kF64bt = 0x7C;  // wasm f64 blocktype
      if (WJGETENV("GECKO_WJ_NOFASTDBL")) {
        auto pushTag = [&]() -> bool {
          return GetLocal(e, localIdx) && e.writeOp(Op::I64Const) &&
                 e.writeVarS64(32) && e.writeOp(Op::I64ShrU) &&
                 e.writeOp(Op::I32WrapI64);
        };
        if (!pushTag() || !e.writeOp(Op::I32Const) ||
            !e.writeVarS32(int32_t(uint32_t(JSVAL_TAG_CLEAR))) ||
            !e.writeOp(Op::I32LeU) || !pushTag() || !e.writeOp(Op::I32Const) ||
            !e.writeVarS32(int32_t(TagWord(JSVAL_TYPE_INT32))) ||
            !e.writeOp(Op::I32Eq) || !e.writeOp(Op::I32Or) || !e.writeOp(Op::I32Eqz))
          return false;
        if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
        if (!EmitDeopt(e, be)) return false;
        if (!e.writeOp(Op::End)) return false;
        if (!pushTag() || !e.writeOp(Op::I32Const) ||
            !e.writeVarS32(int32_t(uint32_t(JSVAL_TAG_CLEAR))) ||
            !e.writeOp(Op::I32LeU) || !e.writeOp(Op::If) || !e.writeFixedU8(kF64bt))
          return false;
        if (!GetLocal(e, localIdx) || !e.writeOp(Op::F64ReinterpretI64)) return false;
        if (!e.writeOp(Op::Else)) return false;
        if (!GetLocal(e, localIdx) || !e.writeOp(Op::I32WrapI64) ||
            !e.writeOp(Op::F64ConvertI32S))
          return false;
        return e.writeOp(Op::End);
      }
      // tag = bits >> 32  (i32, on stack)
      if (!GetLocal(e, localIdx) || !e.writeOp(Op::I64Const) || !e.writeVarS64(32) ||
          !e.writeOp(Op::I64ShrU) || !e.writeOp(Op::I32WrapI64))
        return false;
      if (!e.writeOp(Op::I32Const) ||
          !e.writeVarS32(int32_t(uint32_t(JSVAL_TAG_CLEAR))) ||
          !e.writeOp(Op::I32LeU))
        return false;  // isDouble (common)
      if (!e.writeOp(Op::If) || !e.writeFixedU8(kF64bt)) return false;
      if (!GetLocal(e, localIdx) || !e.writeOp(Op::F64ReinterpretI64))
        return false;  // HOT: double -> reinterpret
      if (!e.writeOp(Op::Else)) return false;
      // tag == INT32 ?
      if (!GetLocal(e, localIdx) || !e.writeOp(Op::I64Const) || !e.writeVarS64(32) ||
          !e.writeOp(Op::I64ShrU) || !e.writeOp(Op::I32WrapI64) ||
          !e.writeOp(Op::I32Const) ||
          !e.writeVarS32(int32_t(TagWord(JSVAL_TYPE_INT32))) ||
          !e.writeOp(Op::I32Eq))
        return false;
      if (!e.writeOp(Op::If) || !e.writeFixedU8(kF64bt)) return false;
      if (!GetLocal(e, localIdx) || !e.writeOp(Op::I32WrapI64) ||
          !e.writeOp(Op::F64ConvertI32S))
        return false;  // int32 -> convert
      if (!e.writeOp(Op::Else)) return false;
      // non-number: deopt. It sits 2 `if`s deep (vs the standard guard's 1), so the
      // OOL deopt br needs one extra scope (relative, composes with any caller nest).
      {
        int32_t saved = int32_t(be.deoptExtraNest);
        be.deoptExtraNest = uint32_t(saved + 1);
        bool ok = EmitDeopt(e, be);
        be.deoptExtraNest = uint32_t(saved);
        if (!ok) return false;
      }
      if (!e.writeOp(Op::F64Const) || !e.writeFixedF64(0.0))
        return false;  // placeholder (unreachable after the deopt's br) for f64 type
      if (!e.writeOp(Op::End)) return false;  // end int32 if
      return e.writeOp(Op::End);             // end double if
    }
    case MIRType::Float32: {
      // Float32 values are held as f64 in our repr (WJValType). The boxed JS value
      // for a Float32-typed def (e.g. a Float32Array element read) is a plain JS
      // number == a double, so unbox it exactly like Double (number -> f64). Without
      // this, gbemu's audio path (executeIteration) bailed the whole opcode loop to
      // PBL on an Unbox:Float32.
      return EmitUnboxLocal(e, be, localIdx, MIRType::Double, fallible);
    }
    default:
      if (WJGETENV("GECKO_WJ_UNBOXDBG"))
        fprintf(stderr, "[wj-unbox-badtype] out=%s\n", StringFromMIRType(out));
      return false;
  }
}

// Push a boxed i64 JS::Value for resume-point operand `v`. Constants are baked
// (GC-thing constants via the traced pool); live values are loaded + boxed.
static bool EmitSpillValue(Encoder& e, WJBackend& be, MDefinition* v) {
  if (v->isConstant()) {
    MConstant* c = v->toConstant();
    MIRType t = c->type();
    if (t == MIRType::Object || t == MIRType::String || t == MIRType::Symbol ||
        t == MIRType::BigInt) {
      uintptr_t slot = WJInternConstant(c->toJSValue().asRawBits());
      if (!slot) { js::wasm::gWJBailReason = "intern-const-full"; return false; }
      return e.writeOp(Op::I32Const) && e.writeVarS32(int32_t(slot)) &&
             e.writeOp(Op::I64Load) && e.writeVarU32(3) && e.writeVarU32(0);
    }
    uint64_t bits;
    switch (t) {
      case MIRType::Int32:
      case MIRType::Double:
      case MIRType::Boolean:
      case MIRType::Null:
      case MIRType::Undefined:
        bits = c->toJSValue().asRawBits();
        break;
      case MIRType::MagicOptimizedOut:
        bits = JS::MagicValue(JS_OPTIMIZED_OUT).asRawBits();
        break;
      case MIRType::MagicIsConstructing:
      case MIRType::MagicUninitializedLexical:
        // The `new`-operator is-constructing sentinel (and uninit-lexical), captured
        // in resume points of the CreateThis/construct path (raytrace `new Vector`).
        // Spelling these makes those guards spill+deopt instead of bailing the whole
        // function -- BUT that exposes a construct-path deopt-RESUME correctness bug:
        // a deopt inside a `new`-call frame rebuilds a PBL frame whose constructing
        // state is wrong, and unwinding it crashes FrameIter (raytrace "unreachable"
        // in JSJitFrameIter::baselineScriptAndPc). Until the construct-frame resume
        // is correct (deltablue-class resume work), keep this OFF by default so the
        // function cleanly bails to PBL (correct). NOW DEFAULT-ON: the construct-
        // frame deopt-RESUME is sound since the icEntry-alignment fix (the prior
        // crash was the unsound resume, not the spill) -- enabling it lets raytrace's
        // rayTrace/getRay compile instead of running in PBL (55->63). NOTE: zeal-
        // deterministic bisection implicates getRay's construct-frame resume in a
        // GC-staleness, but turning this OFF makes raytrace WORSE at the default
        // nursery (2/10 vs 6/10) -- the zeal culprit is not representative; the real
        // residual is spread across the tracing core. Kept DEFAULT-ON; GECKO_WJ_NOCONSTRUCTSPILL reverts.
        if (WJGETENV("GECKO_WJ_NOCONSTRUCTSPILL"))
          return WJBAIL("spill: const type %s (construct-spill disabled)\n",
                        StringFromMIRType(t));
        bits = (t == MIRType::MagicIsConstructing)
                   ? JS::MagicValue(JS_IS_CONSTRUCTING).asRawBits()
                   : JS::MagicValue(JS_UNINITIALIZED_LEXICAL).asRawBits();
        break;
      default:
        return WJBAIL("spill: const type %s\n", StringFromMIRType(t));
    }
    return e.writeOp(Op::I64Const) && e.writeVarS64(int64_t(bits));
  }
  // Parameter: read the GC-rooted COPY local (boxed i64), reloaded across calls
  // by WJCollectRoots and initialized at the top of the prologue. Reading the raw
  // wasm param would spill a STALE object payload after a minor GC moves the arg
  // (earley/nboyer). Fall back to the raw param only if there's no copy local.
  if (v->op() == MDefinition::Opcode::Parameter) {
    const MParameter* p = v->toParameter();
    if (p->index() != MParameter::THIS_SLOT &&
        uint32_t(p->index()) >= kWJMaxArgs)
      return false;
    int32_t cl = be.local(v);
    if (cl >= 0) return GetLocal(e, uint32_t(cl));  // GC-current boxed copy
    uint32_t pidx = (p->index() == MParameter::THIS_SLOT)
                        ? kWJThisParam
                        : (kWJArg0Param + uint32_t(p->index()));
    return GetLocal(e, pidx);  // boxed i64 param, no re-boxing needed
  }
  if (be.local(v) < 0) return WJBAIL("spill: op#%u type %s has no local\n",
                                     unsigned(v->op()), StringFromMIRType(v->type()));
  // Read via GetOp (rematerializes Unbox/GuardShape/Slots/Elements object ptrs
  // from their rooted source) so a spilled/staged object ptr isn't stale across a
  // GC-ing call (earley). GetOp falls back to the cached local for plain nodes.
  if (!GetOp(e, be, v)) return false;
  if (v->type() == MIRType::Value || v->type() == MIRType::Int64) return true;
  if (v->type() == MIRType::Float32) {
    // Float32 is held as an f64 (WJValType); box it as a double Value -- a float
    // typed-array read's JS value IS a double. Without this a Float32 resume operand
    // (gbemu executeIteration captures a LoadUnboxedScalar:Float32 audio read in a
    // resume point) couldn't spill, bailing the WHOLE main opcode loop to PBL.
    return EmitBoxFromStack(e, MIRType::Double);
  }
  if (v->type() == MIRType::Object || v->type() == MIRType::Int32 ||
      v->type() == MIRType::Boolean || v->type() == MIRType::Double ||
      v->type() == MIRType::String || v->type() == MIRType::Symbol ||
      v->type() == MIRType::BigInt) {
    return EmitBoxFromStack(e, v->type());
  }
  return WJBAIL("spill: type %s not boxable\n", StringFromMIRType(v->type()));
}

// OSR-load: the INVERSE of EmitSpillValue for cheap-resume. Load the boxed value the
// deopt spilled at gWJResumeVals[valsIdx] and unbox it into `def`'s wasm local (its
// repr per WJValType). Infallible: we trust the spilled type == the def's type (the
// spill stored exactly this def). No-op if the def has no persistent local (it is
// rematerialized on use). See the OSR design spec in [[jetstream-harness]].
static bool EmitOsrLoadSlot(Encoder& e, WJBackend& be, MDefinition* def,
                            uint32_t valsIdx) {
  int32_t l = be.local(def);
  if (l < 0) return true;  // recomputed on use; nothing persistent to restore
  uintptr_t valsBase = uintptr_t(static_cast<void*>(&gWJResumeVals[0]));
  if (!e.writeOp(Op::I32Const) ||
      !e.writeVarS32(int32_t(valsBase + valsIdx * 8)) || !e.writeOp(Op::I64Load) ||
      !e.writeVarU32(3) || !e.writeVarU32(0))
    return false;  // boxed i64 on stack
  MIRType t = def->type();
  if (t == MIRType::Value || t == MIRType::Int64) {
    // i64 local: keep the boxed bits as-is.
  } else if (t == MIRType::Double || t == MIRType::Float32) {
    if (!e.writeOp(Op::F64ReinterpretI64)) return false;  // f64 local
  } else {
    if (!e.writeOp(Op::I32WrapI64)) return false;  // i32 local (payload)
  }
  return e.writeOp(Op::LocalSet) && e.writeVarU32(uint32_t(l));
}

// Store a number into an object slot as a CANONICAL DOUBLE (leaves the boxed i64
// on the stack). int32 N and double N are JS-indistinguishable, so storing the
// int32 as its f64-bits keeps number slots all-double -> their reads become
// infallible f64 reinterprets (GECKO_WJ_INFALLIBLE_DBL) with no per-read tag
// branch. Sound ONLY for slots read as Double/Value (NOT Int32-typed slots, whose
// infallible read does I32WrapI64). We canonicalize Value-typed stores (the
// raytrace `initialize(x,y,z)` param case) at runtime, and statically-Int32
// stores. GECKO_WJ_SLOTCANON gates; off -> identical to EmitSpillValue.
static bool EmitCanonStoreValue(Encoder& e, WJBackend& be, MDefinition* v) {
  // DEBUG (GECKO_WJ_VALSTORE): validate GC-typed values at slot stores via
  // wjhelp kind 252 (checks IsCellPointerValid + rope children C++-side, aborts
  // with the storing fn's line). Catches a stale value at the WRITE that
  // poisons a heap slot, instead of at the far-downstream read (task #64).
  static int valStore = WJGETENV("GECKO_WJ_VALSTORE") ? 1 : 0;
  if (valStore && WJTypeMaybeGC(v->type())) {
    uintptr_t valAddr = uintptr_t(static_cast<void*>(&gWJHelpVal));
    uintptr_t slotAddr = uintptr_t(static_cast<void*>(&gWJHelpSlot));
    // gWJHelpSlot = the stored def's MIR id (names the source: which Unbox/
    // Call/Phi produced the stale value).
    if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(slotAddr)) ||
        !e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(v->id())) ||
        !e.writeOp(Op::I32Store) || !e.writeVarU32(2) || !e.writeVarU32(0))
      return false;
    if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(valAddr)) ||
        !EmitSpillValue(e, be, v) ||
        !e.writeOp(Op::I64Store) || !e.writeVarU32(3) || !e.writeVarU32(0))
      return false;
    if (!e.writeOp(Op::F64Const) || !e.writeFixedF64(252.0) ||
        !e.writeOp(Op::F64Const) || !e.writeFixedF64(double(gWJCurScriptLine)) ||
        !e.writeOp(Op::Call) || !e.writeVarU32(0) || !e.writeOp(Op::Drop))
      return false;
  }
  static int canon = WJGETENV("GECKO_WJ_SLOTCANON") ? 1 : 0;
  if (!canon) return EmitSpillValue(e, be, v);
  if (v->type() == MIRType::Int32) {
    int32_t vl = be.local(v);
    if (vl >= 0)
      return GetLocal(e, uint32_t(vl)) && e.writeOp(Op::F64ConvertI32S) &&
             e.writeOp(Op::I64ReinterpretF64);
  } else if (v->type() == MIRType::Value) {
    int32_t vl = be.local(v);
    if (vl >= 0) {
      // cond = (tag(v) == INT32)
      if (!GetLocal(e, uint32_t(vl)) || !e.writeOp(Op::I64Const) ||
          !e.writeVarS64(32) || !e.writeOp(Op::I64ShrU) ||
          !e.writeOp(Op::I32WrapI64) || !e.writeOp(Op::I32Const) ||
          !e.writeVarS32(int32_t(TagWord(JSVAL_TYPE_INT32))) ||
          !e.writeOp(Op::I32Eq))
        return false;
      if (!e.writeOp(Op::If) || !e.writeFixedU8(0x7E)) return false;  // result i64
      // then: int32 low bits -> double bits
      if (!GetLocal(e, uint32_t(vl)) || !e.writeOp(Op::I32WrapI64) ||
          !e.writeOp(Op::F64ConvertI32S) || !e.writeOp(Op::I64ReinterpretF64))
        return false;
      if (!e.writeOp(Op::Else)) return false;
      if (!GetLocal(e, uint32_t(vl))) return false;  // keep as-is
      return e.writeOp(Op::End);
    }
  }
  return EmitSpillValue(e, be, v);
}

// Leave a raw i32 object pointer for `v` on the stack (used for the resume env
// chain). Handles object locals, boxed-Value locals, and object constants.
static bool EmitObjPtr(Encoder& e, WJBackend& be, MDefinition* v) {
  if (v->isConstant()) {
    MConstant* c = v->toConstant();
    if (c->type() == MIRType::Object) {
      JSObject* obj = c->toObjectOrNull();
      if (!obj) return e.writeOp(Op::I32Const) && e.writeVarS32(0);
      uintptr_t slot = WJInternConstant(JS::ObjectValue(*obj).asRawBits());
      if (!slot) return false;
      return e.writeOp(Op::I32Const) && e.writeVarS32(int32_t(slot)) &&
             e.writeOp(Op::I64Load) && e.writeVarU32(3) && e.writeVarU32(0) &&
             e.writeOp(Op::I32WrapI64);
    }
    return false;
  }
  // Rematerialize via GetOp (Unbox/GuardShape/etc. re-derive from their rooted
  // source) -- a cached object local goes stale across a GC-ing call (the callee
  // ptr for a 2nd call, the resume env ptr). GetOp falls back to be.local for
  // plain nodes, so this is a safe superset.
  if (v->type() == MIRType::Object) return GetOp(e, be, v);
  if (v->type() == MIRType::Value) {
    return GetOp(e, be, v) && e.writeOp(Op::I32WrapI64);
  }
  if (be.local(v) < 0) return false;
  return GetLocal(e, uint32_t(be.local(v)));
}

// Sound deopt: box the resume point's live state [this,args,locals,stack] into
// gWJResumeVals, record pc + stack depth, and call wjhelp(WJH_RESUME), which
// rebuilds a PBL frame and finishes the function there.
static bool EmitDeoptResumeInline(Encoder& e, WJBackend& be) {
  if (!be.curRp || !be.info) { js::wasm::gWJBailReason = "resume-no-rp"; return WJBAIL("resume: no rp/info\n"); }
  // Default resume-bail reason so an un-tagged return-false below reports THIS,
  // not the stale last-emitted op name (WJBAIL doesn't set gWJBailReason). The
  // intentional bails (forin-loophead-deopt/spill-operand/resume-operand-count)
  // override it. Fixes babel's misleading "reason=BinaryCache" (task #63/#57).
  js::wasm::gWJBailReason = "resume-emit-fail";
  // Build the inline frame chain: index 0 = INNERMOST (the deopt point, possibly
  // in an inlined callee), then its callers out to the OUTERMOST (the compiled
  // function). WJH_RESUME runs them innermost->outermost in PBL, threading each
  // frame's return into the next outer frame's call-result slot. Each frame uses
  // its OWN CompileInfo (script/nargs/nlocals/slots) -- crucial because an
  // inlined frame's operands index its inlinee's script, not be.info's.
  std::vector<MResumePoint*> frames;
  for (MResumePoint* rp = be.curRp; rp; rp = rp->caller()) {
    frames.push_back(rp);
    if (frames.size() > kWJMaxResumeFrames) {
    js::wasm::gWJBailReason = "resume-too-deep";
    return false;  // too deep: bail
  }
  }
  uint32_t nframes = uint32_t(frames.size());
  // Multi-frame (inlined) deopt reconstruction: WJH_RESUME rebuilds each inline
  // frame and resumes the outer (caller) frames AT the call (ResumeAt), so PBL
  // simply re-executes the inlined call -- which is correct. This is the key to
  // keeping hot inlined functions compiled instead of bailing them whole to PBL
  // (verified: richards/deltablue compile fully + run correctly + ~2x faster with
  // this on). Gate OFF with GECKO_WJ_NOINLINERESUME=1 for debugging only.
  static int noInlineResume = WJGETENV("GECKO_WJ_NOINLINERESUME") ? 1 : 0;
  if (nframes > 1 && noInlineResume) return false;

  // Debug: gWJDeoptByOp[curOp]++ at the deopt path (which guard kind deopts).
  // GECKO_WJ_DEOPTLINE: index by the EmitDeopt call's source line instead, so a
  // stale curOp can't mis-attribute (line%kWJNumOps; print maps back).
  static int deoptLine = WJGETENV("GECKO_WJ_DEOPTLINE") ? 1 : 0;
  static int deoptHist = (WJGETENV("GECKO_WJ_DEOPTHIST") || deoptLine) ? 1 : 0;
  uint32_t histIdx = deoptLine ? (be.deoptCallLine / 8) : be.curOp;
  if (deoptHist && histIdx < kWJNumOps) {
    uintptr_t a = uintptr_t(static_cast<void*>(&gWJDeoptByOp[histIdx]));
    if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(a)) ||
        !e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(a)) ||
        !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0) ||
        !e.writeOp(Op::I32Const) || !e.writeVarS32(1) || !e.writeOp(Op::I32Add) ||
        !e.writeOp(Op::I32Store) || !e.writeVarU32(2) || !e.writeVarU32(0))
      return false;
  }
  uintptr_t valsBase = uintptr_t(static_cast<void*>(&gWJResumeVals[0]));
  auto storeI32 = [&](uintptr_t addr, int32_t val) -> bool {
    return e.writeOp(Op::I32Const) && e.writeVarS32(int32_t(addr)) &&
           e.writeOp(Op::I32Const) && e.writeVarS32(val) &&
           e.writeOp(Op::I32Store) && e.writeVarU32(2) && e.writeVarU32(0);
  };

  // DIAGNOSTIC: poison the resume-vals buffer before spilling, so that any slot
  // WJH_RESUME reads but the spill below did NOT write returns a recognizable
  // garbage Value -> turns a resume under-spill (stale-slot read) into a
  // deterministic failure. GECKO_WJ_POISONRESUME.
  if (WJGETENV("GECKO_WJ_POISONRESUME")) {
    for (uint32_t i = 0; i < 160; i++) {
      if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(valsBase + i * 8)) ||
          !e.writeOp(Op::I64Const) || !e.writeVarS64(int64_t(0xfff8ccccccccccccull)) ||
          !e.writeOp(Op::I64Store) || !e.writeVarU32(3) || !e.writeVarU32(0))
        return false;
    }
  }
  uint32_t valsIdx = 0;  // running index into gWJResumeVals across all frames
  for (uint32_t f = 0; f < nframes; f++) {
    MResumePoint* rp = frames[f];
    const CompileInfo& info = rp->block()->info();
    uint32_t nargs = info.nargs();
    uint32_t nlocals = info.nlocals();
    uint32_t firstStack = info.firstStackSlot();
    if (rp->numOperands() < firstStack) {
      js::wasm::gWJBailReason = "resume-operand-count";
      if (WJGETENV("GECKO_WJ_DEOPTRESUMEDBG"))
        fprintf(stderr, "[wj-resume-opcount] f=%u nOps=%u firstStack=%u "
                "nargs=%u nlocals=%u\n", f, unsigned(rp->numOperands()),
                firstStack, nargs, nlocals);
      return false;
    }
    uint32_t stackDepth = uint32_t(rp->numOperands()) - firstStack;
    uint32_t frameValsOff = valsIdx;

    // Return-threading for inlined CALLER frames (f>0). The outer frame's resume
    // point sits AT the inlined call with the call inputs [callee, this, args] on
    // its expr stack. Re-executing the call would re-run a side-effecting callee a
    // SECOND time (the inner frame already ran it) -- deltablue's incrementalAdd
    // (newMark()/satisfy mutate the graph) corrupted state this way. Instead we
    // resume AFTER the call with the inner frame's return value threaded onto the
    // stack top, exactly as a real bailout does: spill only the stack values BELOW
    // the call inputs, then a result slot the runtime fills with the inner return.
    bool threaded = (f > 0);
    uint32_t spillStack = stackDepth;  // # of rp stack operands to spill
    uint32_t outDepth = stackDepth;    // stack depth PBL resumes with
    jsbytecode* callPc = nullptr;
    if (threaded) {
      callPc = rp->pc();
      JSOp jop = JSOp(*callPc);
      if (jop != JSOp::Call && jop != JSOp::CallContent &&
          jop != JSOp::CallIgnoresRv) {
        js::wasm::gWJBailReason = "resume-thread-noncall";
        return false;  // unusual call shape (spread/new/eval): don't inline-compile
      }
      uint32_t inputs = uint32_t(GET_ARGC(callPc)) + 2;  // callee + this + args
      if (stackDepth < inputs) {
        js::wasm::gWJBailReason = "resume-thread-depth";
        return false;
      }
      spillStack = stackDepth - inputs;  // stack values below the call inputs
      outDepth = spillStack + 1;         // + the threaded result slot
    }

    auto spill = [&](MDefinition* v) -> bool {
      if (!e.writeOp(Op::I32Const) ||
          !e.writeVarS32(int32_t(valsBase + valsIdx * 8)))
        return false;
      if (!EmitSpillValue(e, be, v)) {
        if (WJGETENV("GECKO_WJ_DEOPTRESUMEDBG")) {
          JSScript* ss = be.info ? be.info->script() : nullptr;
          fprintf(stderr, "[wj-spill-fail] %s:%u f-operand op=%s ty=%s local=%d\n",
                  ss ? ss->filename() : "?", ss ? unsigned(ss->lineno()) : 0,
                  WJOpName(v->op()), StringFromMIRType(v->type()), be.local(v));
        }
        if (!js::wasm::gWJBailReason ||
            !strstr(js::wasm::gWJBailReason, "spill"))
          js::wasm::gWJBailReason = "spill-operand";
        return false;
      }
      if (!e.writeOp(Op::I64Store) || !e.writeVarU32(3) || !e.writeVarU32(0))
        return false;
      valsIdx++;
      return true;
    };
    if (!spill(rp->getOperand(info.thisSlot()))) return false;
    for (uint32_t i = 0; i < nargs; i++) {
      if (!spill(rp->getOperand(info.argSlot(i)))) return false;
    }
    for (uint32_t i = 0; i < nlocals; i++) {
      if (!spill(rp->getOperand(info.localSlot(i)))) return false;
    }
    for (uint32_t i = 0; i < spillStack; i++) {
      if (!spill(rp->getOperand(info.stackSlot(i)))) return false;
    }
    if (threaded) {
      // Reserve the threaded-result slot (top of the resumed stack). The runtime
      // overwrites it with the inner frame's return before resuming this frame;
      // spill Undefined as a safe default.
      if (!e.writeOp(Op::I32Const) ||
          !e.writeVarS32(int32_t(valsBase + valsIdx * 8)) ||
          !e.writeOp(Op::I64Const) ||
          !e.writeVarS64(int64_t(JS::UndefinedValue().asRawBits())) ||
          !e.writeOp(Op::I64Store) || !e.writeVarU32(3) || !e.writeVarU32(0))
        return false;
      valsIdx++;
    }

    // Innermost (deopt) frame: resume-after modes advance past the producing op.
    // Outer (inlined-call) frames: resume AFTER the call with the threaded result
    // on the stack (set above) -- PBL continues post-call without re-running it.
    jsbytecode* resumePc = rp->pc();
    // A deopt resuming AT a LoopHead whose expr stack carries a live for-in
    // ITERATOR would make PBL re-run JSOp::MoreIter -- advancing the iterator
    // AGAIN and losing the in-flight key (acorn-bench walk: a dropped 'body'
    // key skipped a whole AST subtree). No sound resume exists for such sites
    // (the key was consumed but its body didn't finish), so BAIL the compile:
    // the function runs in PBL. Narrow: only for-in loop heads (iterator
    // operand on the rp stack), not every loop head.
    static int noForinBail = WJGETENV("GECKO_WJ_NOFORINBAIL") ? 1 : 0;
    if (!noForinBail && f == 0 && !IsResumeAfter(rp->mode()) &&
        JSOp(*rp->pc()) == JSOp::LoopHead) {
      uint32_t nfixed = info.nlocals();
      uint32_t base = 1 + info.nargs() + nfixed;  // this + args + locals
      for (uint32_t i = base; i < rp->numOperands(); i++) {
        MDefinition* d = rp->getOperand(i);
        if (d->op() == MDefinition::Opcode::ObjectToIterator ||
            (d->isPhi() && d->toPhi()->numOperands() > 0 &&
             d->toPhi()->getOperand(0)->op() ==
                 MDefinition::Opcode::ObjectToIterator)) {
          js::wasm::gWJBailReason = "forin-loophead-deopt";
          if (WJGETENV("GECKO_WJ_FORINDBG")) {
            JSScript* fs = be.info ? be.info->script() : nullptr;
            fprintf(stderr, "[wj-forin-bail] %s:%u deoptOp=%s\n",
                    fs && fs->filename() ? fs->filename() : "?",
                    fs ? unsigned(fs->lineno()) : 0,
                    WJOpName(jit::MDefinition::Opcode(be.curOp)));
          }
          return WJBAIL("forin-loophead-deopt\n");
        }
      }
    }
    if (f == 0 && IsResumeAfter(rp->mode())) resumePc = GetNextPc(resumePc);
    if (threaded) resumePc = GetNextPc(callPc);
    uint32_t pcOff = uint32_t(resumePc - info.script()->code());
    if (WJGETENV("GECKO_WJ_RPDBG")) {
      fprintf(stderr, "[wj-rpc] f=%u %s:%u mode=%d rawpc=%u op=%s resumepc=%u depth=%u deoptOp=%s\n",
              f, info.script()->filename(), unsigned(info.script()->lineno()),
              int(rp->mode()), unsigned(rp->pc() - info.script()->code()),
              js::CodeName(JSOp(*rp->pc())), pcOff, stackDepth,
              f == 0 ? WJOpName(jit::MDefinition::Opcode(be.curOp)) : "-");
    }
    if (!storeI32(uintptr_t(&gWJResumePc[f]), int32_t(pcOff))) return false;
    if (!storeI32(uintptr_t(&gWJResumeStackDepth[f]), int32_t(outDepth)))
      return false;
    if (!storeI32(uintptr_t(&gWJResumeScriptPtr[f]),
                  int32_t(uintptr_t(info.script()))))
      return false;
    if (!storeI32(uintptr_t(&gWJResumeNArgs[f]), int32_t(nargs))) return false;
    if (!storeI32(uintptr_t(&gWJResumeNLocals[f]), int32_t(nlocals))) return false;
    if (!storeI32(uintptr_t(&gWJResumeValsOff[f]), int32_t(frameValsOff)))
      return false;
    // Env chain: spill the RP's env operand (the frame's actual environment at
    // the deopt point -- the function's own CallObject if one was created, else
    // the captured enclosing env). If the operand can't be emitted, fall back to
    // the RUNTIME env from the persistent root slot for the outer frame -- NOT 0:
    // a 0 made WJH_RESUME use the CANONICAL function's ->environment(), which is
    // WRONG for closures sharing a script (acorn-walk NFE `c` resumed with a
    // garbage env; captured reads returned junk).
    if (!e.writeOp(Op::I32Const) ||
        !e.writeVarS32(int32_t(uintptr_t(&gWJResumeEnvPtr[f]))))
      return false;
    if (!EmitObjPtr(e, be, rp->getOperand(info.environmentChainSlot()))) {
      if (f == nframes - 1 && be.useEnvRoot) {
        uintptr_t rootsBase = uintptr_t(static_cast<void*>(&gWJCallRoots[0]));
        if (!GetLocal(e, be.envRootIdx) || !e.writeOp(Op::I32Const) ||
            !e.writeVarS32(8) || !e.writeOp(Op::I32Mul) ||
            !e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(rootsBase)) ||
            !e.writeOp(Op::I32Add) || !e.writeOp(Op::I64Load) ||
            !e.writeVarU32(3) || !e.writeVarU32(0) || !e.writeOp(Op::I32WrapI64))
          return false;
      } else if (!e.writeOp(Op::I32Const) || !e.writeVarS32(0)) {
        return false;
      }
    }
    if (!e.writeOp(Op::I32Store) || !e.writeVarU32(2) || !e.writeVarU32(0))
      return false;

    // Enclosing env for the resumed frame's PBL prologue. The OUTERMOST frame
    // (f == nframes-1) is THIS compiled function; its enclosing (captured) env is
    // envLocal (= gWJCurrentEnv at entry = closure->environment()). The PBL
    // resume uses this to rebuild the frame env correctly instead of the canonical
    // script->function()->environment() (wrong for a re-instantiated closure).
    // Inlined frames: store 0 (keep canonical fallback -- they re-execute anyway).
    // Runtime callee for the resumed frame (JSOp::Callee must push the ACTUAL
    // closure, not the canonical script->function() -- acorn-walk NFE). Outer
    // frame only; inlined frames re-execute with the canonical fallback.
    if (!e.writeOp(Op::I32Const) ||
        !e.writeVarS32(int32_t(uintptr_t(&gWJResumeCalleeFn[f]))))
      return false;
    if (f == nframes - 1 && be.usesCallee && be.useEnvRoot) {
      uintptr_t rootsBaseC = uintptr_t(static_cast<void*>(&gWJCallRoots[0]));
      if (!GetLocal(e, be.envRootIdx) || !e.writeOp(Op::I32Const) ||
          !e.writeVarS32(8) || !e.writeOp(Op::I32Mul) ||
          !e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(rootsBaseC)) ||
          !e.writeOp(Op::I32Add) || !e.writeOp(Op::I64Load) ||
          !e.writeVarU32(3) || !e.writeVarU32(8) || !e.writeOp(Op::I32WrapI64))
        return false;
    } else if (!e.writeOp(Op::I32Const) || !e.writeVarS32(0)) {
      return false;
    }
    if (!e.writeOp(Op::I32Store) || !e.writeVarU32(2) || !e.writeVarU32(0))
      return false;

    if (!e.writeOp(Op::I32Const) ||
        !e.writeVarS32(int32_t(uintptr_t(&gWJResumeEnclosingEnv[f]))))
      return false;
    if (f == nframes - 1 && be.useEnvRoot) {
      // GC-current env from the persistent root slot (not the possibly-stale
      // envLocal snapshot): gWJCallRoots[envRootIdx] boxed -> unbox to i32.
      uintptr_t rootsBase = uintptr_t(static_cast<void*>(&gWJCallRoots[0]));
      if (!GetLocal(e, be.envRootIdx) || !e.writeOp(Op::I32Const) ||
          !e.writeVarS32(8) || !e.writeOp(Op::I32Mul) ||
          !e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(rootsBase)) ||
          !e.writeOp(Op::I32Add) || !e.writeOp(Op::I64Load) ||
          !e.writeVarU32(3) || !e.writeVarU32(0) || !e.writeOp(Op::I32WrapI64))
        return false;
    } else if (f == nframes - 1) {
      if (!GetLocal(e, be.envLocal)) return false;
    } else if (!e.writeOp(Op::I32Const) || !e.writeVarS32(0)) {
      return false;
    }
    if (!e.writeOp(Op::I32Store) || !e.writeVarU32(2) || !e.writeVarU32(0))
      return false;

    if (valsIdx + 64 >= 1024) return false;  // gWJResumeVals overflow guard
  }
  if (!storeI32(uintptr_t(&gWJResumeNFrames), int32_t(nframes))) return false;

  // Beyond-formal actuals: the resume point only carries FORMALS, so a deopt in
  // an arguments-using fn called with more actuals than formals would resume PBL
  // with argc==nargs -- `arguments` empty/short (lodash overRest saw length 0 and
  // pushed garbage; the residual coffeescript dyn-frame-arg bug). Spill the entry
  // argc snapshot + all staged actuals; WJH_RESUME grafts slots [nargs, argc)
  // into the physical frame's arg region. usesArgc fns only (others can't
  // observe the extras). GECKO_WJ_NOACTUALSGRAFT reverts.
  {
    uintptr_t aaAddr = uintptr_t(static_cast<void*>(&gWJResumeActualArgc));
    if (be.usesArgc && !WJGETENV("GECKO_WJ_NOACTUALSGRAFT")) {
      if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(aaAddr)) ||
          !GetLocal(e, be.actualArgcLocal) || !e.writeOp(Op::I32Store) ||
          !e.writeVarU32(2) || !e.writeVarU32(0))
        return false;
      uintptr_t actBase = uintptr_t(static_cast<void*>(&gWJResumeActuals[0]));
      uintptr_t argRootsBase2 = uintptr_t(static_cast<void*>(&gWJCallRoots[0]));
      uint32_t argRootOff2 = be.usesCallee ? 2 : 1;
      for (uint32_t i = 0; i < kWJMaxArgs; i++) {
        if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(actBase)))
          return false;
        if (be.needsArgRoots && be.useEnvRoot) {
          // GC-current read from the traced shadow slot: raw params go stale
          // across a pre-deopt GC (CreateArgumentsObject alone can trigger one)
          // -> grafting them crashed the compartment check (GetCustomDataProperty).
          if (!GetLocal(e, be.envRootIdx) || !e.writeOp(Op::I32Const) ||
              !e.writeVarS32(int32_t(argRootOff2 + i)) || !e.writeOp(Op::I32Add) ||
              !e.writeOp(Op::I32Const) || !e.writeVarS32(8) ||
              !e.writeOp(Op::I32Mul) || !e.writeOp(Op::I32Const) ||
              !e.writeVarS32(int32_t(argRootsBase2)) || !e.writeOp(Op::I32Add) ||
              !e.writeOp(Op::I64Load) || !e.writeVarU32(3) || !e.writeVarU32(0))
            return false;
        } else {
          uint32_t src = be.argCopyLocal[i] >= 0 ? uint32_t(be.argCopyLocal[i])
                                                 : (kWJArg0Param + i);
          if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(src)) return false;
        }
        if (!e.writeOp(Op::I64Store) || !e.writeVarU32(3) || !e.writeVarU32(i * 8))
          return false;
      }
    } else {
      if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(aaAddr)) ||
          !e.writeOp(Op::I32Const) || !e.writeVarS32(0) ||
          !e.writeOp(Op::I32Store) || !e.writeVarU32(2) || !e.writeVarU32(0))
        return false;
    }
  }

  // call wjhelp(WJH_RESUME, 0) -> f64 flag (0 ok / 1 threw). Old ABI reuses the flag
  // as the function's return flag; EHABI checks it and converts a PBL-resume throw to
  // a wasm throw.
  if (!e.writeOp(Op::F64Const) || !e.writeFixedF64(double(WJH_RESUME)) ||
      !e.writeOp(Op::F64Const) || !e.writeFixedF64(0.0) ||
      !e.writeOp(Op::Call) || !e.writeVarU32(0)) {
    return false;
  }
  if constexpr (kWJEHABI) {
    // f64 flag on stack; if the PBL resume threw, throw a wasm exn (env root left for
    // the catcher's wholesale gWJRootSP restore).
    if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(be.callFlagLocal)) return false;
    if (!GetLocal(e, be.callFlagLocal) || !e.writeOp(Op::F64Const) ||
        !e.writeFixedF64(0.0) || !e.writeOp(Op::F64Ne))
      return false;
    if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
    if (!EmitThrowWjexn(e)) return false;
    if (!e.writeOp(Op::End)) return false;
    // return just the boxed PBL result.
    if (!GetLocal(e, be.scratchBase) || !e.writeOp(Op::I64Load) ||
        !e.writeVarU32(3) || !e.writeVarU32(kWJResultOff))
      return false;
    if (!EmitEnvRootPop(e, be)) return false;
    if (!EmitDepthPop(e, be)) return false;  // this JIT frame is leaving -> un-ratchet the recursion byte-guard
    return e.writeOp(Op::Return);
  }
  // Old ABI: WJH_RESUME left the f64 flag on the stack; push the boxed result so the
  // return matches the [f64 flag, i64 result] signature.
  if (!GetLocal(e, be.scratchBase) || !e.writeOp(Op::I64Load) ||
      !e.writeVarU32(3) || !e.writeVarU32(kWJResultOff)) {
    return false;
  }
  if (!EmitEnvRootPop(e, be)) return false;  // pop the persistent env root slot
  if (!EmitDepthPop(e, be)) return false;    // deopt-return also leaves this JIT frame -> restore gWJJitDepth (else it ratchets -> false over-recursion, e.g. gbemu)
  return e.writeOp(Op::Return);
}

// Guard-miss deopt. In out-of-line mode (the dispatch path's hot block bodies),
// branch to the shared dispatcher (`block $D` wrapping the loop) after recording
// the site's resume point -- this keeps the hot loop body tiny. Otherwise inline
// the full spill+resume sequence (single-block / reloop / terminator paths).
static bool EmitDeoptResume(Encoder& e, WJBackend& be) {
  static int noResume = WJGETENV("GECKO_WJ_NORESUME") ? 1 : 0;
  if (noResume) return false;  // bisect: bail any function needing a resume
  static int tinyDeopt = WJGETENV("GECKO_WJ_TINYDEOPT") ? 1 : 0;
  if (tinyDeopt) return e.writeOp(Op::Unreachable);  // PROBE: measure code-bloat cost
  static int constExit = WJGETENV("GECKO_WJ_CONSTEXIT") ? 1 : 0;
  if (constExit) {  // PROBE: minimal exit reading NO locals (isolates resume liveness)
    return e.writeOp(Op::F64Const) && e.writeFixedF64(0.0) &&
           e.writeOp(Op::I64Const) &&
           e.writeVarS64(int64_t(JS::UndefinedValue().asRawBits())) &&
           e.writeOp(Op::Return);
  }
  if (!be.curRp || !be.info) return WJBAIL("resume: no rp/info\n");
  // Record WHICH op deopted (runtime storm control reads it: GuardGlobalGeneration).
  {
    uintptr_t opAddr = uintptr_t(static_cast<void*>(&gWJLastDeoptOp));
    if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(opAddr)) ||
        !e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(be.curOp)) ||
        !e.writeOp(Op::I32Store) || !e.writeVarU32(2) || !e.writeVarU32(0))
      return false;
  }
  // Categorize this deopt site for the count-trigger discriminator (see WasmJitBackend.h):
  // shape/object guards (whose respec recompile re-bakes GC consts -- unsound) vs all else
  // (numeric/type deopts -- the beneficial re-typing case).
  {
    gWJEmitTotalDeopts++;
    uint32_t op = be.curOp;
    if (op == uint32_t(MDefinition::Opcode::GuardShape) ||
        op == uint32_t(MDefinition::Opcode::GuardShapeList) ||
        op == uint32_t(MDefinition::Opcode::GuardShapeListToOffset) ||
        op == uint32_t(MDefinition::Opcode::GuardProto) ||
        op == uint32_t(MDefinition::Opcode::GuardSpecificFunction) ||
        // GuardGlobalGeneration deopts mean the fn's BAKED GLOBAL BINDINGS are
        // stale -- a count-trigger respec re-bakes them mid-storm (the acorn
        // off-by-one-node mis-parse class), so categorize with the shape family.
        op == uint32_t(MDefinition::Opcode::GuardGlobalGeneration) ||
        op == uint32_t(MDefinition::Opcode::GuardObjectIdentity)) {
      gWJEmitShapeDeopts++;
    }
  }
  // Per-deopt-site hit counter (GECKO_WJ_SITEHIST). Emitted HERE -- at the guard,
  // before the OOL/inline split -- so deoptCallLine (set by the EmitDeopt macro
  // just now) is correct for ALL deopts, not just inline ones. The reliable
  // attribution is (line=guard, JS script:line); curOp may be stale.
  if (gWJSiteHistEnabled < 0) {
    gWJSiteHistEnabled = WJGETENV("GECKO_WJ_SITEHIST") ? 1 : 0;
  }
  if (gWJSiteHistEnabled && gWJSiteCounter < kWJMaxDSites) {
    uint32_t gid = gWJSiteCounter++;
    gWJSiteOp[gid] = uint16_t(be.curOp);
    gWJSiteLine[gid] = be.deoptCallLine;
    JSScript* ss = be.info ? be.info->script() : nullptr;
    gWJSiteFile[gid] = ss && ss->filename() ? ss->filename() : "?";
    gWJSiteJsLine[gid] = ss ? uint32_t(ss->lineno()) : 0;
    uintptr_t h = uintptr_t(static_cast<void*>(&gWJSiteHits[gid]));
    if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(h)) ||
        !e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(h)) ||
        !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0) ||
        !e.writeOp(Op::I32Const) || !e.writeVarS32(1) || !e.writeOp(Op::I32Add) ||
        !e.writeOp(Op::I32Store) || !e.writeVarU32(2) || !e.writeVarU32(0))
      return false;
  }
  if (be.oolDeopt && be.inDispatchBody) {
    uint32_t site = uint32_t(be.deoptSites.size());
    if (WJGETENV("GECKO_WJ_DEOPTSITEDBG")) {
      JSScript* ms = be.info ? be.info->script() : nullptr;
      fprintf(stderr, "[wj-deoptsite] %s:%u curOp=%u(%s) callLine=%u\n",
              ms && ms->filename() ? ms->filename() : "?",
              ms ? unsigned(ms->lineno()) : 0, be.curOp,
              be.curOp < kWJNumOps ? "" : "?", be.deoptCallLine);
    }
    be.deoptSites.push_back({be.curRp, be.curOp});
    // deoptSiteLocal = site; br to `block $D`. Dispatch loop: $D is one scope past
    // loop $L (bodyLoopIdx blocks + $L + the guard `if` = bodyLoopIdx+2). Relooper:
    // $D wraps reloopScopeDepth nested loop/block scopes; from inside the guard `if`
    // that's reloopScopeDepth (relooper scopes) + 1 (guard `if`) to reach $D. Both
    // add deoptExtraNest for any extra `if` levels the op opened around the deopt.
    uint32_t brDepth = be.reloopOol
                           ? (be.reloopScopeDepth + 1 + be.deoptExtraNest)
                           : (be.bodyLoopIdx + 2 + be.deoptExtraNest);
    return e.writeOp(Op::I32Const) && e.writeVarS32(int32_t(site)) &&
           e.writeOp(Op::LocalSet) && e.writeVarU32(be.deoptSiteLocal) &&
           e.writeOp(Op::Br) && e.writeVarU32(brDepth);
  }
  return EmitDeoptResumeInline(e, be);
}


// Emit a call to wjhelp(kind, site) for a VM op whose boxed operands the caller
// has already staged in gWJScratch[0..]. Roots the resume point's live
// Object/Value locals across the call (the helper may GC), propagates a thrown
// exception (return [1.0, 0]), reloads the roots, and leaves the boxed i64
// result (gWJScratch[kWJResultOff]) on the stack. `ins` is the node producing
// the result (excluded from rooting). Returns false => can't emit (bail fn).
// Collect the GC roots that must survive a (possibly-GCing) call at `ins`:
// the resume point's live Object/Value operands PLUS every Object/Value value
// defined earlier in `ins`'s block (so it dominates and definitely ran) that
// still has a use at/after `ins`. The latter is essential because our resume
// points are frequently the stale block-entry RP, which records mid-function
// slots as their entry constants -- missing live temporaries (a call's callee,
// a freshly-constructed object, an operand held across a callee that allocates).
// GetOp-root-set: the local-backed Object/Value defs GetOp(o) actually READS to
// materialize o, recursively through remat passthroughs -- EXACT mirror of GetOpInner/
// WJCanGetOp. Self-rooted nodes (FunctionEnvironment/Parameter/NurseryObject/Constant)
// + non-GC types contribute nothing per-safepoint. Used by the precise-rooting dataflow.
static void WJRootSet(WJBackend& be, const MDefinition* o,
                      std::unordered_set<const MDefinition*>& out, int depth) {
  using Op_ = MDefinition::Opcode;
  if (!o || depth > 64) return;
  if (o->op() == Op_::FunctionEnvironment || o->op() == Op_::NurseryObject ||
      o->isConstant())
    return;
  if (o->op() == Op_::Parameter) {
    // Parameters are read via their prologue COPY LOCAL (GetOpInner), and an
    // Unbox-of-Parameter REMATERIALIZES from that copy on every use -- so the
    // copy local must be spilled/reloaded across GC points where the Parameter
    // is live, like any other root. Historically covered incidentally by the
    // curRp resume-operand rooting (arg slots reference Parameters), but a fn
    // that REASSIGNS an arg (`str = String(str)`) drops the Parameter from
    // later resume points while arg-derived Unboxes still remat from the copy
    // -> stale after any GC. terser's print() stored the stale `str` into the
    // closure slot `last`, poisoning the heap (task #64). GECKO_WJ_NOPARAMROOT
    // reverts.
    static int noParamRoot = WJGETENV("GECKO_WJ_NOPARAMROOT") ? 1 : 0;
    if (!noParamRoot && o->type() == MIRType::Value && be.local(o) >= 0)
      out.insert(o);
    return;
  }
  bool passthrough = (o->type() == MIRType::Slots || o->type() == MIRType::Elements);
  switch (o->op()) {
    case Op_::GuardShape:
    case Op_::GuardShapeList:
    case Op_::GuardSpecificFunction:
    case Op_::GuardToFunction:
    case Op_::GuardFunctionScript:
    case Op_::ConstantProto:
    case Op_::GuardObjectIdentity:
      passthrough = true;
      break;
    case Op_::Unbox:
      if (o->type() == MIRType::Object || o->type() == MIRType::String ||
          o->type() == MIRType::Symbol || o->type() == MIRType::BigInt) {
        MIRType st = o->getOperand(0)->type();
        // GetOpInner rematerializes from src for these src types, so the SOURCE
        // must be rooted -- but the Unbox's own cached local is ALSO read
        // DIRECTLY on some paths (prop-IC receiver staging, resume spills), so
        // root BOTH: recurse into the source here and fall through to the tail
        // insert for the Unbox itself. Rooting only the source left the Unbox
        // local stale after a pre-IC GC -> uglify's WJH_PROPIC '.print' OOB
        // (AST_Binary printer, GVN-merged self.right receiver). Liveness-gated,
        // so the cost is bounded (unlike the old blanket GECKO_WJ_UNBOXROOT).
        if (st == MIRType::Value || st == MIRType::Int64 ||
            st == MIRType::Object || st == MIRType::String ||
            st == MIRType::Symbol || st == MIRType::BigInt) {
          WJRootSet(be, o->getOperand(0), out, depth + 1);
        }
      }
      break;
    default:
      break;
  }
  if (passthrough && o->numOperands() >= 1) {
    WJRootSet(be, o->getOperand(0), out, depth + 1);
    return;
  }
  if ((o->type() == MIRType::Object || o->type() == MIRType::Value) && be.local(o) >= 0)
    out.insert(o);
}

// Proper backward dataflow liveness over the MIR graph (GECKO_WJ_NOROOTALL path only).
// Fills be.wjLiveOut[B] = root-set elements live at B's exit. gen/kill per block +
// fixpoint -- correctly handles loops/cross-block, which the prior ad-hoc per-safepoint
// use-checks did not (the 4 failed attempts). Phi operands are live on their incoming
// edge -> added to the predecessor's live-out.
static void WJComputeLiveness(WJBackend& be, MIRGraph& graph) {
  if (be.wjLiveDone) return;
  be.wjLiveDone = true;
  using DefSet = std::unordered_set<const MDefinition*>;
  std::vector<MBasicBlock*> blocks;
  for (MBasicBlockIterator b = graph.begin(); b != graph.end(); b++)
    blocks.push_back(*b);
  std::unordered_map<const MBasicBlock*, DefSet> gen, kill, phiOut, liveIn;
  auto isRootElem = [&](const MDefinition* d) {
    return (d->type() == MIRType::Object || d->type() == MIRType::Value ||
            // STRING/SYMBOL/BIGINT locals hold raw nursery-movable cell ptrs:
            // an unrooted one goes stale across any GC point (marked's corrupt
            // ropes / crypto's ~1/10 OOB -- string-heavy code, tiny-nursery
            // amplified). Root + reload them exactly like objects.
            d->type() == MIRType::String || d->type() == MIRType::Symbol ||
            d->type() == MIRType::BigInt) &&
           be.local(d) >= 0;
  };
  for (MBasicBlock* B : blocks) {
    DefSet g, k;
    std::vector<MInstruction*> instrs;
    for (MInstructionIterator it = B->begin(); it != B->end(); it++)
      instrs.push_back(*it);
    for (auto rit = instrs.rbegin(); rit != instrs.rend(); ++rit) {
      MInstruction* I = *rit;
      if (isRootElem(I)) { g.erase(I); k.insert(I); }
      for (size_t i = 0; i < I->numOperands(); i++) WJRootSet(be, I->getOperand(i), g, 0);
      // Resume-point operands are LIVE USES at I: a deopt here re-materializes them (via
      // GetOp/EmitSpillValue) -> their root-sets must survive any safepoint before I. The
      // prior dataflow missed these (only instruction operands counted) -> under-rooted.
      if (MResumePoint* rp = I->resumePoint())
        for (uint32_t i = 0; i < rp->numOperands(); i++)
          WJRootSet(be, rp->getOperand(i), g, 0);
    }
    gen[B] = std::move(g);
    kill[B] = std::move(k);
    // phi operands -> live on the incoming edge (predecessor's live-out).
    for (MPhiIterator p = B->phisBegin(); p != B->phisEnd(); p++) {
      MPhi* phi = *p;
      for (size_t k2 = 0; k2 < B->numPredecessors(); k2++)
        WJRootSet(be, phi->getOperand(k2), phiOut[B->getPredecessor(k2)], 0);
    }
  }
  bool changed = true;
  int guard = 0;
  while (changed && guard++ < 2000) {
    changed = false;
    for (auto rit = blocks.rbegin(); rit != blocks.rend(); ++rit) {
      MBasicBlock* B = *rit;
      DefSet out = phiOut[B];
      for (size_t s = 0; s < B->numSuccessors(); s++) {
        MBasicBlock* S = B->getSuccessor(s);
        for (const MDefinition* d : liveIn[S]) out.insert(d);
      }
      DefSet in = gen[B];
      for (const MDefinition* d : out)
        if (!kill[B].count(d)) in.insert(d);
      if (out != be.wjLiveOut[B]) { be.wjLiveOut[B] = std::move(out); changed = true; }
      if (in != liveIn[B]) { liveIn[B] = std::move(in); changed = true; }
    }
  }
}

static void WJCollectRoots(WJBackend& be, MInstruction* ins,
                           std::vector<uint32_t>& rootLocal,
                           std::vector<uint8_t>& rootIsObj) {
  // Root-kind byte: 0=boxed Value (no re-box), 1=Object, 2=String, 3=Symbol,
  // 4=BigInt (raw cell ptrs re-boxed with their tag at the spill).
  auto add = [&](MDefinition* d, uint8_t kind) {
    // A GC-thing CONSTANT is already rooted in the traced const pool (gWJConstPool):
    // every use re-derives it via GetOpInner's isConstant() short-circuit (pool load,
    // GC-current), never from a stale local. So its GC-root spill here is pure
    // redundancy -- skip it. (~10% of gbemu root-adds are Constants.) The separate
    // resume-value spill (EmitSpillValue) still bakes constants for deopt; this only
    // drops the per-call gWJCallRoots spill. GECKO_WJ_NOCONSTROOTSKIP reverts.
    static int noSkip = WJGETENV("GECKO_WJ_NOCONSTROOTSKIP") ? 1 : 0;
    if (!noSkip && d->isConstant()) return;
    int32_t l = be.local(d);
    if (l < 0) return;
    for (uint32_t r = 0; r < rootLocal.size(); r++) {
      if (rootLocal[r] == uint32_t(l)) return;
    }
    rootLocal.push_back(uint32_t(l));
    rootIsObj.push_back(kind);
    // ROOTOPDBG: histogram the DEF OPCODE of each spilled root, to see how many are
    // rematerializable (pure slot-loads off a rooted receiver) vs not (call results,
    // arith, phis). Dumped every 200k adds.
    static int ropdbg = WJGETENV("GECKO_WJ_ROOTOPDBG") ? 1 : 0;
    if (ropdbg) {
      static std::unordered_map<uint32_t, uint64_t> opc;
      static uint64_t tot = 0;
      opc[uint32_t(d->op())]++;
      if ((++tot % 5000) == 0) {
        std::vector<std::pair<uint32_t, uint64_t>> v(opc.begin(), opc.end());
        std::sort(v.begin(), v.end(),
                  [](auto& a, auto& b) { return a.second > b.second; });
        fprintf(stderr, "[wj-rootop] after %llu root-adds:\n",
                (unsigned long long)tot);
        for (size_t i = 0; i < v.size() && i < 12; i++)
          fprintf(stderr, "    %s x%llu\n",
                  WJOpName(jit::MDefinition::Opcode(v[i].first)),
                  (unsigned long long)v[i].second);
      }
    }
  };
  auto kindOf = [](MIRType t) -> uint8_t {
    switch (t) {
      case MIRType::Object: return 1;
      case MIRType::String: return 2;
      case MIRType::Symbol: return 3;
      case MIRType::BigInt: return 4;
      default: return 0;
    }
  };
  // Precise root-set live immediately AFTER ins (populated below when !rootAll, via the
  // proper backward dataflow). consider() roots a dominating Object/Value local iff it's
  // in here. Empty when rootAll (default) -> consider roots all (the old behavior).
  std::unordered_set<const MDefinition*> liveAfterSet;
  // DEFAULT 2026-06-28: PRECISE dataflow rooting (root only the live-after-ins root-set,
  // via WJComputeLiveness). The old conservative rootAll (root ALL dominating object/value
  // locals at every safepoint) was SOUND but over-spilled massively -> a huge pervasive
  // overhead. Proper backward dataflow liveness + GetOp-read-closure (WJRootSet) + resume-
  // point operand uses makes precise rooting SOUND (validated: ~130 normal runs + 75 GC-
  // flaky-bench runs + tiny-nursery & GCZeal=7 stress = ZERO TraceRoot/staleness crashes,
  // zero wrong-answers) and FASTER (+16-56% suite-wide: ubo 2.8x->3.3x, richards +54%,
  // crypto +52%, deltablue +56%, raytrace +30%, earley +33%, navier +20%, gbemu ->JIT-
  // positive). The earlier "unsound for remat" belief was a partial liveness (missing the
  // 5 guard/proto passthroughs + resume-point reads); the "NOROOTALL ~70%" readings were
  // test-CONTENTION artifacts (stale parallel test procs), not under-rooting -- clean
  // serial A/B shows precise == rootAll on crashes (both 0) and faster. GECKO_WJ_ROOTALL=1
  // forces the old conservative rooting (fallback). See [[gc-root-spill-lever]].
  static int rootAll = WJGETENV("GECKO_WJ_ROOTALL") ? 1 : 0;
  // DEBUG bisection (GECKO_WJ_ROOTADD): in NOROOTALL mode, additionally root all
  // dominating values of a chosen op-category beyond the precise liveAfterSet -- to
  // isolate WHICH category my dataflow liveness wrongly drops (the ~30% residual crash).
  // 99=all dominating (sanity: must = rootAll = sound), 1=Phi, 2=Call/Construct,
  // 3=GetProp/LoadSlot family, 4=Unbox, 5=Box/New*.
  static int rootAddCat = WJGETENV("GECKO_WJ_ROOTADD")
                              ? atoi(WJGETENV("GECKO_WJ_ROOTADD")) : 0;
  auto consider = [&](MDefinition* d) {
    using Op_ = MDefinition::Opcode;
    MIRType dt = d->type();
    bool isObj = dt == MIRType::Object;
    bool isVal = dt == MIRType::Value;
    bool isStrCell = dt == MIRType::String || dt == MIRType::Symbol ||
                     dt == MIRType::BigInt;
    if (!((isObj || isVal || isStrCell) && be.local(d) >= 0)) return;
    // String/Symbol/BigInt locals: ALWAYS root (rootAll-style). The precise
    // liveness misses some live string defs (marked's corrupt-rope crash
    // persisted precise-only but passes under ROOTALL) -- until the string
    // read-closure gap is found, over-rooting these rarer kinds is cheap+sound.
    // Object-typed Unbox root (OPT-IN GECKO_WJ_UNBOXROOT): the precise liveness drops
    // some live Unbox-to-Object defs -> uglify-js deterministic mem-OOB (fixed by
    // ROOTADD=4 / this). The Unbox VALUE is rematerialized on GetOp reads, but a DIRECT
    // read (resume-point / Phi) of its cached local isn't in the read-closure, so its
    // rooting is dropped -> stale after a minor GC. Always-rooting ALL Object Unbox FIXES
    // uglify but CRATERS octane ~2x (raytrace/earley/splay/navier -- Object Unbox is
    // pervasive in hot loops), so it's OPT-IN until a SURGICAL fix closes the read-closure
    // gap (root only the direct-read Unbox locals, not all). See [[uglify-rooting-crash]].
    static int unboxRoot = WJGETENV("GECKO_WJ_UNBOXROOT") ? 1 : 0;
    bool unboxObj = isObj && d->op() == Op_::Unbox && unboxRoot;
    bool root = rootAll || isStrCell || unboxObj || liveAfterSet.count(d);
    if (!root && rootAddCat) {
      Op_ op = d->op();
      if (rootAddCat == 99) root = true;
      else if (rootAddCat == 1 && d->isPhi()) root = true;
      else if (rootAddCat == 2 &&
               (op == Op_::Call || op == Op_::ConstructArray ||
                op == Op_::ConstructArgs || op == Op_::ApplyArgs))
        root = true;
      else if (rootAddCat == 3 &&
               (op == Op_::GetPropertyCache || op == Op_::LoadFixedSlot ||
                op == Op_::LoadDynamicSlot || op == Op_::LoadFixedSlotAndUnbox ||
                op == Op_::LoadDynamicSlotAndUnbox || op == Op_::MegamorphicLoadSlot))
        root = true;
      else if (rootAddCat == 4 && op == Op_::Unbox) root = true;
      else if (rootAddCat == 5 &&
               (op == Op_::Box || op == Op_::NewObject || op == Op_::NewArray ||
                op == Op_::NewPlainObject || op == Op_::NewArrayObject))
        root = true;
    }
    if (root) add(d, kindOf(dt));
  };
  // Root values whose def dominates `ins` and are live across it, unioned with
  // the call's resume-point operands (the precise abstract-interpreter live
  // state, covering loop-carried values the dominance walk misses). Interior
  // pointers are kept fresh by GetOp rematerialization, not rooting.
  MBasicBlock* insBlock = ins->block();
  MIRGraph& graph = insBlock->graph();
  if (!rootAll) {
    // Compute the EXACT root-set live immediately after ins: start from the block's
    // live-out (proper fixpoint) and walk backward to ins, applying per-instruction
    // kill(def)/gen(operand root-sets). Sound for loops/remat (unlike the old walk).
    WJComputeLiveness(be, graph);
    liveAfterSet = be.wjLiveOut[insBlock];
    std::vector<MInstruction*> instrs;
    for (MInstructionIterator it = insBlock->begin(); it != insBlock->end(); it++)
      instrs.push_back(*it);
    for (auto rit = instrs.rbegin(); rit != instrs.rend(); ++rit) {
      MInstruction* I = *rit;
      if (I == ins) break;  // stop: liveAfterSet now = live immediately AFTER ins
      if ((I->type() == MIRType::Object || I->type() == MIRType::Value) &&
          be.local(I) >= 0)
        liveAfterSet.erase(I);
      for (size_t i = 0; i < I->numOperands(); i++)
        WJRootSet(be, I->getOperand(i), liveAfterSet, 0);
      if (MResumePoint* rp = I->resumePoint())
        for (uint32_t i = 0; i < rp->numOperands(); i++)
          WJRootSet(be, rp->getOperand(i), liveAfterSet, 0);
    }
  }
  for (MBasicBlockIterator b = graph.begin(); b != graph.end(); b++) {
    MBasicBlock* blk = *b;
    bool same = (blk == insBlock);
    if (!same && !blk->dominates(insBlock)) continue;
    for (MPhiIterator p = blk->phisBegin(); p != blk->phisEnd(); p++) consider(*p);
    for (MInstructionIterator it = blk->begin(); it != blk->end(); it++) {
      MInstruction* d = *it;
      // Break at `ins` in LIST (execution/schedule) order -- NOT by MIR id. Warp's
      // id order != schedule order (an instruction created later, e.g. a call's
      // callee `this.m` evaluated before its args, gets a HIGHER id but executes
      // EARLIER). The old `d->id() >= ins->id()` break skipped such higher-id-but-
      // earlier-executed object locals -> they weren't rooted across an intervening
      // alloc/GC -> stale/freed pointer (richards `this.addTask(...,new X())`: the
      // callee load id=37 executes before the `new X` construct id=27, was missed
      // -> "null function"). Reaching `ins` in the list = exactly the defs executed
      // so far, regardless of id.
      if (same && d == ins) break;
      consider(d);
    }
  }
  // EXPERIMENT (GECKO_WJ_ROOTLIVE): also root EVERY liveAfterSet member, not just
  // those whose def dominates ins. Tests the box2d/COLDCALL hypothesis that a value
  // live across ins with a NON-dominating def (missed by the dominating-block loop
  // above) is the residual GC-staleness. Gated off by default.
  // ROOTLIVE: also root liveAfterSet members whose def does NOT dominate ins (missed
  // by the dominating-block loop above) -- postcss's #19 crash is exactly a live-but-
  // non-dominating value dropped here. OBJECT-only by default (postcss's dropped value
  // is an Object-Unbox; restricting to Object avoids rooting the many Value locals in
  // compute hot loops -> much cheaper than +Value, which cost ~15% on crypto/raytrace).
  // GECKO_WJ_ROOTLIVEVAL adds Value back (fuller, but the compute cost).
  // DEFAULT-ON (Object-only): fixes the non-dominating-live-OBJECT rooting-miss that
  // crashed postcss (#19 class), at ~neutral octane cost (best-of-3: crypto/raytrace/
  // navier unchanged -- rooting only OBJECTs, not the many Value locals in compute
  // loops). Soundness fix: a live object with a non-dominating def IS reachable at ins
  // and must survive a GC. GECKO_WJ_NOROOTLIVE reverts; GECKO_WJ_ROOTLIVEVAL adds Value
  // (fuller but ~15% compute cost).
  static int rootLive = WJGETENV("GECKO_WJ_NOROOTLIVE") ? 0 : 1;
  static int rootLiveVal = WJGETENV("GECKO_WJ_ROOTLIVEVAL") ? 1 : 0;
  if (!rootAll && rootLive) {
    for (const MDefinition* d : liveAfterSet) {
      bool isObj = d->type() == MIRType::Object;
      bool isVal = d->type() == MIRType::Value;
      if (((isObj || (isVal && rootLiveVal))) && be.local(d) >= 0)
        add(const_cast<MDefinition*>(d), isObj);
    }
  }
  for (MResumePoint* rp = be.curRp; rp; rp = rp->caller()) {
    uint32_t n = rp->numOperands();
    for (uint32_t i = 0; i < n; i++) {
      MDefinition* d = rp->getOperand(i);
      bool isObj = d->type() == MIRType::Object;
      bool isVal = d->type() == MIRType::Value;
      if ((isObj || isVal) && be.local(d) >= 0) add(d, isObj);
      // (Do NOT add WJRootSet(d) here: a resume operand's remat SOURCE can be a
      // NON-dominating local -> spilling it reads garbage -> TraceRoot crash. That
      // unconditional addition regressed even default rootAll 10/10 -> 8/10. Reverted.
      // KEY LESSON: only DOMINATING values are valid to root; rooting more is NOT free.)
    }
  }
  // NB: the closure env is NOT rooted here -- it lives in a PERSISTENT GC-shadow
  // slot (gWJCallRoots[envRootIdx]) pushed in the prologue for the whole
  // invocation, and FunctionEnvironment rematerializes from it (always GC-current,
  // even across non-call GC points). See WJEmitBody prologue + GetOp.
}

// EHABI: throw the module-local $wjexn tag (index 0). The JS exception value rides on
// cx (the helper/callee set it); the tag is a payload-less control-flow marker. Caught by
// an enclosing try-region catch_all, or unwinds to the trampoline (which converts it back
// to a sentinel flag at the engine boundary).
static bool EmitThrowWjexn(Encoder& e) {
  return e.writeOp(Op::Throw) && e.writeVarU32(0);
}

// Emit the action taken when a call/helper signals a pending JS exception (the body of
// `if (flag != 0)`). In a function WITH a catch/finally handler (try/catch compiled via
// GECKO_WJ_NOTRYBAIL), route through deopt-in-error: set gWJResumeInError and resume in
// PBL, whose HandleException walks the trynotes and RUNS THE CATCH (or propagates if the
// throwing pc isn't covered) -- Warp builds no catch-block MIR, so the catch must run in
// PBL. Otherwise: EHABI throws a wasm exn; old ABI returns [1.0, 0] (propagate out).
static bool EmitExceptionExit(Encoder& e, WJBackend& be) {
  if (be.hasCatchTrynote) {
    uintptr_t addr = uintptr_t(static_cast<void*>(&gWJResumeInError));
    if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(addr)) ||
        !e.writeOp(Op::I32Const) || !e.writeVarS32(1) || !e.writeOp(Op::I32Store) ||
        !e.writeVarU32(2) || !e.writeVarU32(0))
      return false;
    // Inline resume (this is inside the call's flag-check `if`, not a guard's `if`, so
    // the OOL br-depth would be wrong) -- mirror the Throw case's inDispatchBody save.
    bool saved = be.inDispatchBody;
    be.inDispatchBody = false;
    bool ok = EmitDeoptResume(e, be);
    be.inDispatchBody = saved;
    return ok;
  }
  // Restore gWJJitDepth to this frame's entry value before propagating up (each
  // unwinding frame self-balances; else the depth ratchets up across exceptions).
  if (be.depthChecked) {
    if (!e.writeOp(Op::I32Const) ||
        !e.writeVarS32(int32_t(uintptr_t(static_cast<void*>(&gWJJitDepth)))) ||
        !GetLocal(e, be.savedDepthLocal) || !e.writeOp(Op::I32Store) ||
        !e.writeVarU32(2) || !e.writeVarU32(0))
      return false;
  }
  if constexpr (kWJEHABI) return EmitThrowWjexn(e);
  return e.writeOp(Op::F64Const) && e.writeFixedF64(1.0) &&
         e.writeOp(Op::I64Const) && e.writeVarS64(0) && e.writeOp(Op::Return);
}

static bool EmitHelperCallResult(Encoder& e, WJBackend& be, MInstruction* ins,
                                 int kind, uint32_t site) {
  // Remat cache: this helper call is a MOVING GC point (alloc/GC/user-code) -- any
  // cached re-derived pointer is stale after it. Invalidate (args were staged by the
  // caller before this). Barriers do NOT use this path (direct Op::Call), so the
  // across-barrier cache survives, which is the win.
  be.rematInvalidate();
  std::vector<uint32_t> rootLocal;
  std::vector<uint8_t> rootIsObj;
  WJCollectRoots(be, ins, rootLocal, rootIsObj);
  if (WJGETENV("GECKO_WJ_CROOTDBG") && gWJCurScriptLine ==
      uint32_t(atoi(WJGETENV("GECKO_WJ_CROOTDBG")))) {
    fprintf(stderr, "[crootdbg-helper] line=%u op=%s insId=%u block=%u roots:",
            gWJCurScriptLine, WJOpName(ins->op()), unsigned(ins->id()),
            ins->block() ? unsigned(ins->block()->id()) : 0u);
    for (uint32_t k = 0; k < rootLocal.size(); k++)
      fprintf(stderr, " %u(obj=%d)", rootLocal[k], rootIsObj[k]);
    fprintf(stderr, "\n");
  }
  static int noRoot = WJGETENV("GECKO_WJ_NOROOT") ? 1 : 0;
  if (noRoot) {
    rootLocal.clear();
    rootIsObj.clear();
  }
  uint32_t nRoots = uint32_t(rootLocal.size());
  if (WJGETENV("GECKO_WJ_ROOTDBG") && kind == WJH_NEWPLAIN) {
    fprintf(stderr, "[rootdbg] NEWPLAIN ins#%u nRoots=%u locals:", ins->id(),
            nRoots);
    for (uint32_t k = 0; k < nRoots; k++) fprintf(stderr, " %u", rootLocal[k]);
    fprintf(stderr, "\n");
  }
  uintptr_t rootsBase = uintptr_t(static_cast<void*>(&gWJCallRoots[0]));
  uintptr_t spAddr = uintptr_t(static_cast<void*>(&gWJRootSP));
  auto emitSPAdjust = [&](int32_t delta) -> bool {
    return e.writeOp(Op::I32Const) && e.writeVarS32(int32_t(spAddr)) &&
           e.writeOp(Op::I32Const) && e.writeVarS32(int32_t(spAddr)) &&
           e.writeOp(Op::I32Load) && e.writeVarU32(2) && e.writeVarU32(0) &&
           e.writeOp(Op::I32Const) && e.writeVarS32(delta) &&
           e.writeOp(Op::I32Add) && e.writeOp(Op::I32Store) && e.writeVarU32(2) &&
           e.writeVarU32(0);
  };
  // Compute the spill slot BASE once: rootBaseLocal = gWJRootSP*8 + rootsBase. SP is
  // unchanged between spill and reload (bumped after spill, restored before reload),
  // so the same base serves both; slot k is base + k*8 via the wasm store/load
  // immediate offset -- avoids reloading gWJRootSP and recomputing *8+base per root
  // (was ~8 ops/root x2). splay is call+root-heavy: this is its single biggest lever
  // (GC-root spill = +14% NOROOT). base is 8-aligned (rootsBase + sp*8 both are).
  if (nRoots) {
    if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(spAddr)) ||
        !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0) ||
        !e.writeOp(Op::I32Const) || !e.writeVarS32(8) || !e.writeOp(Op::I32Mul) ||
        !e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(rootsBase)) ||
        !e.writeOp(Op::I32Add) || !e.writeOp(Op::LocalSet) ||
        !e.writeVarU32(be.rootBaseLocal))
      return false;
  }
  auto emitSlotBase = [&]() -> bool { return GetLocal(e, be.rootBaseLocal); };
  for (uint32_t k = 0; k < nRoots; k++) {
    if (!emitSlotBase()) return false;
    if (!GetLocal(e, rootLocal[k])) return false;
    if (rootIsObj[k] == 1 && !EmitBoxFromStack(e, MIRType::Object)) return false;
    if (rootIsObj[k] == 2 && !EmitBoxFromStack(e, MIRType::String)) return false;
    if (rootIsObj[k] == 3 && !EmitBoxFromStack(e, MIRType::Symbol)) return false;
    if (rootIsObj[k] == 4 && !EmitBoxFromStack(e, MIRType::BigInt)) return false;
    if (!e.writeOp(Op::I64Store) || !e.writeVarU32(3) || !e.writeVarU32(k * 8))
      return false;
  }
  if (nRoots && !emitSPAdjust(int32_t(nRoots))) return false;
  // wjhelp(kind, site) -> f64 flag
  if (!e.writeOp(Op::F64Const) || !e.writeFixedF64(double(kind)) ||
      !e.writeOp(Op::F64Const) || !e.writeFixedF64(double(site)) ||
      !e.writeOp(Op::Call) || !e.writeVarU32(0))
    return false;
  if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(be.callFlagLocal)) return false;
  if (nRoots && !emitSPAdjust(-int32_t(nRoots))) return false;
  // exception -> return [1.0, 0]. GECKO_WJ_NOFLAGCHECK skips the per-call exception
  // propagation (CEILING PROBE: measures the cost of our per-call exception-flag ABI
  // vs Ion's native stack unwinding -- UNSOUND for throwing code, correct for the
  // non-throwing target benches).
  static int noFlagCheck = WJGETENV("GECKO_WJ_NOFLAGCHECK") ? 1 : 0;
  if (!noFlagCheck) {
  if (!GetLocal(e, be.callFlagLocal) || !e.writeOp(Op::F64Const) ||
      !e.writeFixedF64(0.0) || !e.writeOp(Op::F64Ne))
    return false;
  if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
  if (!EmitExceptionExit(e, be)) return false;
  if (!e.writeOp(Op::End)) return false;
  }  // end !noFlagCheck
  for (uint32_t k = 0; k < nRoots; k++) {
    if (!emitSlotBase()) return false;
    if (!e.writeOp(Op::I64Load) || !e.writeVarU32(3) || !e.writeVarU32(k * 8))
      return false;
    if (rootIsObj[k] && !e.writeOp(Op::I32WrapI64)) return false;
    if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(rootLocal[k])) return false;
  }
  // boxed result
  return GetLocal(e, be.scratchBase) && e.writeOp(Op::I64Load) &&
         e.writeVarU32(3) && e.writeVarU32(kWJResultOff);
}

// Stage a boxed value operand into gWJScratch[slot] (byte offset slot*8).
static bool EmitStageScratch(Encoder& e, WJBackend& be, MDefinition* v,
                             uint32_t slot) {
  return GetLocal(e, be.scratchBase) && EmitSpillValue(e, be, v) &&
         e.writeOp(Op::I64Store) && e.writeVarU32(3) && e.writeVarU32(slot * 8);
}

// If `ins` is the `this` (arg 0) operand of a constructing call, return that
// MCall. The construct's callee/newTarget locals go stale across `ins`'s
// allocation (an inline-`this` NewObject/NewPlainObject can GC), so the alloc
// site pre-stages them into the TRACED 62/63 scratch slots while still fresh
// (see the NewObject case + the Call(constructing) case).
static MCall* WJConstructConsumer(MInstruction* ins) {
  for (MUseIterator u = ins->usesBegin(); u != ins->usesEnd(); u++) {
    MNode* consumer = u->consumer();
    if (!consumer->isDefinition()) continue;
    MDefinition* cd = consumer->toDefinition();
    if (!cd->isCall()) continue;
    MCall* call = cd->toCall();
    if (call->isConstructing() && call->numStackArgs() > 0 &&
        call->getArg(0) == ins) {
      return call;
    }
  }
  return nullptr;
}

// Box (if Object) and store a call operand into a traced scratch slot, reading
// its CACHED local directly (no GetOp rematerialization, which would re-read a
// now-stale source). Used by the construct callee/newTarget pre-stage where the
// operand local is still fresh (no GC yet). Returns false if the operand has no
// local or an unexpected representation -> caller bails the construct to PBL.
static bool EmitStageOperandLocal(Encoder& e, WJBackend& be, MDefinition* d,
                                  uint32_t slot) {
  int32_t l = be.local(d);
  if (l < 0) {
    js::wasm::gWJBailReason = "construct-operand-no-local";
    return false;
  }
  if (!GetLocal(e, be.scratchBase) || !GetLocal(e, uint32_t(l))) return false;
  if (d->type() == MIRType::Object) {
    if (!EmitBoxFromStack(e, MIRType::Object)) return false;
  } else if (d->type() != MIRType::Value) {
    js::wasm::gWJBailReason = "construct-operand-bad-type";
    return false;
  }
  return e.writeOp(Op::I64Store) && e.writeVarU32(3) && e.writeVarU32(slot * 8);
}

// Stage a compile-time-constant boxed Value into gWJScratch[slot]. GC-thing
// values are baked via the traced constant pool (loaded live at runtime);
// others are immediate.
static bool EmitStageConstBoxed(Encoder& e, WJBackend& be, uint64_t bits,
                                uint32_t slot) {
  if (!GetLocal(e, be.scratchBase)) return false;
  JS::Value v = JS::Value::fromRawBits(bits);
  if (v.isGCThing()) {
    uintptr_t pslot = WJInternConstant(bits);
    if (!pslot) return false;
    if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(pslot)) ||
        !e.writeOp(Op::I64Load) || !e.writeVarU32(3) || !e.writeVarU32(0))
      return false;
  } else if (!e.writeOp(Op::I64Const) || !e.writeVarS64(int64_t(bits))) {
    return false;
  }
  return e.writeOp(Op::I64Store) && e.writeVarU32(3) && e.writeVarU32(slot * 8);
}

// Allocate a prop-IC site for the CURRENT mega read/store, REUSING the same site
// across recompiles via a stable (script, per-compile read-index) key. This keeps
// the site pool bounded by the number of distinct read sites (not by recompile
// count) so it never exhausts -- the exhaustion was why a late mega-recompile's
// read couldn't convert and fell back to a wrong fixed-slot load (deltablue 741).
static uint32_t WJPropSite(WJBackend& be) {
  JSScript* s = be.info ? be.info->script() : nullptr;
  uint64_t key = (uint64_t(uint32_t(uintptr_t(static_cast<void*>(s)))) << 20) |
                 (uint64_t(be.propReadIdx++) & 0xFFFFF);
  uint32_t site = js::wasm::WJAllocPropSiteKeyed(key);
  // GECKO_WJ_PROPSITEDBG=<site>: name the fn that owns a crashing prop-IC site
  // (pairs with WJDumpCrashState's lastHelpSite).
  static const char* psd = getenv("GECKO_WJ_PROPSITEDBG");
  static int psdSite = psd ? atoi(psd) : -1;
  if (psdSite >= 0 && site == uint32_t(psdSite)) {
    fprintf(stderr, "[wj-propsite] site=%u fn=%s:%u readIdx=%u curOp=%s\n", site,
            s && s->filename() ? s->filename() : "?",
            s ? unsigned(s->lineno()) : 0, be.propReadIdx - 1,
            WJOpName(jit::MDefinition::Opcode(be.curOp)));
  }
  return site;
}

// Inline property-load IC for an own data property with a compile-time-known
// name (MegamorphicLoadSlot, named GetPropertyCache). `object` must be a
// receiver of MIRType::Object. Loads obj->shape(), compares to up to kWJPropWays
// per-site cached shapes; on a match loads the slot inline (fixed or dynamic) --
// no C++ hop. On a miss it calls wjhelp(WJH_PROPIC, site), which does a pure
// lookup, fills the IC, and returns the value. Leaves the boxed value (i64) on
// the stack. `site` must be a valid (nonzero) pre-allocated prop-site. Mirrors
// Ion's megamorphic-cache inline load; the per-site cache avoids hash collisions.
static bool EmitPropIC(Encoder& e, WJBackend& be, MInstruction* ins,
                       MDefinition* object, uint64_t keyBits, uint32_t site,
                       js::Shape* bakedShape = nullptr, uint32_t bakedByteOff = 0,
                       bool bakedFixed = false, bool knownMega = false) {
  uint32_t base = site * kWJPropWays;
  // Sites are REUSED across recompiles (WJPropSite, keyed by script+read-index). If
  // a warmer recompile shifts which reads convert, a site could be repurposed for a
  // DIFFERENT property -- so reset the key AND the cached (shape->offset) ways here
  // each compile. A stale shape entry from the old property would otherwise match a
  // same-shaped receiver and load the OLD property's offset (wrong slot).
  if (gWJPropKey[site] != keyBits) {
    for (uint32_t w = 0; w < kWJPropWays; w++) gWJPropShape[base + w] = 0;
  }
  gWJPropKey[site] = keyBits;
  uintptr_t shapeArr = uintptr_t(static_cast<void*>(&gWJPropShape[0]));
  uintptr_t offArr = uintptr_t(static_cast<void*>(&gWJPropOff[0]));
  // propObj = receiver pointer (rematerialized, GC-current).
  if (!GetOp(e, be, object)) return false;
  if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(be.propObjLocal)) return false;
  // propShape = I32Load(propObj + offsetOfShape).
  if (!GetLocal(e, be.propObjLocal)) return false;
  if (!e.writeOp(Op::I32Load) || !e.writeVarU32(2) ||
      !e.writeVarU32(uint32_t(offsetof(JS::shadow::Object, shape))))
    return false;
  if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(be.propShapeLocal)) return false;

  uintptr_t holderArr = uintptr_t(static_cast<void*>(&gWJPropHolder[0]));
  // Load the slot value for way `w` (tagged offset at gWJPropOff[base+w]). If
  // gWJPropHolder[base+w] is set (proto-read way), load from that holder object
  // instead of the receiver. objBase scratch reuses propShapeLocal (the matched
  // way's then-branch never reads it again).
  auto emitSlotLoad = [&](uint32_t w) -> bool {
    // tagged offset -> propTaggedLocal; if it's the MISSING sentinel, yield
    // `undefined` (no slot load) -- the missing-property cache (uBlock reads tons of
    // absent optional fields). Both arms of this If yield i64.
    if (!e.writeOp(Op::I32Const) ||
        !e.writeVarS32(int32_t(offArr + (base + w) * 4)) ||
        !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0) ||
        !e.writeOp(Op::LocalTee) || !e.writeVarU32(be.propTaggedLocal) ||
        !e.writeOp(Op::I32Const) ||
        !e.writeVarS32(int32_t(kWJPropMissingSentinel)) || !e.writeOp(Op::I32Eq))
      return false;
    if (!e.writeOp(Op::If) || !e.writeFixedU8(uint8_t(TypeCode::I64))) return false;
    if (!e.writeOp(Op::I64Const) ||
        !e.writeVarS64(int64_t(JS::UndefinedValue().asRawBits())))
      return false;  // [undefined]
    if (!e.writeOp(Op::Else)) return false;
    // objBase = gWJPropHolder[w] ? holder : receiver
    if (!e.writeOp(Op::I32Const) ||
        !e.writeVarS32(int32_t(holderArr + (base + w) * 4)) ||
        !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0) ||
        !e.writeOp(Op::LocalTee) || !e.writeVarU32(be.propShapeLocal))
      return false;                                      // [holder], propShapeLocal=holder
    if (!GetLocal(e, be.propObjLocal)) return false;     // [holder, receiver]
    if (!GetLocal(e, be.propShapeLocal)) return false;   // [holder, receiver, holder(cond)]
    if (!e.writeOp(Op::SelectNumeric)) return false;            // [objBase]
    if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(be.propShapeLocal))
      return false;                                      // propShapeLocal = objBase
    // baseAddr = (tagged & 1) ? objBase : I32Load(objBase + offsetOfSlots)
    if (!GetLocal(e, be.propTaggedLocal) || !e.writeOp(Op::I32Const) ||
        !e.writeVarS32(1) || !e.writeOp(Op::I32And))
      return false;
    if (!e.writeOp(Op::If) || !e.writeFixedU8(uint8_t(TypeCode::I32))) return false;
    if (!GetLocal(e, be.propShapeLocal)) return false;
    if (!e.writeOp(Op::Else)) return false;
    if (!GetLocal(e, be.propShapeLocal) || !e.writeOp(Op::I32Load) ||
        !e.writeVarU32(2) ||
        !e.writeVarU32(uint32_t(js::NativeObject::offsetOfSlots())))
      return false;
    if (!e.writeOp(Op::End)) return false;
    // + (tagged >> 1)  (byte offset)
    if (!GetLocal(e, be.propTaggedLocal) || !e.writeOp(Op::I32Const) ||
        !e.writeVarS32(1) || !e.writeOp(Op::I32ShrU) || !e.writeOp(Op::I32Add))
      return false;
    if (!e.writeOp(Op::I64Load) || !e.writeVarU32(3) || !e.writeVarU32(0))
      return false;  // [value]
    // DEBUG: catch reuse-staleness -- if the loaded value's bits == the SWEPT nursery
    // poison pattern (0x2B*), this read came from a FREED (collected) cell whose
    // memory wasn't yet reused. Abort with the site (GECKO_WJ_READVALIDATE).
    static int readValidate = WJGETENV("GECKO_WJ_READVALIDATE") ? 1 : 0;
    if (readValidate) {
      if (!e.writeOp(Op::LocalTee) || !e.writeVarU32(be.unboxScratch)) return false;
      // if (value == 0x2B2B2B2B2B2B2B2B) report poison
      if (!GetLocal(e, be.unboxScratch) || !e.writeOp(Op::I64Const) ||
          !e.writeVarS64(int64_t(0x2B2B2B2B2B2B2B2BULL)) || !e.writeOp(Op::I64Eq))
        return false;
      if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
      if (!e.writeOp(Op::F64Const) || !e.writeFixedF64(double(WJH_CHECKCELL)) ||
          !e.writeOp(Op::F64Const) || !e.writeFixedF64(double(int(site))) ||
          !e.writeOp(Op::Call) || !e.writeVarU32(0) || !e.writeOp(Op::Drop))
        return false;
      if (!e.writeOp(Op::End)) return false;  // [value] still on stack from the tee
    }
    if (!e.writeOp(Op::End)) return false;  // end missing-sentinel If (both arms i64)
    return true;
  };

  // MEGAMORPHIC CACHE fallback (GECKO_WJ_MEGACACHE, default-OFF): when the per-site
  // monomorphic/poly ways all MISS, consult the per-runtime js::MegamorphicCache for an
  // OWN data property (hopsAndKind_==0) before the WJH_PROPIC C++ hop -- megamorphic
  // React/DOM sites (the discord.com/login GETPROP/PROPIC 450K-hop ceiling) hit here.
  // Constant key -> compile-time hash + key. Cache base is a stable per-runtime addr.
  // Mega-cache probe is emitted ONLY for compile-time-KNOWN-megamorphic sites
  // (MMegamorphicLoadSlot -> knownMega=true). Mono/poly GetPropertyCache sites use
  // ways+helper: emitting the ~40-op probe on their saturated-but-not-truly-mega
  // misses REGRESSED octane 3-14% (2026-07-11 clean A/B: deltablue 0.86, richards
  // 0.95, box2d/crypto/raytrace ~0.96) for zero gain -- those sites' shapes miss the
  // global cache, so the probe is pure overhead on top of the helper. Warp's op
  // choice is the right signal (it marks a site mega only after the IC saturates).
  // Default-ON for the targeted case; GECKO_WJ_NOMEGACACHE reverts. (Legacy
  // GECKO_WJ_MEGACACHE forced it on for ALL sites -- kept as an override for A/B.)
  static int megaCacheOff = WJGETENV("GECKO_WJ_NOMEGACACHE") ? 1 : 0;
  static int megaCacheAll = WJGETENV("GECKO_WJ_MEGACACHE") ? 1 : 0;
  bool useMega = (knownMega || megaCacheAll) && !megaCacheOff;
  uintptr_t megaCacheAddr = 0;
  uint32_t megaHashConst = 0;
  if (useMega) {
    JSContext* mcx = js::TlsContext.get();
    if (mcx) {
      megaCacheAddr = uintptr_t(&mcx->runtime()->caches().megamorphicCache);
      megaHashConst = uint32_t(
          js::HashPropertyKeyThreadSafe(JS::PropertyKey::fromRawBits(keyBits)));
    }
  }
  auto emitHelperMiss = [&]() -> bool {
    return EmitStageScratch(e, be, object, 0) &&
           EmitHelperCallResult(e, be, ins, WJH_PROPIC, site);
  };
  auto emitMegaOrMiss = [&]() -> bool {
    if (!megaCacheAddr) return emitHelperMiss();
    using MC = js::MegamorphicCache;
    using ME = js::MegamorphicCacheEntry;
    // MEGAMORPHIC GATE: only consult the mega cache when this SITE is saturated (all
    // kWJPropWays populated) -- i.e. genuinely megamorphic. A low-poly site (some ways
    // still empty) is served by the ways; running the mega hash+lookup on its misses is
    // pure overhead (navier's numeric field access regressed ~28% without this gate).
    // allFull = (way0 != 0) & (way1 != 0) & ...  -> if !allFull, straight to helper.
    if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(shapeArr + base * 4)) ||
        !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0) ||
        !e.writeOp(Op::I32Eqz) || !e.writeOp(Op::I32Eqz))
      return false;  // (way0 != 0) as 0/1
    for (uint32_t w = 1; w < kWJPropWays; w++) {
      if (!e.writeOp(Op::I32Const) ||
          !e.writeVarS32(int32_t(shapeArr + (base + w) * 4)) ||
          !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0) ||
          !e.writeOp(Op::I32Eqz) || !e.writeOp(Op::I32Eqz) || !e.writeOp(Op::I32And))
        return false;  // & (wayW != 0)
    }
    if (!e.writeOp(Op::If) || !e.writeFixedU8(uint8_t(TypeCode::I64))) return false;
    // h = ((shape>>S1) ^ (shape>>S2) + hashConst) & (NumEntries-1)  ; entry in propTaggedLocal
    if (!GetLocal(e, be.propShapeLocal) || !e.writeOp(Op::I32Const) ||
        !e.writeVarS32(MC::ShapeHashShift1) || !e.writeOp(Op::I32ShrU) ||
        !GetLocal(e, be.propShapeLocal) || !e.writeOp(Op::I32Const) ||
        !e.writeVarS32(MC::ShapeHashShift2) || !e.writeOp(Op::I32ShrU) ||
        !e.writeOp(Op::I32Xor) || !e.writeOp(Op::I32Const) ||
        !e.writeVarS32(int32_t(megaHashConst)) || !e.writeOp(Op::I32Add) ||
        !e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(MC::NumEntries - 1)) ||
        !e.writeOp(Op::I32And) || !e.writeOp(Op::I32Const) ||
        !e.writeVarS32(int32_t(sizeof(MC::Entry))) || !e.writeOp(Op::I32Mul) ||
        !e.writeOp(Op::I32Const) ||
        !e.writeVarS32(int32_t(megaCacheAddr + MC::offsetOfEntries())) ||
        !e.writeOp(Op::I32Add) || !e.writeOp(Op::LocalSet) ||
        !e.writeVarU32(be.propTaggedLocal))
      return false;
    // cond = shape==entry.shape & key==entry.key & gen==entry.gen & entry.hopsAndKind==0
    if (!GetLocal(e, be.propTaggedLocal) || !e.writeOp(Op::I32Load) ||
        !e.writeVarU32(2) || !e.writeVarU32(uint32_t(ME::offsetOfShape())) ||
        !GetLocal(e, be.propShapeLocal) || !e.writeOp(Op::I32Eq))
      return false;
    if (!GetLocal(e, be.propTaggedLocal) || !e.writeOp(Op::I32Load) ||
        !e.writeVarU32(2) || !e.writeVarU32(uint32_t(ME::offsetOfKey())) ||
        !e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(uint32_t(keyBits))) ||
        !e.writeOp(Op::I32Eq) || !e.writeOp(Op::I32And))
      return false;
    if (!GetLocal(e, be.propTaggedLocal) || !e.writeOp(Op::I32Load16U) ||
        !e.writeVarU32(1) || !e.writeVarU32(uint32_t(ME::offsetOfGeneration())) ||
        !e.writeOp(Op::I32Const) ||
        !e.writeVarS32(int32_t(megaCacheAddr + MC::offsetOfGeneration())) ||
        !e.writeOp(Op::I32Load16U) || !e.writeVarU32(1) || !e.writeVarU32(0) ||
        !e.writeOp(Op::I32Eq) || !e.writeOp(Op::I32And))
      return false;
    if (!GetLocal(e, be.propTaggedLocal) || !e.writeOp(Op::I32Load8U) ||
        !e.writeVarU32(0) || !e.writeVarU32(uint32_t(ME::offsetOfHopsAndKind())) ||
        !e.writeOp(Op::I32Eqz) || !e.writeOp(Op::I32And))
      return false;
    if (!e.writeOp(Op::If) || !e.writeFixedU8(uint8_t(TypeCode::I64))) return false;
    // hit: tag = entry.slotOffset (reuse propShapeLocal); base = (tag&1)?obj:obj->slots; +(tag>>1)
    if (!GetLocal(e, be.propTaggedLocal) || !e.writeOp(Op::I32Load) ||
        !e.writeVarU32(2) || !e.writeVarU32(uint32_t(ME::offsetOfSlotOffset())) ||
        !e.writeOp(Op::LocalSet) || !e.writeVarU32(be.propShapeLocal))
      return false;
    if (!GetLocal(e, be.propObjLocal) || !GetLocal(e, be.propObjLocal) ||
        !e.writeOp(Op::I32Load) || !e.writeVarU32(2) ||
        !e.writeVarU32(uint32_t(js::NativeObject::offsetOfSlots())) ||
        !GetLocal(e, be.propShapeLocal) || !e.writeOp(Op::I32Const) ||
        !e.writeVarS32(1) || !e.writeOp(Op::I32And) || !e.writeOp(Op::SelectNumeric))
      return false;  // (tag&1)?obj:slots
    if (!GetLocal(e, be.propShapeLocal) || !e.writeOp(Op::I32Const) ||
        !e.writeVarS32(1) || !e.writeOp(Op::I32ShrU) || !e.writeOp(Op::I32Add) ||
        !e.writeOp(Op::I64Load) || !e.writeVarU32(3) || !e.writeVarU32(0))
      return false;  // [value]
    if (!e.writeOp(Op::Else)) return false;    // cond (shape/key/gen/hops) mismatch
    if (!emitHelperMiss()) return false;
    if (!e.writeOp(Op::End)) return false;     // close cond-if
    if (!e.writeOp(Op::Else)) return false;    // !allFull (site not yet megamorphic)
    if (!emitHelperMiss()) return false;
    return e.writeOp(Op::End);                  // close allFull-if
  };
  // Runtime-cache ways + helper miss path (the polymorphic fallback). Each way:
  // gWJPropShape[base+w] == propShape ? load that way's (runtime) tagged offset.
  auto emitWaysAndHelper = [&]() -> bool {
    for (uint32_t w = 0; w < kWJPropWays; w++) {
      if (!e.writeOp(Op::I32Const) ||
          !e.writeVarS32(int32_t(shapeArr + (base + w) * 4)) ||
          !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0) ||
          !GetLocal(e, be.propShapeLocal) || !e.writeOp(Op::I32Eq))
        return false;
      if (!e.writeOp(Op::If) || !e.writeFixedU8(uint8_t(TypeCode::I64)))
        return false;  // both arms yield i64
      if (!emitSlotLoad(w)) return false;
      if (!e.writeOp(Op::Else)) return false;
    }
    if (!emitMegaOrMiss()) return false;
    for (uint32_t w = 0; w < kWJPropWays; w++) {
      if (!e.writeOp(Op::End)) return false;
    }
    return true;
  };

  // BAKED MONOMORPHIC FAST PATH (Ion-like): for a read whose receiver shape is
  // known at compile time (a GuardShape site), guard obj->shape == the baked
  // (interned, relocation-safe) shape and load at a COMPILE-TIME-CONSTANT offset --
  // no runtime offset load, no fixed/dynamic tagged-bit branch (those are the
  // EmitPropIC overhead that made mega ~10% slower than a plain guarded load on
  // monomorphic benches like splay). On a shape mismatch (polymorphic receiver, or
  // a moved-then-different shape) fall to the runtime cache + helper -- NO deopt,
  // so still correct + fast for deltablue's polymorphic constraint methods.
  uint32_t bakedSlot = bakedShape ? WJInternShape(uintptr_t(bakedShape)) : 0;
  if (bakedShape && bakedSlot) {
    if (!GetLocal(e, be.propShapeLocal)) return false;
    if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(bakedSlot)) ||
        !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0))
      return false;  // live (relocated) baked shape from the pool
    if (!e.writeOp(Op::I32Eq)) return false;
    if (!e.writeOp(Op::If) || !e.writeFixedU8(uint8_t(TypeCode::I64))) return false;
    // then: baked constant-offset load.
    if (!GetLocal(e, be.propObjLocal)) return false;
    if (!bakedFixed) {  // dynamic slot: base = obj->slots
      if (!e.writeOp(Op::I32Load) || !e.writeVarU32(2) ||
          !e.writeVarU32(uint32_t(js::NativeObject::offsetOfSlots())))
        return false;
    }
    if (bakedByteOff) {
      if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(bakedByteOff)) ||
          !e.writeOp(Op::I32Add))
        return false;
    }
    if (!e.writeOp(Op::I64Load) || !e.writeVarU32(3) || !e.writeVarU32(0))
      return false;
    if (!e.writeOp(Op::Else)) return false;
    if (!emitWaysAndHelper()) return false;
    return e.writeOp(Op::End);
  }
  return emitWaysAndHelper();
}

// Inline polymorphic GetProperty IC for `obj.name` (constant-atom key). Unlike the
// bare WJH_GETPROP helper (a full VM GetProperty per access -- gbemu's chained
// `a.b.c` property-access wall), this self-guards a per-site runtime shape->offset
// cache and loads the slot INLINE on a hit (any cached shape); on a miss (or a
// non-object receiver) it calls WJH_PROPIC, which fills the cache for native data
// props and otherwise does a generic VM get (proto/accessor/non-object/throw). NO
// deopt -- correct for any receiver. The receiver is a BOXED value (post
// prop-unbox-elision). Leaves the boxed result (i64) on the stack. The key is
// staged in gWJPropKey[site]. GECKO_WJ_GETPROPIC gates it.
static bool EmitGetPropIC(Encoder& e, WJBackend& be, MInstruction* ins,
                          MDefinition* recv, uint64_t keyBits, uint32_t site) {
  uint32_t base = site * kWJPropWays;
  if (gWJPropKey[site] != keyBits)
    for (uint32_t w = 0; w < kWJPropWays; w++) gWJPropShape[base + w] = 0;
  gWJPropKey[site] = keyBits;
  uintptr_t shapeArr = uintptr_t(static_cast<void*>(&gWJPropShape[0]));
  uintptr_t offArr = uintptr_t(static_cast<void*>(&gWJPropOff[0]));
  // Stage the boxed receiver into gWJScratch[0] (WJH_PROPIC reads it). No GC can
  // move it between here and the helper call (the ways path is GC-free).
  if (!EmitStageScratch(e, be, recv, 0)) return false;
  auto loadRecvBoxed = [&]() -> bool {  // gWJScratch[0] via scratchBase local
    return GetLocal(e, be.scratchBase) && e.writeOp(Op::I64Load) &&
           e.writeVarU32(3) && e.writeVarU32(0);
  };
  // tag = (recv >> 32) ; isObject = tag == TagWord(OBJECT)
  if (!loadRecvBoxed() || !e.writeOp(Op::I64Const) || !e.writeVarS64(32) ||
      !e.writeOp(Op::I64ShrU) || !e.writeOp(Op::I32WrapI64) ||
      !e.writeOp(Op::I32Const) ||
      !e.writeVarS32(int32_t(TagWord(JSVAL_TYPE_OBJECT))) || !e.writeOp(Op::I32Eq))
    return false;
  if (!e.writeOp(Op::If) || !e.writeFixedU8(uint8_t(TypeCode::I64))) return false;
  // --- object branch: shape-cache fast path ---
  if (!loadRecvBoxed() || !e.writeOp(Op::I32WrapI64) || !e.writeOp(Op::LocalSet) ||
      !e.writeVarU32(be.propObjLocal))
    return false;
  if (!GetLocal(e, be.propObjLocal) || !e.writeOp(Op::I32Load) || !e.writeVarU32(2) ||
      !e.writeVarU32(uint32_t(offsetof(JS::shadow::Object, shape))) ||
      !e.writeOp(Op::LocalSet) || !e.writeVarU32(be.propShapeLocal))
    return false;
  uintptr_t gpHolderArr = uintptr_t(static_cast<void*>(&gWJPropHolder[0]));
  auto emitSlotLoad = [&](uint32_t w) -> bool {
    // tagged offset -> propTaggedLocal; MISSING sentinel -> undefined (no load).
    if (!e.writeOp(Op::I32Const) ||
        !e.writeVarS32(int32_t(offArr + (base + w) * 4)) ||
        !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0) ||
        !e.writeOp(Op::LocalTee) || !e.writeVarU32(be.propTaggedLocal) ||
        !e.writeOp(Op::I32Const) ||
        !e.writeVarS32(int32_t(kWJPropMissingSentinel)) || !e.writeOp(Op::I32Eq))
      return false;
    if (!e.writeOp(Op::If) || !e.writeFixedU8(uint8_t(TypeCode::I64))) return false;
    if (!e.writeOp(Op::I64Const) ||
        !e.writeVarS64(int64_t(JS::UndefinedValue().asRawBits())))
      return false;
    if (!e.writeOp(Op::Else)) return false;
    // objBase = gWJPropHolder[w] ? holder : receiver (proto-read support). objBase
    // scratch reuses propShapeLocal (free in the matched way's then-branch).
    if (!e.writeOp(Op::I32Const) ||
        !e.writeVarS32(int32_t(gpHolderArr + (base + w) * 4)) ||
        !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0) ||
        !e.writeOp(Op::LocalTee) || !e.writeVarU32(be.propShapeLocal))
      return false;                                      // [holder], propShapeLocal=holder
    if (!GetLocal(e, be.propObjLocal)) return false;     // [holder, receiver]
    if (!GetLocal(e, be.propShapeLocal)) return false;   // [holder, receiver, holder(cond)]
    if (!e.writeOp(Op::SelectNumeric)) return false;            // [objBase]
    if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(be.propShapeLocal))
      return false;                                      // propShapeLocal = objBase
    if (!GetLocal(e, be.propTaggedLocal) || !e.writeOp(Op::I32Const) ||
        !e.writeVarS32(1) || !e.writeOp(Op::I32And))
      return false;
    if (!e.writeOp(Op::If) || !e.writeFixedU8(uint8_t(TypeCode::I32))) return false;
    if (!GetLocal(e, be.propShapeLocal)) return false;
    if (!e.writeOp(Op::Else)) return false;
    if (!GetLocal(e, be.propShapeLocal) || !e.writeOp(Op::I32Load) ||
        !e.writeVarU32(2) ||
        !e.writeVarU32(uint32_t(js::NativeObject::offsetOfSlots())))
      return false;
    if (!e.writeOp(Op::End)) return false;
    if (!GetLocal(e, be.propTaggedLocal) || !e.writeOp(Op::I32Const) ||
        !e.writeVarS32(1) || !e.writeOp(Op::I32ShrU) || !e.writeOp(Op::I32Add))
      return false;
    if (!e.writeOp(Op::I64Load) || !e.writeVarU32(3) || !e.writeVarU32(0))
      return false;
    return e.writeOp(Op::End);  // end missing-sentinel If
  };
  for (uint32_t w = 0; w < kWJPropWays; w++) {
    if (!e.writeOp(Op::I32Const) ||
        !e.writeVarS32(int32_t(shapeArr + (base + w) * 4)) ||
        !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0) ||
        !GetLocal(e, be.propShapeLocal) || !e.writeOp(Op::I32Eq))
      return false;
    if (!e.writeOp(Op::If) || !e.writeFixedU8(uint8_t(TypeCode::I64))) return false;
    if (!emitSlotLoad(w)) return false;
    if (!e.writeOp(Op::Else)) return false;
  }
  // object miss: scratch[0] already staged; WJH_PROPIC fills cache + returns value
  if (!EmitHelperCallResult(e, be, ins, WJH_PROPIC, site)) return false;
  for (uint32_t w = 0; w < kWJPropWays; w++)
    if (!e.writeOp(Op::End)) return false;
  // --- else (non-object): generic get via WJH_PROPIC ---
  if (!e.writeOp(Op::Else)) return false;
  if (!EmitHelperCallResult(e, be, ins, WJH_PROPIC, site)) return false;
  return e.writeOp(Op::End);  // end is-object if (yields i64)
}

// GetName IC fast path. Loads the per-site cached holder (gWJNameHolder[site], a
// traced realm-singleton global object/lexical); if present and its shape still
// matches, loads the cached slot directly (the value may have mutated, the slot
// location hasn't). On holder==0 or shape mismatch, stages env+name and calls
// wjhelp(WJH_GETNAME, site) which resolves + fills the cache + returns the value.
// Leaves the boxed value (i64) on the stack. `nameBits` = StringValue(name) bits.
static bool EmitNameIC(Encoder& e, WJBackend& be, MInstruction* ins,
                       MDefinition* envOp, uint64_t nameBits, uint32_t site,
                       bool isTypeof = false) {
  gWJNameKey[site] = nameBits;
  uintptr_t holderAddr = uintptr_t(static_cast<void*>(&gWJNameHolder[site]));
  uintptr_t shapeAddr = uintptr_t(static_cast<void*>(&gWJNameShape[site]));
  uintptr_t offAddr = uintptr_t(static_cast<void*>(&gWJNameOff[site]));
  // holder = load gWJNameHolder[site] -> propObjLocal (reuse prop-IC scratch).
  if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(holderAddr)) ||
      !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0) ||
      !e.writeOp(Op::LocalSet) || !e.writeVarU32(be.propObjLocal))
    return false;
  // valid = holder!=0 ? (holder->shape == gWJNameShape[site]) : 0
  if (!GetLocal(e, be.propObjLocal) || !e.writeOp(Op::I32Eqz)) return false;
  if (!e.writeOp(Op::If) || !e.writeFixedU8(uint8_t(TypeCode::I32))) return false;
  if (!e.writeOp(Op::I32Const) || !e.writeVarS32(0)) return false;  // holder==0
  if (!e.writeOp(Op::Else)) return false;
  if (!GetLocal(e, be.propObjLocal) || !e.writeOp(Op::I32Load) ||
      !e.writeVarU32(2) ||
      !e.writeVarU32(uint32_t(offsetof(JS::shadow::Object, shape))) ||
      !e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(shapeAddr)) ||
      !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0) ||
      !e.writeOp(Op::I32Eq))
    return false;
  if (!e.writeOp(Op::End)) return false;
  // if (valid) load cached slot else miss-helper. Both arms yield i64.
  if (!e.writeOp(Op::If) || !e.writeFixedU8(uint8_t(TypeCode::I64))) return false;
  // hit: tagged = gWJNameOff[site]; base = (tagged&1)?holder:holder->slots_;
  //      value = I64Load(base + (tagged>>1)).
  if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(offAddr)) ||
      !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0) ||
      !e.writeOp(Op::LocalSet) || !e.writeVarU32(be.propTaggedLocal))
    return false;
  if (!GetLocal(e, be.propTaggedLocal) || !e.writeOp(Op::I32Const) ||
      !e.writeVarS32(1) || !e.writeOp(Op::I32And))
    return false;
  if (!e.writeOp(Op::If) || !e.writeFixedU8(uint8_t(TypeCode::I32))) return false;
  if (!GetLocal(e, be.propObjLocal)) return false;
  if (!e.writeOp(Op::Else)) return false;
  if (!GetLocal(e, be.propObjLocal) || !e.writeOp(Op::I32Load) ||
      !e.writeVarU32(2) ||
      !e.writeVarU32(uint32_t(js::NativeObject::offsetOfSlots())))
    return false;
  if (!e.writeOp(Op::End)) return false;
  if (!GetLocal(e, be.propTaggedLocal) || !e.writeOp(Op::I32Const) ||
      !e.writeVarS32(1) || !e.writeOp(Op::I32ShrU) || !e.writeOp(Op::I32Add))
    return false;
  if (!e.writeOp(Op::I64Load) || !e.writeVarU32(3) || !e.writeVarU32(0))
    return false;
  if (!e.writeOp(Op::Else)) return false;
  // miss: stage env + name, call WJH_GETNAME(site) (resolves + fills cache).
  // isTypeof (encoded in the site's high bit) makes the handler's not-found slow
  // path use GetNameMode::TypeOf (return undefined, DON'T throw) -- mirrors the
  // interpreter's GetNameOperation typeof kludge. Fixes `typeof UndefinedGlobal`
  // throwing under JIT (real-site feature-detection: typeof window/Symbol/etc.).
  if (!EmitStageScratch(e, be, envOp, 0)) return false;
  if (!EmitStageConstBoxed(e, be, nameBits, 1)) return false;
  uint32_t siteArg = site | (isTypeof ? 0x80000000u : 0u);
  if (!EmitHelperCallResult(e, be, ins, WJH_GETNAME, siteArg)) return false;
  return e.writeOp(Op::End);
}

// Emit a guarded incremental-GC pre-write barrier on an OLD value about to be
// overwritten. `emitOldValue` must push the old boxed Value (i64) on the stack.
// Fast path (the common case, flag clear): load+test the zone marking flag and
// skip. Only when an incremental GC is in progress do we stash the old value in
// gWJHelpVal and call WJH_PREBARRIER (gc::ValuePreWriteBarrier). Lets object/
// unknown-typed slot+element stores stay in JIT instead of bailing to PBL.
template <typename F>
static bool EmitGuardedValuePreBarrier(Encoder& e, WJBackend& be, F emitOldValue) {
  uintptr_t flagAddr = gWJMarkBarrierAddr;
  if (!flagAddr) return false;  // address not baked -> caller bails (sound)
  uintptr_t valAddr = uintptr_t(static_cast<void*>(&gWJHelpVal));
  if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(flagAddr)) ||
      !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0) ||
      !e.writeOp(Op::I32Const) || !e.writeVarS32(1) || !e.writeOp(Op::I32And))
    return false;
  if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
  if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(valAddr))) return false;
  if (!emitOldValue()) return false;  // pushes the old i64 value
  if (!e.writeOp(Op::I64Store) || !e.writeVarU32(3) || !e.writeVarU32(0))
    return false;
  if (!e.writeOp(Op::F64Const) || !e.writeFixedF64(double(WJH_PREBARRIER)) ||
      !e.writeOp(Op::F64Const) || !e.writeFixedF64(0.0) ||
      !e.writeOp(Op::Call) || !e.writeVarU32(0) || !e.writeOp(Op::Drop))
    return false;
  return e.writeOp(Op::End);
}

// Inline property-STORE IC for a writable own data property (SetPropertyCache /
// named set). Mirrors EmitPropIC: matches obj->shape() against per-site cached
// shapes; on a hit stores the value into the slot inline, with a guarded
// incremental pre-write barrier on the old value and a (self-guarded) whole-cell
// post-write barrier when the stored value may be a GC pointer. On a miss it
// calls wjhelp(WJH_SETPROPIC), which converts idval, fills the IC for a writable
// own data property, and stores (or falls back to the generic set). This is an
// EFFECT: it leaves nothing on the stack. `site` must be nonzero.
static bool EmitPropStoreIC(Encoder& e, WJBackend& be, MInstruction* ins,
                            MDefinition* object, MDefinition* idval,
                            MDefinition* value, bool strict, uint32_t site,
                            uint64_t constKeyValBits = 0) {
  gWJPropStrict[site] = strict ? 1 : 0;
  uint32_t base = site * kWJPropWays;
  uintptr_t shapeArr = uintptr_t(static_cast<void*>(&gWJPropShape[0]));
  uintptr_t offArr = uintptr_t(static_cast<void*>(&gWJPropOff[0]));
  uint32_t shapeOff = uint32_t(offsetof(JS::shadow::Object, shape));
  if (object->type() == MIRType::Object) {
    // propObj = receiver pointer (rematerialized); propShape = its shape.
    if (!GetOp(e, be, object)) return false;
    if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(be.propObjLocal)) return false;
    if (!GetLocal(e, be.propObjLocal)) return false;
    if (!e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(shapeOff))
      return false;
    if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(be.propShapeLocal)) return false;
  } else {
    // BOXED (Value) receiver -- richards' polymorphic Task/Packet refs are typed
    // Value, so the store IC must guard the object tag (mirrors EmitGetPropIC).
    // propShape = isObject ? load(unbox->shape) : -1 (sentinel never matches a way,
    // so a non-object receiver cleanly misses -> WJH_SETPROPIC handles it). The
    // shape load is GUARDED so a non-object's bits are never dereferenced.
    if (!GetOp(e, be, object) || !e.writeOp(Op::I64Const) || !e.writeVarS64(32) ||
        !e.writeOp(Op::I64ShrU) || !e.writeOp(Op::I32WrapI64) ||
        !e.writeOp(Op::I32Const) ||
        !e.writeVarS32(int32_t(TagWord(JSVAL_TYPE_OBJECT))) || !e.writeOp(Op::I32Eq))
      return false;
    if (!e.writeOp(Op::If) || !e.writeFixedU8(uint8_t(TypeCode::I32))) return false;
    if (!GetOp(e, be, object) || !e.writeOp(Op::I32WrapI64) ||
        !e.writeOp(Op::LocalSet) || !e.writeVarU32(be.propObjLocal))
      return false;
    if (!GetLocal(e, be.propObjLocal) || !e.writeOp(Op::I32Load) ||
        !e.writeVarU32(2) || !e.writeVarU32(shapeOff))
      return false;
    if (!e.writeOp(Op::Else)) return false;
    if (!e.writeOp(Op::I32Const) || !e.writeVarS32(-1)) return false;
    if (!e.writeOp(Op::End)) return false;
    if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(be.propShapeLocal)) return false;
  }
  // boxed value -> unboxScratch (reused for store + post-barrier inspection).
  if (!EmitSpillValue(e, be, value)) return false;
  if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(be.unboxScratch)) return false;

  // boxed KEY -> propKeyLocal. Every way hit additionally guards the key against
  // the way's cached atom (gWJPropWayKey): a dynamic-key site (`obj[k] = v`) sees
  // many keys per receiver shape, and a shape-only match stores into the OTHER
  // key's slot (babylon Node.__clone corrupted cloned AST nodes). Constant-key
  // sites pay one extra compare per way and gain site-reuse staleness safety.
  if (constKeyValBits) {
    if (!e.writeOp(Op::I64Const) || !e.writeVarS64(int64_t(constKeyValBits)) ||
        !e.writeOp(Op::LocalSet) || !e.writeVarU32(be.propKeyLocal))
      return false;
  } else {
    if (!EmitSpillValue(e, be, idval)) return false;
    if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(be.propKeyLocal)) return false;
  }

  MIRType vt = value->type();
  bool valMaybeGC = (vt == MIRType::Object || vt == MIRType::Value ||
                     vt == MIRType::String || vt == MIRType::Symbol ||
                     vt == MIRType::BigInt);

  static int storeValidate = WJGETENV("GECKO_WJ_STOREVALIDATE") ? 1 : 0;
  auto emitCheckCell = [&](uint32_t loc, double siteId) -> bool {  // loc = i32 ptr local
    // REAL validator: load the cell's first word (shape ptr for an object); if it
    // is the swept-nursery poison (0x2B2B2B2B), the cell was FREED -> report+crash.
    // Only then call WJH_CHECKCELL (which unconditionally crashes). Without this
    // guard the probe crashed on EVERY store (false positive).
    uintptr_t objAddr = uintptr_t(static_cast<void*>(&gWJHelpObj));
    if (!GetLocal(e, loc) || !e.writeOp(Op::I32Load) || !e.writeVarU32(2) ||
        !e.writeVarU32(0))
      return false;  // *(cell) = shape ptr (or poison)
    if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(0x2B2B2B2B)) ||
        !e.writeOp(Op::I32Eq))
      return false;
    if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;  // poisoned?
    if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(objAddr)) ||
        !GetLocal(e, loc) || !e.writeOp(Op::I32Store) || !e.writeVarU32(2) ||
        !e.writeVarU32(0))
      return false;
    if (!e.writeOp(Op::F64Const) || !e.writeFixedF64(double(WJH_CHECKCELL)) ||
        !e.writeOp(Op::F64Const) || !e.writeFixedF64(siteId) ||
        !e.writeOp(Op::Call) || !e.writeVarU32(0) || !e.writeOp(Op::Drop))
      return false;
    return e.writeOp(Op::End);
  };
  auto emitStore = [&](uint32_t w) -> bool {
    // propTagged = gWJPropOff[base+w]; slotAddr -> propTaggedLocal repurposed.
    if (!e.writeOp(Op::I32Const) ||
        !e.writeVarS32(int32_t(offArr + (base + w) * 4)) ||
        !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0) ||
        !e.writeOp(Op::LocalSet) || !e.writeVarU32(be.propTaggedLocal))
      return false;
    // DEBUG: validate the receiver is a live (non-forwarded) cell before storing.
    if (storeValidate && !emitCheckCell(be.propObjLocal, 1.0)) return false;  // receiver
    // slotBase = (tagged & 1) ? propObj : I32Load(propObj + offsetOfSlots)
    if (!GetLocal(e, be.propTaggedLocal) || !e.writeOp(Op::I32Const) ||
        !e.writeVarS32(1) || !e.writeOp(Op::I32And))
      return false;
    if (!e.writeOp(Op::If) || !e.writeFixedU8(uint8_t(TypeCode::I32))) return false;
    if (!GetLocal(e, be.propObjLocal)) return false;
    if (!e.writeOp(Op::Else)) return false;
    if (!GetLocal(e, be.propObjLocal) || !e.writeOp(Op::I32Load) ||
        !e.writeVarU32(2) ||
        !e.writeVarU32(uint32_t(js::NativeObject::offsetOfSlots())))
      return false;
    if (!e.writeOp(Op::End)) return false;
    // slotAddr = slotBase + (tagged >> 1)  -> propShapeLocal (free in this arm)
    if (!GetLocal(e, be.propTaggedLocal) || !e.writeOp(Op::I32Const) ||
        !e.writeVarS32(1) || !e.writeOp(Op::I32ShrU) || !e.writeOp(Op::I32Add) ||
        !e.writeOp(Op::LocalSet) || !e.writeVarU32(be.propShapeLocal))
      return false;
    // Guarded incremental pre-write barrier on the OLD slot value.
    if (!EmitGuardedValuePreBarrier(e, be, [&]() {
          return GetLocal(e, be.propShapeLocal) && e.writeOp(Op::I64Load) &&
                 e.writeVarU32(3) && e.writeVarU32(0);
        }))
      return false;
    // DEBUG: validate the VALUE payload is a live (non-forwarded) cell, if GC-typed.
    if (storeValidate && valMaybeGC) {
      if (!GetLocal(e, be.unboxScratch) || !e.writeOp(Op::I32WrapI64) ||
          !e.writeOp(Op::LocalSet) || !e.writeVarU32(be.propTaggedLocal))
        return false;
      if (!emitCheckCell(be.propTaggedLocal, 2.0)) return false;  // stored value
    }
    // Store: *(slotAddr) = value.
    if (!GetLocal(e, be.propShapeLocal) || !GetLocal(e, be.unboxScratch) ||
        !e.writeOp(Op::I64Store) || !e.writeVarU32(3) || !e.writeVarU32(0))
      return false;
    // Inline guarded whole-cell post-write barrier. Buffer obj ONLY when obj is
    // tenured AND the stored value is a nursery GC cell (mirrors Ion). The nursery
    // test loads chunk(ptr)->storeBuffer (nonzero == nursery); chunkBase = ptr &
    // ~ChunkMask. The payload deref is short-circuited behind the GC-thing check
    // so a non-GC Value never dereferences a non-pointer payload.
    if (valMaybeGC) {
      const uint32_t chunkMaskInv = uint32_t(~js::gc::ChunkMask);
      const uint32_t sbOff = uint32_t(js::gc::ChunkStoreBufferOffset);
      // objTenured = (chunk(obj)->storeBuffer == 0)
      if (!GetLocal(e, be.propObjLocal) || !e.writeOp(Op::I32Const) ||
          !e.writeVarS32(int32_t(chunkMaskInv)) || !e.writeOp(Op::I32And) ||
          !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(sbOff) ||
          !e.writeOp(Op::I32Eqz))
        return false;
      // isGCThing
      if (vt == MIRType::Value) {
        // (unboxScratch >> 32) u>= JSVAL_TAG_STRING
        if (!GetLocal(e, be.unboxScratch) || !e.writeOp(Op::I64Const) ||
            !e.writeVarS64(32) || !e.writeOp(Op::I64ShrU) ||
            !e.writeOp(Op::I32WrapI64) || !e.writeOp(Op::I32Const) ||
            !e.writeVarS32(int32_t(uint32_t(JSVAL_TAG_STRING))) ||
            !e.writeOp(Op::I32GeU))
          return false;
      } else if (!e.writeOp(Op::I32Const) || !e.writeVarS32(1)) {
        return false;  // Object/String/Symbol/BigInt: always a GC thing
      }
      if (!e.writeOp(Op::I32And)) return false;  // objTenured && isGCThing
      if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
      // valNursery = chunk(payload)->storeBuffer (nonzero => nursery), as a boolean.
      if (!GetLocal(e, be.unboxScratch) || !e.writeOp(Op::I32WrapI64) ||
          !e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(chunkMaskInv)) ||
          !e.writeOp(Op::I32And) || !e.writeOp(Op::I32Load) || !e.writeVarU32(2) ||
          !e.writeVarU32(sbOff) || !e.writeOp(Op::I32Const) || !e.writeVarS32(0) ||
          !e.writeOp(Op::I32Ne))
        return false;
      // Ion's one-element whole-cell cache: && obj != *gWJWholeCellLastAddr (skip the
      // helper when obj is already the last buffered cell -- see EmitForcePostBarrier).
      static int noLastCellS = WJGETENV("GECKO_WJ_NOLASTCELL") ? 1 : 0;
      if (!noLastCellS && gWJWholeCellLastAddr) {
        if (!GetLocal(e, be.propObjLocal) || !e.writeOp(Op::I32Const) ||
            !e.writeVarS32(int32_t(gWJWholeCellLastAddr)) || !e.writeOp(Op::I32Load) ||
            !e.writeVarU32(2) || !e.writeVarU32(0) || !e.writeOp(Op::I32Ne) ||
            !e.writeOp(Op::I32And))
          return false;
      }
      if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
      uintptr_t objAddr = uintptr_t(static_cast<void*>(&gWJHelpObj));
      if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(objAddr)) ||
          !GetLocal(e, be.propObjLocal) || !e.writeOp(Op::I32Store) ||
          !e.writeVarU32(2) || !e.writeVarU32(0))
        return false;
      if (!e.writeOp(Op::F64Const) || !e.writeFixedF64(double(WJH_POSTBARRIER)) ||
          !e.writeOp(Op::F64Const) || !e.writeFixedF64(0.0) ||
          !e.writeOp(Op::Call) || !e.writeVarU32(0) || !e.writeOp(Op::Drop))
        return false;
      if (!e.writeOp(Op::End) || !e.writeOp(Op::End)) return false;
    }
    return true;
  };

  // Miss: stage obj, key, value; call WJH_SETPROPIC (fills + stores); drop.
  // The key is either an MDefinition (SetPropertyCache) or a constant PropertyKey
  // value (MegamorphicStoreSlot, which has no idval node).
  auto emitMissHelper = [&]() -> bool {
    if (!EmitStageScratch(e, be, object, 0)) return false;
    if (constKeyValBits) {
      if (!EmitStageConstBoxed(e, be, constKeyValBits, 1)) return false;
    } else if (!EmitStageScratch(e, be, idval, 1)) {
      return false;
    }
    if (!EmitStageScratch(e, be, value, 2)) return false;
    if (!EmitHelperCallResult(e, be, ins, WJH_SETPROPIC, site)) return false;
    return e.writeOp(Op::Drop);
  };
  uintptr_t wayKeyArr = uintptr_t(static_cast<void*>(&gWJPropWayKey[0]));
  auto emitNamedWays = [&]() -> bool {
    for (uint32_t w = 0; w < kWJPropWays; w++) {
      if (!e.writeOp(Op::I32Const) ||
          !e.writeVarS32(int32_t(shapeArr + (base + w) * 4)) ||
          !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0) ||
          !GetLocal(e, be.propShapeLocal) || !e.writeOp(Op::I32Eq))
        return false;
      // AND the key guard: boxed idval == StringValue(way's cached atom).
      if (!GetLocal(e, be.propKeyLocal) || !e.writeOp(Op::I32Const) ||
          !e.writeVarS32(int32_t(wayKeyArr + (base + w) * 4)) ||
          !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0) ||
          !e.writeOp(Op::I64ExtendI32U) || !e.writeOp(Op::I64Const) ||
          !e.writeVarS64(int64_t(uint64_t(TagWord(JSVAL_TYPE_STRING)) << 32)) ||
          !e.writeOp(Op::I64Or) || !e.writeOp(Op::I64Eq) ||
          !e.writeOp(Op::I32And))
        return false;
      if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;  // void
      if (!emitStore(w)) return false;
      if (!e.writeOp(Op::Else)) return false;
    }
    // ADD-IC way: a property-ADD (the receiver is in the exact pre-add shape) is
    // done INLINE -- set obj->shape = cached newShape, store the value at the cached
    // fixed slot, post-barrier -- instead of the WJH_SETPROPIC -> SetObjectElement
    // C++ hop. The cached shapes live in the traced gWJShapePool (GC-current, so the
    // newShape written into the object is valid). The shape pre-write barrier is
    // safely elided (an ADD transition's newShape has oldShape as its tree parent,
    // SATB holds -- see AddAndStoreSlot). GECKO_WJ_ADDIC gates it; runtime fills the
    // cache on the first add at the site (splay's Node ctor inits, earley conses...).
    static int addIC = WJGETENV("GECKO_WJ_NOADDIC") ? 0 : 1;  // DEFAULT-ON
    if (addIC) {
      uintptr_t oldA = uintptr_t(static_cast<void*>(&gWJAddOldShape[site]));
      uintptr_t newA = uintptr_t(static_cast<void*>(&gWJAddNewShape[site]));
      uintptr_t offA = uintptr_t(static_cast<void*>(&gWJAddOff[site]));
      // poolAddrOld = gWJAddOldShape[site]; if (poolAddrOld != 0) ...
      if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(oldA)) ||
          !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0) ||
          !e.writeOp(Op::LocalTee) || !e.writeVarU32(be.propTaggedLocal) ||
          !e.writeOp(Op::I32Const) || !e.writeVarS32(0) || !e.writeOp(Op::I32Ne))
        return false;
      if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
      // (*poolAddrOld == obj->shape) ?
      if (!GetLocal(e, be.propTaggedLocal) || !e.writeOp(Op::I32Load) ||
          !e.writeVarU32(2) || !e.writeVarU32(0) || !GetLocal(e, be.propShapeLocal) ||
          !e.writeOp(Op::I32Eq))
        return false;
      // AND the add-key guard: boxed idval == StringValue(gWJAddKey[site]) --
      // one dynamic site ADDS many different keys from the same pre-add shape.
      uintptr_t addKeyA = uintptr_t(static_cast<void*>(&gWJAddKey[site]));
      if (!GetLocal(e, be.propKeyLocal) || !e.writeOp(Op::I32Const) ||
          !e.writeVarS32(int32_t(addKeyA)) || !e.writeOp(Op::I32Load) ||
          !e.writeVarU32(2) || !e.writeVarU32(0) ||
          !e.writeOp(Op::I64ExtendI32U) || !e.writeOp(Op::I64Const) ||
          !e.writeVarS64(int64_t(uint64_t(TagWord(JSVAL_TYPE_STRING)) << 32)) ||
          !e.writeOp(Op::I64Or) || !e.writeOp(Op::I64Eq) ||
          !e.writeOp(Op::I32And))
        return false;
      if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
      // INLINE ADD. obj->shape = *(*newA)
      uint32_t shapeOff = uint32_t(offsetof(JS::shadow::Object, shape));
      if (!GetLocal(e, be.propObjLocal) || !e.writeOp(Op::I32Const) ||
          !e.writeVarS32(int32_t(newA)) || !e.writeOp(Op::I32Load) ||
          !e.writeVarU32(2) || !e.writeVarU32(0) || !e.writeOp(Op::I32Load) ||
          !e.writeVarU32(2) || !e.writeVarU32(0) || !e.writeOp(Op::I32Store) ||
          !e.writeVarU32(2) || !e.writeVarU32(shapeOff))
        return false;
      // *(obj + *offA) = value
      if (!GetLocal(e, be.propObjLocal) || !e.writeOp(Op::I32Const) ||
          !e.writeVarS32(int32_t(offA)) || !e.writeOp(Op::I32Load) ||
          !e.writeVarU32(2) || !e.writeVarU32(0) || !e.writeOp(Op::I32Add) ||
          !GetLocal(e, be.unboxScratch) || !e.writeOp(Op::I64Store) ||
          !e.writeVarU32(3) || !e.writeVarU32(0))
        return false;
      // post-write barrier (value may be a nursery GC cell into a tenured obj).
      if (valMaybeGC) {
        const uint32_t chunkMaskInv = uint32_t(~js::gc::ChunkMask);
        const uint32_t sbOff = uint32_t(js::gc::ChunkStoreBufferOffset);
        if (!GetLocal(e, be.propObjLocal) || !e.writeOp(Op::I32Const) ||
            !e.writeVarS32(int32_t(chunkMaskInv)) || !e.writeOp(Op::I32And) ||
            !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(sbOff) ||
            !e.writeOp(Op::I32Eqz))
          return false;  // objTenured
        if (vt == MIRType::Value) {
          if (!GetLocal(e, be.unboxScratch) || !e.writeOp(Op::I64Const) ||
              !e.writeVarS64(32) || !e.writeOp(Op::I64ShrU) ||
              !e.writeOp(Op::I32WrapI64) || !e.writeOp(Op::I32Const) ||
              !e.writeVarS32(int32_t(uint32_t(JSVAL_TAG_STRING))) ||
              !e.writeOp(Op::I32GeU))
            return false;
        } else if (!e.writeOp(Op::I32Const) || !e.writeVarS32(1)) {
          return false;
        }
        if (!e.writeOp(Op::I32And)) return false;
        if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
        if (!GetLocal(e, be.unboxScratch) || !e.writeOp(Op::I32WrapI64) ||
            !e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(chunkMaskInv)) ||
            !e.writeOp(Op::I32And) || !e.writeOp(Op::I32Load) || !e.writeVarU32(2) ||
            !e.writeVarU32(sbOff))
          return false;
        if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
        uintptr_t objAddr = uintptr_t(static_cast<void*>(&gWJHelpObj));
        if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(objAddr)) ||
            !GetLocal(e, be.propObjLocal) || !e.writeOp(Op::I32Store) ||
            !e.writeVarU32(2) || !e.writeVarU32(0))
          return false;
        if (!e.writeOp(Op::F64Const) || !e.writeFixedF64(double(WJH_POSTBARRIER)) ||
            !e.writeOp(Op::F64Const) || !e.writeFixedF64(0.0) ||
            !e.writeOp(Op::Call) || !e.writeVarU32(0) || !e.writeOp(Op::Drop))
          return false;
        if (!e.writeOp(Op::End) || !e.writeOp(Op::End)) return false;
      }
      if (!e.writeOp(Op::Else)) return false;  // shape mismatch -> miss
      if (!emitMissHelper()) return false;
      if (!e.writeOp(Op::End)) return false;  // end shape-match if
      if (!e.writeOp(Op::Else)) return false;  // no cached add -> miss
      if (!emitMissHelper()) return false;
      if (!e.writeOp(Op::End)) return false;  // end poolAddrOld!=0 if
    } else {
      if (!emitMissHelper()) return false;
    }
    for (uint32_t w = 0; w < kWJPropWays; w++) {
      if (!e.writeOp(Op::End)) return false;
    }
    return true;
  };

  // ELEMENT-STORE IC: gbemu's 1.13M integer-index stores (arr[i]=v via
  // SetPropertyCache) all fall through the named IC as uncacheable misses ->
  // generic SetObjectElement C++ hops + a GC-root spill EACH. If the runtime key
  // is INT32 and the receiver is a dense Array with index < initializedLength,
  // store elements[index] inline (Ion's StoreElement). The clasp guard makes
  // reading elements_ sound (only NativeObject ArrayObjects); a non-array / OOB /
  // non-int key cleanly falls to the helper. GECKO_WJ_NOELEMSTOREIC disables.
  // DEFAULT-OFF (proven non-delivering, 2026-06-27): the inline typed-array element
  // store correctly eliminates gbemu's 1.13M WJH_SETPROPIC C++ hops, but A/B shows
  // that is NEUTRAL for gbemu (those stores are frequent but NOT its time sink) and
  // the runtime int-vs-named key dispatch this adds to EVERY idval store NET-REGRESSES
  // the suite (richards -5%, deltablue -4%, earley -6%). Kept opt-in (GECKO_WJ_ELEMSTOREIC)
  // -- the helper-elimination is correct and may compound with a future fix to gbemu's
  // real bottleneck (deopts / per-op render compute), but it is not a win on its own.
  static int elemStoreIC = WJGETENV("GECKO_WJ_ELEMSTOREIC") ? 1 : 0;
  bool mayBeInt = elemStoreIC && idval && !constKeyValBits &&
                  (idval->type() == MIRType::Value ||
                   idval->type() == MIRType::Int32);
  if (WJGETENV("GECKO_WJ_STOREDBG"))
    fprintf(stderr, "[wj-elemic] mayBeInt=%d idval=%p constKey=%llu idvalType=%s\n",
            mayBeInt, (void*)idval, (unsigned long long)constKeyValBits,
            idval ? StringFromMIRType(idval->type()) : "null");
  if (!mayBeInt) return emitNamedWays();

  // Stage the boxed key ONCE into propKeyLocal (the key MDefinition may be single-
  // use / not localized, and we read it for both the tag check and the index).
  if (idval->type() == MIRType::Int32) {
    // raw i32 index -> box as an INT32 Value so the tag check + payload extract
    // below are uniform with the Value-typed case.
    if (!GetOp(e, be, idval) || !e.writeOp(Op::I64ExtendI32U) ||
        !e.writeOp(Op::I64Const) ||
        !e.writeVarS64(int64_t(uint64_t(TagWord(JSVAL_TYPE_INT32))) << 32) ||
        !e.writeOp(Op::I64Or) || !e.writeOp(Op::LocalSet) ||
        !e.writeVarU32(be.propKeyLocal))
      return false;
  } else {
    if (!GetOp(e, be, idval) || !e.writeOp(Op::LocalSet) ||
        !e.writeVarU32(be.propKeyLocal))
      return false;
  }

  // Typed-array element store: addr = obj->dataOffset + (index << shift); store the
  // INT32 value payload at the element width (I32Store8/16/32 truncates to width,
  // matching JS ToInt32+truncate for an int value). Raw scalar -> NO GC barrier.
  auto emitTAStore = [&](int shift, Op storeOp, uint32_t storeAlign) -> bool {
    if (!GetLocal(e, be.propObjLocal) || !e.writeOp(Op::I32Load) ||
        !e.writeVarU32(2) ||
        !e.writeVarU32(uint32_t(js::ArrayBufferViewObject::dataOffset())))
      return false;  // dataPtr
    if (!GetLocal(e, be.propTaggedLocal)) return false;  // index
    if (shift) {
      if (!e.writeOp(Op::I32Const) || !e.writeVarS32(shift) ||
          !e.writeOp(Op::I32Shl))
        return false;
    }
    if (!e.writeOp(Op::I32Add)) return false;  // addr = dataPtr + index*width
    if (!GetLocal(e, be.unboxScratch) || !e.writeOp(Op::I32WrapI64))
      return false;  // value (int32 payload)
    return e.writeOp(storeOp) && e.writeVarU32(storeAlign) && e.writeVarU32(0);
  };
  // clasp == &fixedLengthClasses[t] ? (one of two types) -> push i32 cond
  auto emitClaspIs = [&](js::Scalar::Type t1, js::Scalar::Type t2) -> bool {
    auto one = [&](js::Scalar::Type t) -> bool {
      return GetLocal(e, be.propShapeLocal) && e.writeOp(Op::I32Const) &&
             e.writeVarS32(int32_t(
                 uintptr_t(&js::TypedArrayObject::fixedLengthClasses[t]))) &&
             e.writeOp(Op::I32Eq);
    };
    return one(t1) && one(t2) && e.writeOp(Op::I32Or);
  };

  // Compute the array index into propTaggedLocal + leave a validity i32. The key
  // may be INT32-tagged OR an integer-valued DOUBLE (gbemu computes pixel indices
  // as doubles, so ToPropertyKey turns 5.0 into index 5). Non-integer / non-number
  // keys are invalid -> fall to the named-store path.
  if (!GetLocal(e, be.propKeyLocal) || !e.writeOp(Op::I64Const) ||
      !e.writeVarS64(32) || !e.writeOp(Op::I64ShrU) || !e.writeOp(Op::I32WrapI64) ||
      !e.writeOp(Op::I32Const) ||
      !e.writeVarS32(int32_t(TagWord(JSVAL_TYPE_INT32))) || !e.writeOp(Op::I32Eq))
    return false;
  if (!e.writeOp(Op::If) || !e.writeFixedU8(uint8_t(TypeCode::I32)))
    return false;  // int32 key
  if (!GetLocal(e, be.propKeyLocal) || !e.writeOp(Op::I32WrapI64) ||
      !e.writeOp(Op::LocalSet) || !e.writeVarU32(be.propTaggedLocal) ||
      !e.writeOp(Op::I32Const) || !e.writeVarS32(1))
    return false;  // index=payload, valid=1
  if (!e.writeOp(Op::Else)) return false;
  // double key? (tag u<= JSVAL_TAG_CLEAR == 0xFFFFFF80)
  if (!GetLocal(e, be.propKeyLocal) || !e.writeOp(Op::I64Const) ||
      !e.writeVarS64(32) || !e.writeOp(Op::I64ShrU) || !e.writeOp(Op::I32WrapI64) ||
      !e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(0xFFFFFF80)) ||
      !e.writeOp(Op::I32LeU))
    return false;
  if (!e.writeOp(Op::If) || !e.writeFixedU8(uint8_t(TypeCode::I32)))
    return false;  // is double
  // index = trunc(d); valid = ((double)index == d)  (exact integer in i32 range)
  if (!GetLocal(e, be.propKeyLocal) || !e.writeOp(Op::F64ReinterpretI64) ||
      !e.writeOp(MiscOp::I32TruncSatF64S) || !e.writeOp(Op::LocalSet) ||
      !e.writeVarU32(be.propTaggedLocal))
    return false;
  if (!GetLocal(e, be.propTaggedLocal) || !e.writeOp(Op::F64ConvertI32S) ||
      !GetLocal(e, be.propKeyLocal) || !e.writeOp(Op::F64ReinterpretI64) ||
      !e.writeOp(Op::F64Eq))
    return false;
  if (!e.writeOp(Op::Else)) return false;
  if (!e.writeOp(Op::I32Const) || !e.writeVarS32(0)) return false;  // non-number key
  if (!e.writeOp(Op::End)) return false;   // end double-check
  if (!e.writeOp(Op::End)) return false;   // end int32-check (leaves validity i32)
  if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;  // valid index
  // obj is an object? (propShape != -1 sentinel -- guards the clasp deref)
  if (!GetLocal(e, be.propShapeLocal) || !e.writeOp(Op::I32Const) ||
      !e.writeVarS32(-1) || !e.writeOp(Op::I32Ne))
    return false;
  if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;  // is object
  // clasp = obj->shape->base->clasp -> propShapeLocal (shape no longer needed)
  if (!GetLocal(e, be.propShapeLocal) || !e.writeOp(Op::I32Load) ||
      !e.writeVarU32(2) ||
      !e.writeVarU32(uint32_t(js::Shape::offsetOfBaseShape())) ||
      !e.writeOp(Op::I32Load) || !e.writeVarU32(2) ||
      !e.writeVarU32(uint32_t(js::BaseShape::offsetOfClasp())) ||
      !e.writeOp(Op::LocalSet) || !e.writeVarU32(be.propShapeLocal))
    return false;
  // value is INT32-tagged? (integer typed arrays store an int; doubles/etc -> helper)
  if (!GetLocal(e, be.unboxScratch) || !e.writeOp(Op::I64Const) ||
      !e.writeVarS64(32) || !e.writeOp(Op::I64ShrU) || !e.writeOp(Op::I32WrapI64) ||
      !e.writeOp(Op::I32Const) ||
      !e.writeVarS32(int32_t(TagWord(JSVAL_TYPE_INT32))) || !e.writeOp(Op::I32Eq))
    return false;
  if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;  // value is int
  // in bounds? (uint)index < (uint)obj->length
  if (!GetLocal(e, be.propObjLocal) || !e.writeOp(Op::I32Load) ||
      !e.writeVarU32(2) ||
      !e.writeVarU32(uint32_t(js::ArrayBufferViewObject::lengthOffset())))
    return false;
  if (!GetLocal(e, be.propTaggedLocal) || !e.writeOp(Op::I32GtU))
    return false;  // length > index
  if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;  // in bounds
  // dispatch on the typed-array element type (clasp); integer types only.
  if (!emitClaspIs(js::Scalar::Uint16, js::Scalar::Int16)) return false;
  if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
  if (!emitTAStore(1, Op::I32Store16, 1)) return false;
  if (!e.writeOp(Op::Else)) return false;
  if (!emitClaspIs(js::Scalar::Uint8, js::Scalar::Int8)) return false;
  if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
  if (!emitTAStore(0, Op::I32Store8, 0)) return false;
  if (!e.writeOp(Op::Else)) return false;
  if (!emitClaspIs(js::Scalar::Int32, js::Scalar::Uint32)) return false;
  if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
  if (!emitTAStore(2, Op::I32Store, 2)) return false;
  if (!e.writeOp(Op::Else)) return false;  // float/clamped/bigint -> helper
  if (!emitMissHelper()) return false;
  if (!e.writeOp(Op::End)) return false;   // end I32/U32
  if (!e.writeOp(Op::End)) return false;   // end U8/I8
  if (!e.writeOp(Op::End)) return false;   // end U16/I16
  if (!e.writeOp(Op::Else)) return false;  // OOB -> helper
  if (!emitMissHelper()) return false;
  if (!e.writeOp(Op::End)) return false;  // end in-bounds if
  if (!e.writeOp(Op::Else)) return false;  // value not int -> helper
  if (!emitMissHelper()) return false;
  if (!e.writeOp(Op::End)) return false;  // end value-is-int if
  if (!e.writeOp(Op::Else)) return false;  // not object -> helper
  if (!emitMissHelper()) return false;
  if (!e.writeOp(Op::End)) return false;  // end is-object if
  if (!e.writeOp(Op::Else)) return false;  // key not int -> named store
  if (!emitNamedWays()) return false;
  return e.writeOp(Op::End);  // end key-is-int if
}

// Megamorphic recompile (be.forceMega) support. Find the property KEY for a
// fixed slot on a shape (the inverse of the cold-IC monomorphic GuardShape: we
// re-derive the name so a multi-shape EmitPropIC can self-guard + helper-fall-
// back instead of deopting on a polymorphic receiver). Native shapes only.
static bool WJDerivePropKey(js::Shape* shape, uint32_t slot, uint64_t* keyBits) {
  if (!shape || !shape->isShared()) return false;  // shared native shapes only
  for (js::SharedShapePropertyIter<js::NoGC> it(&shape->asShared()); !it.done();
       it++) {
    if (it->hasSlot() && it->slot() == slot) {
      // EmitPropIC/WJH_PROPIC expect a PropertyKey (jsid) raw bits, NOT a Value.
      *keyBits = it->key().asRawBits();
      return true;
    }
  }
  return false;
}
// A representative shared shape for a shape guard (GuardShape or, after the IC
// warms polymorphic, GuardShapeList). For a list all shapes share the property's
// slot (Warp only folds same-slot shapes), so shapes()[0] derives the same key.
static js::Shape* WJGuardRepShape(jit::MDefinition* g) {
  if (g->isGuardShape()) return g->toGuardShape()->shape();
  if (g->isGuardShapeList()) {
    const auto& shapes = g->toGuardShapeList()->shapeList()->shapes();
    for (size_t i = 0; i < 4; i++)
      if (shapes[i]) return shapes[i];
  }
  return nullptr;
}
static bool WJIsShapeGuard(jit::MDefinition* g) {
  return g && (g->isGuardShape() || g->isGuardShapeList());
}
// A shape guard (GuardShape/GuardShapeList) is mega-convertible iff every SHAPE-
// DEPENDENT use is a fixed/dynamic-slot READ or derivable STORE. Then the guard
// becomes a no-op passthrough (no deopt -- crucially, GuardShapeList no longer
// deopts on a shape outside its captured set, which was deltablue 414's storm)
// and reads become self-guarding multi-shape EmitPropIC. Shape-AGNOSTIC uses
// (calls/compares/returns) are fine unguarded.
// Shape-guard PropIC hybrid (GECKO_WJ_SHAPEHYBRID): convert a convertible
// GuardShape + fixed/dynamic-slot read into a self-guarding baked PropIC (bare
// shape-compare fast path + ways/helper fallback, NO deopt) -- independent of the
// global forceMega (which also reroutes stores/GSLTO and regresses richards). The
// EmitPropIC baked fast path is ~the cost of a plain guarded read, so always-match
// receivers don't slow down, while a stale/dictionary/poly shape that today
// deopts-forever-to-PBL (gbemu's GuardShape storm) stays in JIT. See
// [[gbemu-deopt-storm]].
static bool WJShapeHybrid() {
  static int v = WJGETENV("GECKO_WJ_NOSHAPEHYBRID") ? 0 : 1;  // default-ON (validated)
  if (!v) return false;
  // Per-fn bisection (GECKO_WJ_NOHYBLINE=<lineno>): disable the hybrid in ONE
  // fn only. Deterministic across the guard-passthrough site and its dependent
  // load/store conversion sites (they re-derive the same decision), unlike a
  // global conversion counter which would desynchronize them.
  static int noLine =
      WJGETENV("GECKO_WJ_NOHYBLINE") ? atoi(WJGETENV("GECKO_WJ_NOHYBLINE")) : -1;
  if (noLine >= 0 && gWJCurScriptLine == uint32_t(noLine)) return false;
  return true;
}
static bool WJValueMightBeGCThing(jit::MIRType t);  // fwd (defined near EmitForcePostBarrier)
static bool EmitForcePostBarrier(Encoder& e, WJBackend& be, jit::MDefinition* objDef,
                                 jit::MDefinition* valDef = nullptr);  // fwd (UnsafeSetReservedSlot inline)
static bool WJHasPostBarrierNode(jit::MDefinition* obj);  // fwd (defined near EmitForcePostBarrier)
// A slot store is HYBRID-convertible (passthrough guard + store-IC, no deopt) iff
// its stored value can never be a GC pointer: a non-GC value (Int32/Double/Boolean
// /etc.) needs NO post-write barrier, so the store-IC's plain I64Store is correct
// and provably sidesteps the store-IC's historical GC barrier/staleness crash
// (which only bites GC-pointer stores). forceMega (allowStores) takes any store.
static bool WJStoreHybridOK(jit::MDefinition* storeVal, bool allowStores) {
  if (allowStores) return true;  // forceMega
  // Hybrid store-IC: convert NON-GC-value store-feeding guards to the self-guarding
  // store IC (no deopt, stores INLINE -- not via the WJH_SETPROP C++ helper). The
  // earlier "ERR=Gameboy" was the WJCollectRoots id-vs-execution-order staleness
  // ([[wjcollectroots-listorder-fix]]), now FIXED -- validated correct across the
  // suite (raytrace 3/3 clean), net-positive (splay +5%, crypto/richards/deltablue
  // up). DEFAULT-ON; GECKO_WJ_NOHYBSTORE disables. Non-GC values need no post-write
  // barrier, so this stays clear of the barrier-staleness class.
  static int hybStore = WJGETENV("GECKO_WJ_NOHYBSTORE") ? 0 : 1;
  // HYBSTOREGC: also convert GC-value stores. EmitPropStoreIC emits correct
  // pre/post-write barriers for GC values; the receiver staleness that
  // historically crashed this path was the WJCollectRoots id-order bug
  // ([[wjcollectroots-listorder-fix]]), now FIXED. Gated for validation.
  static int hybStoreGC = WJGETENV("GECKO_WJ_HYBSTOREGC") ? 1 : 0;
  if (!hybStore) return false;
  if (!hybStoreGC && WJValueMightBeGCThing(storeVal->type())) return false;
  // Bisection: GECKO_WJ_HYBSTORELINE=<n> scopes conversion to the function whose
  // script starts at line n (0/unset = all functions).
  static int onlyLine = WJGETENV("GECKO_WJ_HYBSTORELINE")
                            ? atoi(WJGETENV("GECKO_WJ_HYBSTORELINE")) : 0;
  if (onlyLine && gWJCurScriptLine != uint32_t(onlyLine)) return false;
  static int minLine = WJGETENV("GECKO_WJ_HYBSTOREMIN")
                           ? atoi(WJGETENV("GECKO_WJ_HYBSTOREMIN")) : 0;
  static int maxLine = WJGETENV("GECKO_WJ_HYBSTOREMAX")
                           ? atoi(WJGETENV("GECKO_WJ_HYBSTOREMAX")) : 0;
  if (minLine && int(gWJCurScriptLine) < minLine) return false;
  if (maxLine && int(gWJCurScriptLine) > maxLine) return false;
  if (WJGETENV("GECKO_WJ_HYBSTORELOG"))
    fprintf(stderr, "[wj-hybstore] convert store in fn@line=%u valType=%s\n",
            gWJCurScriptLine, StringFromMIRType(storeVal->type()));
  return true;
}
// allowStores=false (the shape-hybrid default) makes this READ-ONLY: a guard that
// feeds ANY slot/element store is NOT convertible, because passthrough'ing the
// guard while the store still writes at the baked (stale-shape) offset would
// corrupt the heap (the store mega/IC path is forceMega-gated, not hybrid). Only
// forceMega (which DOES convert stores) passes allowStores=true.
// A missing-own-property GET on a shape-guarded receiver is folded by Warp into an
// MConstantProto (the receiver's prototype baked as a CONSTANT) + a proto-chain
// GuardShape + a constant result -- proving the property is absent both on the
// receiver (via ITS shape) and up its proto chain. MConstantProto keeps its receiver
// as a BARE POINTER (receiverObject_), NOT an operand (see MIR.h), so this fold's
// dependence on the receiver's shape guard is INVISIBLE to a def-use walk over the
// guard. If we passthrough (remove) that guard, a polymorphic receiver that DOES have
// its own copy of the property reaches the folded (absent) result with NO deopt:
// destructuring-with-defaults `var {a=1,b=2,c=3}=o` warmed on `{a:10}` returned all
// DEFAULTS for `{b:20}` (b was folded to the default 2, ignoring the real own b=20).
// Keep the guard whenever any MConstantProto in a dominated block was folded against it.
static bool WJGuardBacksConstantProto(jit::MDefinition* gg) {
  jit::MBasicBlock* gb = gg->block();
  if (!gb) return false;
  // MConstantProto keeps its receiver as a BARE pointer that is frequently DISCARDED
  // by the time we lower (getReceiverObject() then returns null -- MIR.h: "we basically
  // just become an MConstant for the object's proto"), so identity-matching the receiver
  // is unreliable. Instead link SEMANTICALLY: the missing-prop fold baked the RECEIVER'S
  // prototype as its constant; if the guard we're considering guards a shape whose proto
  // is that same object, this fold assumes THIS receiver's shape and the guard is
  // load-bearing for it. (Over-match on a shared proto only keeps an extra guard = a
  // cheap shape-compare + deopt-only-if-polymorphic; never a correctness risk.)
  js::Shape* shape = WJGuardRepShape(gg);
  if (!shape) return false;
  js::TaggedProto tp = shape->proto();
  JSObject* guardProto = tp.isObject() ? tp.toObject() : nullptr;
  static int dbgc = -1;
  if (dbgc < 0) dbgc = WJGETENV("GECKO_WJ_PROTODBG") ? 1 : 0;
  bool dbg = dbgc != 0;
  jit::MIRGraph& graph = gb->graph();
  for (jit::MBasicBlockIterator b = graph.begin(); b != graph.end(); b++) {
    jit::MBasicBlock* blk = *b;
    if (blk != gb && !gb->dominates(blk)) continue;
    for (jit::MInstructionIterator it = blk->begin(); it != blk->end(); it++) {
      jit::MInstruction* d = *it;
      if (d->op() != jit::MDefinition::Opcode::ConstantProto) continue;
      jit::MDefinition* po = d->toConstantProto()->protoObject();
      JSObject* cpProto =
          po && po->isConstant() ? po->toConstant()->toObjectOrNull() : nullptr;
      if (dbg)
        fprintf(stderr, "[protodbg] gg=%p guardProto=%p cpProto=%p match=%d\n",
                (void*)gg, (void*)guardProto, (void*)cpProto,
                (guardProto && guardProto == cpProto));
      if (guardProto && guardProto == cpProto) return true;
    }
  }
  if (dbg) fprintf(stderr, "[protodbg] gg=%p NO matching ConstantProto\n", (void*)gg);
  return false;
}

// Does the guarded object feed a CHAINED downstream shape guard? I.e. a value
// loaded (LoadFixedSlot[AndUnbox]) from the guarded object, or the guarded
// object itself, is the input of another GuardShape/GuardShapeList/
// GuardMultipleShapes. A chained guard structure is where Warp folds
// cross-shape assumptions the use-walk can't see (babylon's tokenizer:
// GuardShape(this) -> load this.state -> GuardShape(state) -> ...). Dropping a
// speculative monomorphic guard that anchors such a chain misparses. gbemu's
// flat memory-array guards are NOT chained, so they still convert. Bounded to
// one load-hop (the observed pattern); polymorphic/stormProven guards are
// exempt (their conversion is the whole point / already earned).
static bool WJGuardAnchorsChain(jit::MDefinition* gg) {
  using Op = jit::MDefinition::Opcode;
  // NOTE (task #73): gating this on WJGuardFieldMutated(d) recovered gbemu +
  // deltablue to baseline BUT babylon FAILED again -- so babylon's culprit
  // chain has a CONVERTIBLE (read-only) inner guard, not a field-mutated one.
  // The exact distinguisher between babylon's unsound chain and gbemu's sound
  // one is still open; this coarse form (any chained guard) is the correctness-
  // complete lever (gbemu -15% cost), kept opt-in.
  // A chained inner guard is only babylon-unsound if it feeds NAMED-PROPERTY
  // access (Slots/LoadFixedSlot -> the this.state.x pattern). If it feeds only
  // ELEMENT access (ArrayBufferViewElements/Elements -- gbemu's
  // parentObj.memory[i]), converting the OUTER guard is sound (the by-name IC
  // reads .memory for any parentObj shape; element access is self-contained),
  // so we must NOT keep it -- keeping it reinstated gbemu's 750K deopt storm.
  auto isNamedPropGuard = [](jit::MDefinition* g) {
    using O = jit::MDefinition::Opcode;
    if (!(g->isGuardShape() || g->isGuardShapeList() ||
          g->isGuardMultipleShapes()))
      return false;
    for (jit::MUseIterator u = g->usesBegin(); u != g->usesEnd(); u++) {
      if (!u->consumer()->isDefinition()) continue;
      O o = u->consumer()->toDefinition()->op();
      if (o == O::Slots || o == O::LoadFixedSlot ||
          o == O::LoadFixedSlotAndUnbox || o == O::StoreFixedSlot)
        return true;
    }
    return false;
  };
  auto isGuard = [&](jit::MDefinition* d) { return isNamedPropGuard(d); };
  static int chainDbg = WJGETENV("GECKO_WJ_CHAINDBG") ? 1 : 0;
  auto innerMutated = [&](jit::MDefinition* ig) -> bool {
    for (jit::MUseIterator u = ig->usesBegin(); u != ig->usesEnd(); u++) {
      if (!u->consumer()->isDefinition()) continue;
      jit::MDefinition* d = u->consumer()->toDefinition();
      if (d->op() == Op::StoreFixedSlot) return true;
      if (d->op() == Op::Slots)
        for (jit::MUseIterator s = d->usesBegin(); s != d->usesEnd(); s++)
          if (s->consumer()->isDefinition() &&
              s->consumer()->toDefinition()->op() == Op::StoreDynamicSlot)
            return true;
    }
    return false;
  };
  auto hit = [&](jit::MDefinition* inner) -> bool {
    if (chainDbg)
      fprintf(stderr, "[wj-chain] fn@%u gg#%u -> inner %s#%u mutated=%d\n",
              gWJCurScriptLine, gg->id(), WJOpName(inner->op()), inner->id(),
              int(innerMutated(inner)));
    return true;
  };
  for (jit::MUseIterator u = gg->usesBegin(); u != gg->usesEnd(); u++) {
    if (!u->consumer()->isDefinition()) continue;
    jit::MDefinition* d = u->consumer()->toDefinition();
    if (isGuard(d)) return hit(d);  // guarded object directly re-guarded
    if (d->op() == Op::LoadFixedSlot || d->op() == Op::LoadFixedSlotAndUnbox ||
        d->op() == Op::Slots) {
      for (jit::MUseIterator u2 = d->usesBegin(); u2 != d->usesEnd(); u2++) {
        if (!u2->consumer()->isDefinition()) continue;
        jit::MDefinition* d2 = u2->consumer()->toDefinition();
        if (isGuard(d2)) return hit(d2);
        // one more hop: Slots -> LoadDynamicSlot[AndUnbox] -> guard
        if (d2->op() == Op::LoadDynamicSlot ||
            d2->op() == Op::LoadDynamicSlotAndUnbox) {
          for (jit::MUseIterator u3 = d2->usesBegin(); u3 != d2->usesEnd();
               u3++) {
            if (u3->consumer()->isDefinition() &&
                isGuard(u3->consumer()->toDefinition()))
              return hit(u3->consumer()->toDefinition());
          }
        }
      }
    }
  }
  return false;
}

static bool WJMegaConvertibleGuard(jit::MDefinition* gg, bool allowStores = true,
                                   bool stormProven = false) {
  using Op = jit::MDefinition::Opcode;
  js::Shape* shape = WJGuardRepShape(gg);
  if (!shape || !shape->isShared()) return false;
  // MONOMORPHIC-guard tradeoff (task #72). Speculatively converting a plain
  // GuardShape (single shape) to a passthrough is KNOWN-UNSOUND in general:
  // it drops a guard that shape-dependent downstream code (folded proto/callee/
  // offset assumptions Warp proved under the guard, INVISIBLE to this use-walk)
  // may rely on -- babylon's tokenizer misparses this way, and it is NOT the
  // converted read (PROPNOFILL still fails) but the guard REMOVAL itself. BUT
  // it is load-bearing for octane (gbemu -17%, deltablue -40% when gated), so
  // it stays DEFAULT-ON. GECKO_WJ_MONOHYBSTORM opts into the SOUND mode:
  // convert a monomorphic guard only after a proven deopt-storm (stormProven =
  // be.forceMega) -- fixes babylon at the octane cost. Polymorphic
  // GuardShapeList always converts (its whole point is the >4-shape storm). The
  // real fix is a surgical guard-keep (cf. WJGuardBacksConstantProto #42) that
  // keeps the speculative conversion but re-adds the guard when it backs an
  // invisible fold.
  if (gg->isGuardShape() && !stormProven) {
    static int monoStorm = WJGETENV("GECKO_WJ_MONOHYBSTORM") ? 1 : 0;
    if (monoStorm) return false;
    // Structural soundness rule (task #73), DEFAULT-ON: don't speculatively
    // pass through a monomorphic guard that ANCHORS a downstream guard chain.
    // Speculative mono-guard conversion is PROVEN UNSOUND (babylon's tokenizer
    // misparses -- the dropped guard lets a wrong-shaped object reach a folded
    // shape-dependent op; PROPNOFILL still fails => control-flow fold, not the
    // read value), and the chained-guard shape is where that fold lives. This
    // rule keeps such guards while still converting FLAT mono guards (gbemu's
    // memory arrays -- empirically sound). Cost: gbemu -15% (already JIT<PBL
    // 850 vs 959, so defending a losing bench); deltablue fully preserved;
    // babylon fixed. ~-1.5% octane geomean for a real correctness/soundness
    // win. GECKO_WJ_NOCHAINKEEP reverts to the old speculative always-convert.
    static int noChainKeep = WJGETENV("GECKO_WJ_NOCHAINKEEP") ? 1 : 0;
    if (!noChainKeep && WJGuardAnchorsChain(gg)) return false;
  }
  uint32_t nfixed = shape->asShared().numFixedSlots();
  // A guard with NO definition-uses (only resume-point uses) is a PURE SHAPE
  // ASSERTION: Warp emitted it to prove the receiver's shape so that a SIBLING
  // baked assumption -- a Constant prototype + GuardSpecificFunction method
  // resolution -- is sound, even though that sibling chain doesn't take the guard
  // as an operand. Converting such a guard to a no-op passthrough silently removes
  // the assertion, so a wrong-shaped receiver (e.g. pdfjs Lexer_getObj reading a
  // DecodeStream when the baked proto is base Stream's) reaches the baked (wrong)
  // method with NO deopt -> getObj lexed the raw compressed stream ("Unknown command
  // xÚ.."). Keep the guard (real shape check + deopt) when it has no def-use.
  bool hasDefUse = false;
  for (jit::MUseIterator u = gg->usesBegin(); u != gg->usesEnd(); u++) {
    jit::MNode* c = u->consumer();
    if (!c->isDefinition()) continue;  // resume points etc. are shape-agnostic
    hasDefUse = true;
    jit::MDefinition* d = c->toDefinition();
    Op op = d->op();
    uint64_t k;
    if (op == Op::LoadFixedSlot) {
      if (!WJDerivePropKey(shape, uint32_t(d->toLoadFixedSlot()->slot()), &k))
        return false;
    } else if (op == Op::LoadFixedSlotAndUnbox) {
      if (!WJDerivePropKey(shape, uint32_t(d->toLoadFixedSlotAndUnbox()->slot()), &k))
        return false;
    } else if (op == Op::StoreFixedSlot) {
      // store converts to a baked-key store IC; require a derivable key + that the
      // object operand is THIS guard (operand 0), not the stored value.
      auto* s = d->toStoreFixedSlot();
      if (s->object() != gg) return false;  // gg is the stored VALUE -> bail
      if (!WJStoreHybridOK(s->value(), allowStores)) return false;  // hybrid: non-GC only
      if (!WJDerivePropKey(shape, uint32_t(s->slot()), &k)) return false;
    } else if (op == Op::Slots) {
      // dynamic slots: every use of the Slots must be a derivable dynamic READ.
      for (jit::MUseIterator su = d->usesBegin(); su != d->usesEnd(); su++) {
        if (!su->consumer()->isDefinition()) continue;
        jit::MDefinition* sd = su->consumer()->toDefinition();
        Op so = sd->op();
        uint32_t dynSlot;
        if (so == Op::LoadDynamicSlot)
          dynSlot = uint32_t(sd->toLoadDynamicSlot()->slot());
        else if (so == Op::LoadDynamicSlotAndUnbox)
          dynSlot = uint32_t(sd->toLoadDynamicSlotAndUnbox()->slot());
        else if (so == Op::StoreDynamicSlot) {
          // dynamic-slot store: converts to a self-guarding store IC (re-checks
          // shape, looks up the per-shape slot offset, helper-falls-back on miss --
          // EmitPropStoreIC handles dynamic slots via the tagged-offset low bit).
          // This is gbemu's dominant deopt source: GuardShape -> Slots ->
          // StoreDynamicSlot (obj.field = v on out-of-line slots, e.g.
          // computeAudioChannels' channel counters). forceMega: MEGADYNSTORE-gated
          // (any value); hybrid: non-GC value only (no barrier needed). d (the Slots
          // node) must be the store's `slots` operand, not the stored value.
          auto* st = sd->toStoreDynamicSlot();
          bool ok = allowStores ? (WJGETENV("GECKO_WJ_MEGADYNSTORE") != nullptr)
                                : WJStoreHybridOK(st->value(), /*allowStores=*/false);
          if (!ok) return false;
          if (st->slots() != d) return false;
          dynSlot = uint32_t(st->slot());
        } else
          return false;  // other use -> keep guard
        if (!WJDerivePropKey(shape, nfixed + dynSlot, &k)) return false;
      }
    } else if (op == Op::Elements && WJGETENV("GECKO_WJ_MEGAELEM")) {
      // Elements loads obj->elements_ (FIXED offset, shape-agnostic). Safe to
      // passthrough the guard IFF every element access through it is a READ
      // (LoadElement/AndUnbox) -- a store could need the shape (grow/transition).
      // gbemu's memory arrays are monomorphic; the guard deopts only because the
      // array's shape mutated (length grew) -- 357K+ deopt storm. Element reads on
      // the same dense array stay valid.
      for (jit::MUseIterator eu = d->usesBegin(); eu != d->usesEnd(); eu++) {
        if (!eu->consumer()->isDefinition()) continue;
        Op eo = eu->consumer()->toDefinition()->op();
        // Element reads AND stores are shape-agnostic for a dense array (the
        // elements_ pointer is at a fixed offset; store-or-grow is handled by the
        // store op / its helper). Passthrough'ing the guard removes gbemu's element-
        // store shape-guard deopt storm. GECKO_WJ_MEGAESTORE also allows stores.
        bool readOk = eo == Op::LoadElement || eo == Op::LoadElementAndUnbox ||
                      eo == Op::ArrayLength || eo == Op::InitializedLength ||
                      eo == Op::BoundsCheck;
        bool storeOk = allowStores && WJGETENV("GECKO_WJ_MEGAESTORE") &&
                       (eo == Op::StoreElement || eo == Op::StoreElementHole ||
                        eo == Op::StoreElementHole || eo == Op::SetInitializedLength);
        if (!readOk && !storeOk)
          return false;  // non-element / other -> keep the guard
      }
    } else if (op == Op::Call || op == Op::ApplyArgs || op == Op::ApplyArray) {
      // A Call using the guarded object is a GETTER/METHOD call whose CALLEE is
      // BAKED from the guarded shape (o.v getter / o.m() method resolved via the
      // shape's proto). Removing the guard lets a polymorphic receiver call the
      // WRONG baked getter/method with NO deopt: `class Sub extends Base` with an
      // overridden `get v()` -- warmed on Base, Sub returned Base's getter (10 vs
      // 20). Keep the guard (real shape check + deopt to PBL for the right callee).
      // (This is the Call-use analogue of the pdfjs baked-getChar assertion below.)
      return false;
    } else if (op == Op::KeepAliveObject || op == Op::PostWriteBarrier ||
               op == Op::Compare || op == Op::Box || op == Op::IsObject ||
               op == Op::TypeOf || op == Op::TypeOfIs ||
               op == Op::IsCallable || op == Op::IsConstructor) {
      // Provably shape-agnostic uses of the guarded object: liveness markers,
      // barriers (container identity), identity compares, boxing, type tests.
    } else {
      // UNKNOWN use: assume SHAPE-DEPENDENT and keep the guard. The old
      // behavior (unknown -> "shape-agnostic, safe") was a heap-corruption
      // landmine: ArrayBufferViewElements/ArrayBufferViewLength fell through
      // it, so a typed-array receiver's guard was passthrough'd and a
      // Float32Array ran code compiled for Float64Array -- 8-byte scaled
      // stores over a 4-byte-element buffer (jit-test
      // ion/setelem-float32-typedarray-ic, GC-sweep crash). Explicitly
      // shape-DEPENDENT ops that used to be listed (Elements w/o MEGAELEM,
      // StoreDynamicSlot direct, *FromOffset, GuardShape chains,
      // AddAndStoreSlot/AllocateAndStoreSlot -- the raytrace corruption) now
      // land here too. GECKO_WJ_MEGAUNKNOWNOK restores the old default for
      // A/B.
      static int unknownOk = WJGETENV("GECKO_WJ_MEGAUNKNOWNOK") ? 1 : 0;
      if (WJGETENV("GECKO_WJ_MEGAREJDBG"))
        fprintf(stderr, "[wj-megarej] unknown use op=%s of guard (line=%u)\n",
                WJOpName(op), gWJCurScriptLine);
      if (!unknownOk) return false;
    }
  }
  // Pure shape assertion (no def-use): keeping the guard (real shape check + deopt)
  // FIXES pdfjs Lexer_getObj (a wrongly-passthrough'd assertion let a DecodeStream
  // reach a baked base-Stream getChar). BUT it REGRESSES acorn: keeping the guard
  // introduces a DEOPT, and acorn's deopt-RESUME at these sites corrupts -> mis-parse
  // (SyntaxError). A no-def-use guard has no uses to inspect, so pdfjs's load-bearing
  // assertion can't be distinguished from acorn's harmful one -- it's a pdfjs-vs-acorn
  // tradeoff rooted in the deep deopt-resume-correctness bug (task #15). DEFAULT-OFF
  // until that resume bug is fixed; OPT-IN via GECKO_WJ_KEEPASSERTGUARD (with it on,
  // STRCMP2 is correct on pdfjs but acorn breaks).
  // DEFAULT-ON (2026-06-29): now that the valve-result deopt-resume fix landed
  // (WasmJitRuntime.cpp valveReturn), keeping the no-def-use pure-assertion guard
  // no longer breaks acorn (its deopt-resume was the valve-rerun double-exec bug,
  // now fixed). Keeping the guard fixes pdfjs getObj (wrong base-Stream getChar on a
  // DecodeStream). Revert: GECKO_WJ_NOKEEPASSERTGUARD.
  if (!hasDefUse && !WJGETENV("GECKO_WJ_NOKEEPASSERTGUARD")) return false;
  // Even with legit convertible def-uses (e.g. a present-property LoadFixedSlot), a
  // guard can ALSO back an invisible MConstantProto missing-property fold. Keep it.
  static int noProtoFold = -1;
  if (noProtoFold < 0) noProtoFold = WJGETENV("GECKO_WJ_NOPROTOFOLDGUARD") ? 1 : 0;
  if (!noProtoFold && WJGuardBacksConstantProto(gg)) return false;
  if (WJGETENV("GECKO_WJ_MEGAUSES")) {
    fprintf(stderr, "[wj-megauses] guard %s CONV uses:", WJOpName(gg->op()));
    for (jit::MUseIterator u = gg->usesBegin(); u != gg->usesEnd(); u++) {
      if (!u->consumer()->isDefinition()) { fprintf(stderr, " <rp>"); continue; }
      fprintf(stderr, " %s", WJOpName(u->consumer()->toDefinition()->op()));
    }
    fprintf(stderr, "\n");
  }
  return true;  // no disallowed shape-dependent use -> safe to passthrough
}
// The object operand guarded by a shape guard (GuardShape/GuardShapeList).
static jit::MDefinition* WJGuardObject(jit::MDefinition* g) {
  if (g->isGuardShape()) return g->toGuardShape()->object();
  if (g->isGuardShapeList()) return g->toGuardShapeList()->object();
  return nullptr;
}
// If `slots` is MSlots(obj) where obj is a mega-convertible shape guard, return
// that guard (dynamic reads through it convert to EmitPropIC); else null.
static jit::MDefinition* WJMegaSlotsGuard(jit::MDefinition* slots,
                                          bool allowStores = true,
                                          bool stormProven = false) {
  if (!slots || !slots->isSlots()) return nullptr;
  jit::MDefinition* obj = slots->getOperand(0);
  if (!WJIsShapeGuard(obj) ||
      !WJMegaConvertibleGuard(obj, allowStores, stormProven))
    return nullptr;
  return obj;
}
// GuardShapeListToOffset (polymorphic read where the property sits at DIFFERENT
// slots per shape -> shape->offset table + deopt on an unlisted shape, deltablue
// 414's storm). For mega: derive the property key (all shapes hold the same
// property; shapes()[0]+offsets()[0] suffices -- offsets are dynamic-slot BYTE
// offsets, so abs slot = numFixedSlots + offset/8). Returns the guarded object +
// key iff convertible.
static bool WJMegaGSLTO(jit::MDefinition* g, jit::MDefinition** objOut,
                        uint64_t* keyOut) {
  if (!g->isGuardShapeListToOffset()) return false;
  auto* gg = g->toGuardShapeListToOffset();
  const auto& shapes = gg->shapeList()->shapes();
  const auto& offsets = gg->shapeList()->offsets();
  js::Shape* s0 = nullptr;
  uint32_t off0 = 0;
  for (size_t i = 0; i < 4; i++) {
    if (shapes[i]) { s0 = shapes[i]; off0 = offsets[i]; break; }
  }
  if (!s0 || !s0->isShared()) return false;
  if (off0 % sizeof(JS::Value) != 0) return false;
  uint32_t slot = s0->asShared().numFixedSlots() + off0 / uint32_t(sizeof(JS::Value));
  if (!WJDerivePropKey(s0, slot, keyOut)) return false;
  *objOut = gg->object();
  return true;
}

// A generic-IC helper (WJH_BINARYARITH/UNARYARITH/...) leaves a BOXED i64 result
// on the stack. If Warp TYPED the node (Int32/Double/Boolean), coerce the box to
// that wasm type so it matches the node's local (else host rejects the module).
static bool EmitHelperResultAsType(Encoder& e, WJBackend& be, jit::MIRType type,
                                   bool guardTag = false) {
  if (type == jit::MIRType::Value) return true;  // box already matches i64 local
  if (type == jit::MIRType::Double || type == jit::MIRType::Float32) {
    // A numeric box may be int32-tagged (e.g. 2+3=5) or double-tagged. Convert
    // both to f64 WITHOUT deopting (arith of a Double-typed node is always
    // numeric): int32 -> f64.convert_i32_s; else -> f64.reinterpret. (Raw
    // reinterpret alone mangles an int32 box -> garbage f64 -> OOB/trap.)
    uint32_t s = be.unboxScratch;
    if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(s)) return false;
    // tag == INT32 ?
    if (!GetLocal(e, s) || !e.writeOp(Op::I64Const) || !e.writeVarS64(32) ||
        !e.writeOp(Op::I64ShrU) || !e.writeOp(Op::I32WrapI64) ||
        !e.writeOp(Op::I32Const) ||
        !e.writeVarS32(int32_t(TagWord(JSVAL_TYPE_INT32))) || !e.writeOp(Op::I32Eq))
      return false;
    if (!e.writeOp(Op::If) || !e.writeFixedU8(0x7C)) return false;  // result f64
    if (!GetLocal(e, s) || !e.writeOp(Op::I32WrapI64) ||
        !e.writeOp(Op::F64ConvertI32S))
      return false;
    if (!e.writeOp(Op::Else)) return false;
    if (!GetLocal(e, s) || !e.writeOp(Op::F64ReinterpretI64)) return false;
    return e.writeOp(Op::End);
  }
  // Int32 / Boolean: the payload is the low 32 bits of the NUNBOX box. When
  // `guardTag` is set (the mega/by-name PROPERTY-load path -- EmitPropIC for a
  // speculated-Int32/Boolean property whose value can turn out non-int32, e.g.
  // undefined/string), tag-GUARD and DEOPT on mismatch first: blindly wrapping a
  // non-int32 box read its low 32 bits as a bogus int32 with NO deopt (a loop
  // `t += (s.v+i)|0` with undefined s.v computed `t += i`; box2d/regexp wrong
  // answers). Sound now that the deopt-resume IC-entry is aligned. The guard is
  // ONLY for the mega prop-load caller -- arith/CharCodeAt results are genuinely
  // typed and must NOT guard (guarding them deopt-storms richards/navier to PBL).
  if (guardTag && (type == jit::MIRType::Int32 || type == jit::MIRType::Boolean) &&
      true) {
    uint32_t s = be.unboxScratch;
    uint32_t wantTag = (type == jit::MIRType::Boolean)
                           ? TagWord(JSVAL_TYPE_BOOLEAN)
                           : TagWord(JSVAL_TYPE_INT32);
    if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(s)) return false;
    if (!GetLocal(e, s) || !e.writeOp(Op::I64Const) || !e.writeVarS64(32) ||
        !e.writeOp(Op::I64ShrU) || !e.writeOp(Op::I32WrapI64) ||
        !e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(wantTag)) ||
        !e.writeOp(Op::I32Ne))
      return false;
    if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
    if (!EmitDeopt(e, be)) return false;
    if (!e.writeOp(Op::End)) return false;
    return GetLocal(e, s) && e.writeOp(Op::I32WrapI64);
  }
  return e.writeOp(Op::I32WrapI64);
}

// Attribute a potential bail to `ins`: record its op name AND its precise source
// location (inlinee script + line + bytecode offset from its trackedSite), so the
// LOGBAIL audit pinpoints the exact failing op rather than just the compiled
// function's defining line. trackedSite()->pc() can point at a JumpTarget for some
// ops, so the offset is approximate, but the line is the op's real origin.
static void WJSetBailOp(MInstruction* ins) {
  js::wasm::gWJBailReason = WJOpName(ins->op());
  js::wasm::gWJBailOpLine = 0;
  js::wasm::gWJBailOpOff = 0;
  js::wasm::gWJBailOpFile = nullptr;
  const jit::BytecodeSite* site = ins->trackedSite();
  if (site && site->script() && site->pc()) {
    JSScript* s = site->script();
    js::wasm::gWJBailOpFile = s->filename();
    js::wasm::gWJBailOpOff = uint32_t(site->pc() - s->code());
    js::wasm::gWJBailOpLine = js::PCToLineNumber(s, site->pc());
  }
}

// GECKO_WJ_OOBLOAD: a BoundsCheck whose EVERY use is a Value-typed plain
// LoadElement can SKIP its deopt -- the load self-bounds (returns undefined on
// OOB) instead, eliminating gbemu's 357K BoundsCheck deopt-storm-into-PBL. Only
// the common (min==0,max==0) form, Value type only (a typed LoadElementAndUnbox
// can't take undefined). Sound for plain dense arrays with Array.prototype (no
// indexed proto props) -- OOB read == undefined; in-bounds holes still handled by
// the load's hole-check. CORRECTNESS-GATED + tested; off by default.
static bool WJBoundsCheckOOBSafe(jit::MDefinition* bcDef) {
  static int oob = WJGETENV("GECKO_WJ_OOBLOAD") ? 1 : 0;
  if (!oob || !bcDef->isBoundsCheck()) return false;
  jit::MBoundsCheck* bc = bcDef->toBoundsCheck();
  if (bc->minimum() != 0 || bc->maximum() != 0) return false;
  if (!bcDef->hasUses()) return false;
  if (WJGETENV("GECKO_WJ_BCUSEDBG")) {
    for (MUseIterator u = bcDef->usesBegin(); u != bcDef->usesEnd(); u++) {
      MNode* c = u->consumer();
      if (c->isDefinition())
        fprintf(stderr, "[wj-bcuse] %s type=%d\n",
                WJOpName(c->toDefinition()->op()), int(c->toDefinition()->type()));
      else
        fprintf(stderr, "[wj-bcuse] <resumepoint>\n");
    }
  }
  for (MUseIterator u = bcDef->usesBegin(); u != bcDef->usesEnd(); u++) {
    MNode* c = u->consumer();
    if (!c->isDefinition()) return false;
    jit::MDefinition* d = c->toDefinition();
    // Value LoadElement (OOB -> undefined) or Int32 LoadElementAndUnbox
    // (OOB -> 0, i.e. ToInt32(undefined); correct for dense int arrays like an
    // emulator's RAM where unread memory reads as 0).
    bool okValueLoad = d->op() == jit::MDefinition::Opcode::LoadElement &&
                       d->type() == jit::MIRType::Value;
    bool okInt32Load =
        d->op() == jit::MDefinition::Opcode::LoadElementAndUnbox &&
        d->type() == jit::MIRType::Int32;
    if (!okValueLoad && !okInt32Load) return false;
  }
  return true;
}

// Compute a node's value, leaving exactly one wasm value on the stack. (For
// effectful/void nodes EmitEffect is used instead.) Returns false => unsupported.
// Inline charCodeAt: load the idx-th char of a LINEAR string (the operand is post
// LinearizeForCharAccess, so inline-or-non-inline, latin1-or-twobyte, never a rope)
// without the WJH_CHARCODEAT helper hop (~426K/iter in ubo). OOB -> -1 if negOnOob else
// deopt. Result: int32 char code on the wasm stack.
static bool EmitInlineCharCodeAt(Encoder& e, WJBackend& be, MInstruction* ins,
                                 bool negOnOob) {
  MDefinition* strD = ins->getOperand(0);
  MDefinition* idxD = ins->getOperand(1);
  if (strD->type() != MIRType::String) return false;
  int32_t idxL = be.local(idxD);
  if (idxD->type() != MIRType::Int32 || idxL < 0) return false;
  const uint32_t flagsOff = uint32_t(JSString::offsetOfFlags());
  const uint32_t lenOff = uint32_t(JSString::offsetOfLength());
  const uint32_t inlineOff =
      uint32_t(offsetof(JS::shadow::String, inlineStorageLatin1));
  const uint32_t ninlineOff =
      uint32_t(offsetof(JS::shadow::String, nonInlineCharsLatin1));
  const int32_t L1 = int32_t(js::StringFlags::LATIN1_CHARS_BIT);
  const int32_t INL = int32_t(js::StringFlags::INLINE_CHARS_BIT);
  const int32_t LIN = int32_t(js::StringFlags::LINEAR_BIT);
  auto loadFlags = [&]() {
    return GetOp(e, be, strD) && e.writeOp(Op::I32Load) && e.writeVarU32(2) &&
           e.writeVarU32(flagsOff);
  };
  // Rope guard: a JSRope stores `left` (a JSString*) where a linear string
  // stores nonInlineChars, so reading rope chars inline yields garbage. Ropes
  // need C++ flattening -> deopt to PBL (rare; ropes are cold here).
  if (!loadFlags() || !e.writeOp(Op::I32Const) || !e.writeVarS32(LIN) ||
      !e.writeOp(Op::I32And) || !e.writeOp(Op::I32Eqz) || !e.writeOp(Op::If) ||
      !e.writeFixedU8(0x40))
    return false;
  {
    bool saved = be.inDispatchBody;
    be.inDispatchBody = false;
    bool ok = EmitDeopt(e, be);
    be.inDispatchBody = saved;
    if (!ok) return false;
  }
  if (!e.writeOp(Op::End)) return false;
  // (uint)idx >= length ?  (OOB)
  if (!GetLocal(e, uint32_t(idxL)) || !GetOp(e, be, strD) ||
      !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(lenOff) ||
      !e.writeOp(Op::I32GeU))
    return false;
  if (!e.writeOp(Op::If) || !e.writeFixedU8(0x7F)) return false;  // -> i32
  if (negOnOob) {
    if (!e.writeOp(Op::I32Const) || !e.writeVarS32(-1)) return false;
  } else {
    bool saved = be.inDispatchBody;
    be.inDispatchBody = false;  // inline deopt (nested ifs would break the OOL br depth)
    bool ok = EmitDeopt(e, be);
    be.inDispatchBody = saved;
    if (!ok) return false;
  }
  if (!e.writeOp(Op::Else)) return false;
  // charsPtr = (flags & INLINE) ? str+inlineOff : load(str, nonInlineCharsOff)
  if (!loadFlags() || !e.writeOp(Op::I32Const) || !e.writeVarS32(INL) ||
      !e.writeOp(Op::I32And) || !e.writeOp(Op::If) || !e.writeFixedU8(0x7F))
    return false;
  if (!GetOp(e, be, strD) || !e.writeOp(Op::I32Const) ||
      !e.writeVarS32(int32_t(inlineOff)) || !e.writeOp(Op::I32Add))
    return false;
  if (!e.writeOp(Op::Else)) return false;
  if (!GetOp(e, be, strD) || !e.writeOp(Op::I32Load) || !e.writeVarU32(2) ||
      !e.writeVarU32(ninlineOff))
    return false;
  if (!e.writeOp(Op::End)) return false;
  if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(be.callScratch)) return false;
  // char = (flags & LATIN1) ? u8[charsPtr+idx] : u16[charsPtr+idx*2]
  if (!loadFlags() || !e.writeOp(Op::I32Const) || !e.writeVarS32(L1) ||
      !e.writeOp(Op::I32And) || !e.writeOp(Op::If) || !e.writeFixedU8(0x7F))
    return false;
  if (!GetLocal(e, be.callScratch) || !GetLocal(e, uint32_t(idxL)) ||
      !e.writeOp(Op::I32Add) || !e.writeOp(Op::I32Load8U) || !e.writeVarU32(0) ||
      !e.writeVarU32(0))
    return false;
  if (!e.writeOp(Op::Else)) return false;
  if (!GetLocal(e, be.callScratch) || !GetLocal(e, uint32_t(idxL)) ||
      !e.writeOp(Op::I32Const) || !e.writeVarS32(2) || !e.writeOp(Op::I32Mul) ||
      !e.writeOp(Op::I32Add) || !e.writeOp(Op::I32Load16U) || !e.writeVarU32(1) ||
      !e.writeVarU32(0))
    return false;
  if (!e.writeOp(Op::End)) return false;  // end latin1/twobyte
  return e.writeOp(Op::End);              // end OOB else
}

// LoadSlotByIteratorIndex[Indexed]: inline the CacheIR/Ion pointer arithmetic that
// reads an own-data-property from `obj` using the (index,kind) PropertyIndex the
// for-in NativeIterator recorded -- NO megamorphic-cache/VM hop. Mirrors
// CodeGenerator::visitLoadSlotByIteratorIndexCommon + extract*IndexAndKindFromIterator.
// explicitIndex==nullptr -> Current variant (uses the iterator cursor, disp -=
// sizeof(PropertyIndex) since the cursor already advanced); else Indexed (uses the
// operand as inIndex, no disp adjust). Leaves the boxed Value (i64) on the stack.
// On wasm32 a PrivateValue is the raw ptr in the low 32 bits (plain i32 load), and
// sizeof(IteratorProperty)==sizeof(PropertyIndex)==4 so no count<<1. Deterministic
// (reads live iterator/object memory), no GC hazard.
// Extract the PropertyIndex (index,kind) from the iterator into be.iterIdx/iterKind.
// explicitIndex==nullptr -> Current variant (cursor, disp -= sizeof(PropertyIndex));
// else Indexed (operand). Shared by the load and store paths.
static bool EmitIterExtract(Encoder& e, WJBackend& be, MDefinition* iterator,
                            MDefinition* explicitIndex) {
  const uint32_t offIterSlot =
      uint32_t(js::PropertyIteratorObject::offsetOfIteratorSlot());
  const uint32_t offPropCount = uint32_t(js::NativeIterator::offsetOfPropertyCount());
  const uint32_t offPropCursor =
      uint32_t(js::NativeIterator::offsetOfPropertyCursor());
  const uint32_t offFirstProp =
      uint32_t(js::NativeIterator::offsetOfFirstProperty());
  static_assert(sizeof(js::IteratorProperty) == sizeof(js::PropertyIndex),
                "wasm32 iter-slot extract assumes equal sizes (no count<<1)");
  // iterIdx (A) = NativeIterator* = i32-load[ iterator + offIterSlot ]
  if (!GetOp(e, be, iterator)) return false;
  if (!e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(offIterSlot))
    return false;
  if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(be.iterIdx)) return false;
  // iterKind (B) = propertyCount + (cursor | inIndex)
  if (!GetLocal(e, be.iterIdx)) return false;
  if (!e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(offPropCount))
    return false;
  if (explicitIndex) {
    if (!GetOp(e, be, explicitIndex)) return false;
  } else {
    if (!GetLocal(e, be.iterIdx)) return false;
    if (!e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(offPropCursor))
      return false;
  }
  if (!e.writeOp(Op::I32Add)) return false;
  if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(be.iterKind)) return false;
  // A = PropertyIndex bits = i32-load[ A + (B<<2) + disp ]
  uint32_t disp = explicitIndex ? offFirstProp
                                : offFirstProp - uint32_t(sizeof(js::PropertyIndex));
  if (!GetLocal(e, be.iterIdx) || !GetLocal(e, be.iterKind)) return false;
  if (!e.writeOp(Op::I32Const) || !e.writeVarS32(2) || !e.writeOp(Op::I32Shl))
    return false;
  if (!e.writeOp(Op::I32Add)) return false;
  if (!e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(disp))
    return false;
  if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(be.iterIdx)) return false;
  // B = kind = A >> KindShift
  if (!GetLocal(e, be.iterIdx)) return false;
  if (!e.writeOp(Op::I32Const) ||
      !e.writeVarS32(int32_t(js::PropertyIndex::KindShift)) ||
      !e.writeOp(Op::I32ShrU))
    return false;
  if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(be.iterKind)) return false;
  // A = index = A & IndexMask
  if (!GetLocal(e, be.iterIdx)) return false;
  if (!e.writeOp(Op::I32Const) ||
      !e.writeVarS32(int32_t(js::PropertyIndex::IndexMask)) || !e.writeOp(Op::I32And))
    return false;
  return e.writeOp(Op::LocalSet) && e.writeVarU32(be.iterIdx);
}

// addIdxShifted: (base already on stack) + (be.iterIdx<<3)  [sizeof(Value)==8]
static bool EmitAddIdxShifted(Encoder& e, WJBackend& be) {
  return GetLocal(e, be.iterIdx) && e.writeOp(Op::I32Const) && e.writeVarS32(3) &&
         e.writeOp(Op::I32Shl) && e.writeOp(Op::I32Add);
}

static bool EmitIterSlotLoad(Encoder& e, WJBackend& be, MDefinition* object,
                             MDefinition* iterator, MDefinition* explicitIndex) {
  if (!EmitIterExtract(e, be, iterator, explicitIndex)) return false;
  // switch on kind -> boxed Value (i64) on stack
  if (!GetLocal(e, be.iterKind) || !e.writeOp(Op::I32Eqz)) return false;  // ==DynamicSlot(0)
  if (!e.writeOp(Op::If) || !e.writeFixedU8(uint8_t(TypeCode::I64))) return false;
  {  // DynamicSlot: i64-load[ i32-load[obj+offSlots] + (A<<3) ]
    if (!GetOp(e, be, object)) return false;
    if (!e.writeOp(Op::I32Load) || !e.writeVarU32(2) ||
        !e.writeVarU32(uint32_t(js::NativeObject::offsetOfSlots())))
      return false;
    if (!EmitAddIdxShifted(e, be)) return false;
    if (!e.writeOp(Op::I64Load) || !e.writeVarU32(3) || !e.writeVarU32(0)) return false;
  }
  if (!e.writeOp(Op::Else)) return false;
  if (!GetLocal(e, be.iterKind) || !e.writeOp(Op::I32Const) || !e.writeVarS32(1) ||
      !e.writeOp(Op::I32Eq))
    return false;  // ==FixedSlot(1)
  if (!e.writeOp(Op::If) || !e.writeFixedU8(uint8_t(TypeCode::I64))) return false;
  {  // FixedSlot: i64-load[ obj + (A<<3) ] @ sizeof(NativeObject)
    if (!GetOp(e, be, object)) return false;
    if (!EmitAddIdxShifted(e, be)) return false;
    if (!e.writeOp(Op::I64Load) || !e.writeVarU32(3) ||
        !e.writeVarU32(uint32_t(sizeof(js::NativeObject))))
      return false;
  }
  if (!e.writeOp(Op::Else)) return false;
  {  // Element: i64-load[ i32-load[obj+offElements] + (A<<3) ]
    if (!GetOp(e, be, object)) return false;
    if (!e.writeOp(Op::I32Load) || !e.writeVarU32(2) ||
        !e.writeVarU32(uint32_t(js::NativeObject::offsetOfElements())))
      return false;
    if (!EmitAddIdxShifted(e, be)) return false;
    if (!e.writeOp(Op::I64Load) || !e.writeVarU32(3) || !e.writeVarU32(0)) return false;
  }
  if (!e.writeOp(Op::End)) return false;  // end fixed/element
  return e.writeOp(Op::End);              // end dynamic/rest
}

// StoreSlotByIteratorIndex[Indexed]: `o[Object.keys(o)[i]] = v` fast-path write.
// Compute the effective slot/element address into be.iterAddr (3-way kind switch,
// fixed-slot folds sizeof(NativeObject) into the address), then gated pre-write
// barrier on the OLD value + inline i64 store + gated post-write barrier. Mirrors
// CodeGenerator::visitStoreSlotByIteratorIndexCommon.
static bool EmitIterSlotStore(Encoder& e, WJBackend& be, MDefinition* object,
                              MDefinition* iterator, MDefinition* explicitIndex,
                              MDefinition* value) {
  if (!EmitIterExtract(e, be, iterator, explicitIndex)) return false;
  // compute store address -> be.iterAddr (void-typed switch)
  if (!GetLocal(e, be.iterKind) || !e.writeOp(Op::I32Eqz)) return false;  // ==DynamicSlot
  if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
  {  // DynamicSlot: i32-load[obj+offSlots] + (A<<3)
    if (!GetOp(e, be, object)) return false;
    if (!e.writeOp(Op::I32Load) || !e.writeVarU32(2) ||
        !e.writeVarU32(uint32_t(js::NativeObject::offsetOfSlots())))
      return false;
    if (!EmitAddIdxShifted(e, be)) return false;
    if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(be.iterAddr)) return false;
  }
  if (!e.writeOp(Op::Else)) return false;
  if (!GetLocal(e, be.iterKind) || !e.writeOp(Op::I32Const) || !e.writeVarS32(1) ||
      !e.writeOp(Op::I32Eq))
    return false;  // ==FixedSlot
  if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
  {  // FixedSlot: obj + (A<<3) + sizeof(NativeObject)
    if (!GetOp(e, be, object)) return false;
    if (!EmitAddIdxShifted(e, be)) return false;
    if (!e.writeOp(Op::I32Const) ||
        !e.writeVarS32(int32_t(sizeof(js::NativeObject))) || !e.writeOp(Op::I32Add))
      return false;
    if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(be.iterAddr)) return false;
  }
  if (!e.writeOp(Op::Else)) return false;
  {  // Element: i32-load[obj+offElements] + (A<<3)
    if (!GetOp(e, be, object)) return false;
    if (!e.writeOp(Op::I32Load) || !e.writeVarU32(2) ||
        !e.writeVarU32(uint32_t(js::NativeObject::offsetOfElements())))
      return false;
    if (!EmitAddIdxShifted(e, be)) return false;
    if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(be.iterAddr)) return false;
  }
  if (!e.writeOp(Op::End)) return false;  // end fixed/element
  if (!e.writeOp(Op::End)) return false;  // end dynamic/rest
  // gated pre-write barrier on the OLD value at [iterAddr]
  if (!EmitGuardedValuePreBarrier(e, be, [&]() {
        return GetLocal(e, be.iterAddr) && e.writeOp(Op::I64Load) &&
               e.writeVarU32(3) && e.writeVarU32(0);
      }))
    return false;
  // store: [iterAddr] = boxed value
  if (!GetLocal(e, be.iterAddr)) return false;
  if (!EmitCanonStoreValue(e, be, value)) return false;
  if (!e.writeOp(Op::I64Store) || !e.writeVarU32(3) || !e.writeVarU32(0)) return false;
  // gated post-write barrier (obj may now reference a nursery value)
  if (WJValueMightBeGCThing(value->type()) &&
      !EmitForcePostBarrier(e, be, object, value))
    return false;
  return true;
}

static bool EmitValue(Encoder& e, WJBackend& be, MInstruction* ins) {
  WJSetBailOp(ins);  // attribute an internal bail to this op (+ precise op line)
  switch (ins->op()) {
    case MDefinition::Opcode::Parameter: {
      MParameter* p = ins->toParameter();
      // Register ABI: this = wasm param kWJThisParam; arg i = param kWJArg0Param+i.
      if (p->index() == MParameter::THIS_SLOT) {
        return e.writeOp(Op::LocalGet) && e.writeVarU32(kWJThisParam);
      }
      if (uint32_t(p->index()) >= kWJMaxArgs) {
        js::wasm::gWJBailReason = "too-many-args";
        return false;  // too many args: bail
      }
      return e.writeOp(Op::LocalGet) &&
             e.writeVarU32(kWJArg0Param + uint32_t(p->index()));
    }
    case MDefinition::Opcode::Constant: {
      MConstant* c = ins->toConstant();
      switch (c->type()) {
        case MIRType::Int32:
          return e.writeOp(Op::I32Const) && e.writeVarS32(c->toInt32());
        case MIRType::IntPtr:  // wasm32: pointer-sized int constant == i32
          return e.writeOp(Op::I32Const) &&
                 e.writeVarS32(int32_t(c->toIntPtr()));
        case MIRType::Boolean:
          return e.writeOp(Op::I32Const) && e.writeVarS32(c->toBoolean() ? 1 : 0);
        case MIRType::Double:
          return e.writeOp(Op::F64Const) && e.writeFixedF64(c->toDouble());
        case MIRType::Float32:
          // Float32 held as f64 in our repr (WJValType); widening f32->f64 is exact.
          return e.writeOp(Op::F64Const) &&
                 e.writeFixedF64(double(c->toFloat32()));
        case MIRType::Null:
          return e.writeOp(Op::I64Const) &&
                 e.writeVarS64(int64_t(JS::NullValue().asRawBits()));
        case MIRType::Undefined:
          return e.writeOp(Op::I64Const) &&
                 e.writeVarS64(int64_t(JS::UndefinedValue().asRawBits()));
        case MIRType::Object: {
          // Bake the object via the traced GC-constant pool, then load the live
          // pointer (low 32 bits of the boxed Value) at runtime.
          JSObject* obj = c->toObjectOrNull();
          if (!obj) return false;
          uintptr_t slot = WJInternConstant(JS::ObjectValue(*obj).asRawBits());
          if (!slot) return false;
          if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(slot)))
            return false;
          if (!e.writeOp(Op::I64Load) || !e.writeVarU32(3) || !e.writeVarU32(0))
            return false;
          return e.writeOp(Op::I32WrapI64);  // -> object pointer
        }
        case MIRType::String:
        case MIRType::Symbol:
        case MIRType::BigInt: {
          // GC-thing constant: bake the boxed Value via the traced pool, load
          // the live pointer (low 32 bits) at runtime.
          uintptr_t slot = WJInternConstant(c->toJSValue().asRawBits());
          if (!slot) return false;
          if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(slot)))
            return false;
          if (!e.writeOp(Op::I64Load) || !e.writeVarU32(3) || !e.writeVarU32(0))
            return false;
          return e.writeOp(Op::I32WrapI64);  // -> GC pointer
        }
        default:
          return false;
      }
    }
    case MDefinition::Opcode::Elements: {
      int32_t objLocal = be.local(ins->toElements()->object());
      if (objLocal < 0) return false;
      uint32_t off = uint32_t(js::NativeObject::offsetOfElements());
      if (!GetLocal(e, uint32_t(objLocal))) return false;
      return e.writeOp(Op::I32Load) && e.writeVarU32(2) && e.writeVarU32(off);
    }
    case MDefinition::Opcode::Box: {
      MDefinition* in = ins->toBox()->input();
      if (in->type() == MIRType::Value) return GetOp(e, be, in);  // already boxed
      // Magic values have no payload -- bake the boxed magic constant directly.
      JSWhyMagic why;
      bool isMagic = true;
      switch (in->type()) {
        case MIRType::MagicOptimizedOut: why = JS_OPTIMIZED_OUT; break;
        case MIRType::MagicHole: why = JS_ELEMENTS_HOLE; break;
        case MIRType::MagicIsConstructing: why = JS_IS_CONSTRUCTING; break;
        case MIRType::MagicUninitializedLexical:
          why = JS_UNINITIALIZED_LEXICAL;
          break;
        default: isMagic = false; break;
      }
      if (isMagic) {
        return e.writeOp(Op::I64Const) &&
               e.writeVarS64(int64_t(JS::MagicValue(why).asRawBits()));
      }
      // Payload-less values: bake the boxed constant.
      if (in->type() == MIRType::Undefined) {
        return e.writeOp(Op::I64Const) &&
               e.writeVarS64(int64_t(JS::UndefinedValue().asRawBits()));
      }
      if (in->type() == MIRType::Null) {
        return e.writeOp(Op::I64Const) &&
               e.writeVarS64(int64_t(JS::NullValue().asRawBits()));
      }
      if (!GetOp(e, be, in)) return false;
      return EmitBoxFromStack(e, in->type());
    }
    case MDefinition::Opcode::BoxNonStrictThis: {
      // `this` in a sloppy function. For an object receiver this is identity;
      // primitives go through ToObject (rare) -> deopt. Guard object, extract ptr.
      int32_t inLocal = be.local(ins->getOperand(0));
      if (inLocal < 0) return false;
      return EmitUnboxLocal(e, be, uint32_t(inLocal), MIRType::Object, /*fallible=*/true);
    }
    case MDefinition::Opcode::Unbox: {
      MUnbox* u = ins->toUnbox();
      int32_t inLocal = be.local(u->input());
      if (inLocal < 0) {
        if (WJGETENV("GECKO_WJ_UNBOXDBG"))
          fprintf(stderr, "[wj-unbox-nolocal] input op=%s ty=%s\n",
                  WJOpName(u->input()->op()), StringFromMIRType(u->input()->type()));
        return false;
      }
      return EmitUnboxLocal(e, be, uint32_t(inLocal), u->type(),
                            u->mode() == MUnbox::Fallible);
    }
    case MDefinition::Opcode::GuardShape: {
      // Passthrough the object; deopt if obj->shape() != the recorded Shape*.
      // The expected shape is loaded from a GC-traced pool slot (NOT baked as a
      // raw immediate): a compacting GC relocates the shape and updates the slot,
      // so the guard keeps comparing against the live pointer. Baking it raw made
      // a moved shape ALWAYS fail -> permanent deopt storm (crypto am3: ~475k).
      MGuardShape* g = ins->toGuardShape();
      // PROBE (GECKO_WJ_NOSHAPEGUARD): passthrough ALL shape guards (no deopt).
      // UNSAFE for polymorphic receivers, but CORRECT for monomorphic ones (same
      // object, shape merely mutated) -- measures the ceiling if gbemu's shape-guard
      // deopt storm were eliminated.
      if (WJGETENV("GECKO_WJ_NOSHAPEGUARD")) return GetOp(e, be, g->object());
      // Megamorphic recompile: if every use is a derivable fixed-slot read, the
      // reads become self-guarding EmitPropIC (handled in LoadFixedSlot), so this
      // guard is a no-op passthrough -- NO deopt on a polymorphic receiver.
      if (be.forceMega) {
        bool conv = WJMegaConvertibleGuard(g, /*allowStores=*/true, /*stormProven=*/true);
        if (WJGETENV("GECKO_WJ_MEGADBG")) {
          js::Shape* sh = g->shape();
          const char* badop = "none";
          if (conv) badop = "CONVERTED";
          else if (!sh || !sh->isShared()) badop = "not-shared-shape";
          else {
            for (jit::MUseIterator u = g->usesBegin(); u != g->usesEnd(); u++) {
              if (!u->consumer()->isDefinition()) continue;
              badop = WJOpName(u->consumer()->toDefinition()->op());
              jit::MDefinition::Opcode o = u->consumer()->toDefinition()->op();
              if (o == jit::MDefinition::Opcode::LoadFixedSlot ||
                  o == jit::MDefinition::Opcode::LoadFixedSlotAndUnbox)
                continue;
              break;
            }
          }
          JSScript* ms = be.info ? be.info->script() : nullptr;
          fprintf(stderr, "[wj-mega] %s:%u GuardShape conv=%d firstbad=%s\n",
                  ms && ms->filename() ? ms->filename() : "?",
                  ms ? unsigned(ms->lineno()) : 0, conv, badop);
        }
        if (conv) return GetOp(e, be, g->object());
        if (WJGETENV("GECKO_WJ_MEGAFAIL")) {
          JSScript* ms = be.info ? be.info->script() : nullptr;
          fprintf(stderr, "[wj-megafail] %s:%u GuardShape NOTCONV alluses:",
                  ms && ms->filename() ? ms->filename() : "?",
                  ms ? unsigned(ms->lineno()) : 0);
          for (jit::MUseIterator u = g->usesBegin(); u != g->usesEnd(); u++) {
            if (!u->consumer()->isDefinition()) { fprintf(stderr, " <rp>"); continue; }
            fprintf(stderr, " %s", WJOpName(u->consumer()->toDefinition()->op()));
          }
          fprintf(stderr, "\n");
        }
      }
      // Shape-guard PropIC hybrid: a convertible guard becomes a no-op passthrough;
      // its dependent fixed/dynamic-slot reads emit a self-guarding baked PropIC
      // (handled in LoadFixedSlot etc.). No deopt on a stale/poly/dictionary shape.
      if (WJShapeHybrid() && WJMegaConvertibleGuard(g, /*allowStores=*/false, /*stormProven=*/be.forceMega)) {
        if (WJGETENV("GECKO_WJ_HYBCONVLOG")) {
          JSScript* ms = be.info ? be.info->script() : nullptr;
          int nshapes = g->isGuardShapeList()
              ? int([&]{ int n=0; const auto& sl=g->toGuardShapeList()->shapeList()->shapes();
                         for(size_t i=0;i<4;i++) if(sl[i]) n++; return n; }())
              : (g->isGuardShape() ? 1 : -1);
          fprintf(stderr, "[wj-hybconv] passthrough guard fn@%u kind=%s nshapes=%d uses:",
                  ms ? unsigned(ms->lineno()) : 0, WJOpName(g->op()), nshapes);
          for (jit::MUseIterator u = g->usesBegin(); u != g->usesEnd(); u++) {
            if (!u->consumer()->isDefinition()) { fprintf(stderr, " <rp>"); continue; }
            jit::MDefinition* cd = u->consumer()->toDefinition();
            fprintf(stderr, " %s", WJOpName(cd->op()));
            if (cd->isSlots()) {
              for (jit::MUseIterator su = cd->usesBegin(); su != cd->usesEnd(); su++)
                if (su->consumer()->isDefinition())
                  fprintf(stderr, "/%s",
                          WJOpName(su->consumer()->toDefinition()->op()));
            }
          }
          fprintf(stderr, "\n");
        }
        return GetOp(e, be, g->object());
      }
      if (WJGETENV("GECKO_WJ_HYBFAIL")) {  // why a guard ISN'T convertible (still deopts)
        js::Shape* sh = WJGuardRepShape(g);
        JSScript* ms = be.info ? be.info->script() : nullptr;
        fprintf(stderr, "[wj-hybfail] %s:%u shared=%d uses:",
                ms && ms->filename() ? ms->filename() : "?",
                ms ? unsigned(ms->lineno()) : 0, sh ? int(sh->isShared()) : -1);
        for (jit::MUseIterator u = g->usesBegin(); u != g->usesEnd(); u++) {
          if (!u->consumer()->isDefinition()) { fprintf(stderr, " <rp>"); continue; }
          fprintf(stderr, " %s", WJOpName(u->consumer()->toDefinition()->op()));
        }
        fprintf(stderr, "\n");
      }
      static int forceDeoptLine =
          WJGETENV("GECKO_WJ_FORCEDEOPTLINE") ? atoi(WJGETENV("GECKO_WJ_FORCEDEOPTLINE")) : 0;
      if (forceDeoptLine && be.info && be.info->script() &&
          uint32_t(be.info->script()->lineno()) == uint32_t(forceDeoptLine)) {
        if (!e.writeOp(Op::I32Const) || !e.writeVarS32(1)) return false;
        if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
        if (!EmitDeopt(e, be)) return false;
        if (!e.writeOp(Op::End)) return false;
        return GetOp(e, be, g->object());
      }
      int32_t objLocal = be.local(g->object());
      if (objLocal < 0) return false;
      uint32_t shapeOff = uint32_t(offsetof(JS::shadow::Object, shape));
      // BAKESHAPE: compare obj->shape_ to the shape pointer baked as an i32
      // immediate (no GC-pool load). A compacting GC that relocates the shape
      // makes the guard fail -> deopt -> the recompile-on-storm valve rebakes the
      // new shape. Saves one memory load per guard on the hot path. Default off.
      static int bakeShape = WJGETENV("GECKO_WJ_BAKESHAPE") ? 1 : 0;
      if (bakeShape) {
        if (!GetLocal(e, uint32_t(objLocal))) return false;  // obj
        if (!e.writeOp(Op::I32Load) || !e.writeVarU32(2) ||
            !e.writeVarU32(shapeOff))
          return false;  // obj->shape_
        if (!e.writeOp(Op::I32Const) ||
            !e.writeVarS32(int32_t(uint32_t(uintptr_t(g->shape())))))
          return false;  // baked expected shape
        if (!e.writeOp(Op::I32Ne)) return false;
        if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
        if (!EmitDeopt(e, be)) return false;
        if (!e.writeOp(Op::End)) return false;
        return GetLocal(e, uint32_t(objLocal));
      }
      uintptr_t slot = WJInternShape(uintptr_t(g->shape()));
      if (!slot) return false;
      // Read the object via GetOp (rematerialize from the rooted source) so the
      // shape check + passthrough see the GC-CURRENT object, not a stale cached
      // pointer from before a GC-ing call (earley sc_reverse).
      if (!GetOp(e, be, g->object())) return false;  // obj (current)
      if (!e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(shapeOff))
        return false;  // obj->shape_
      if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(slot)) ||
          !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0))
        return false;  // current (relocated) expected shape from the pool
      if (!e.writeOp(Op::I32Ne)) return false;
      if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
      if (WJGETENV("GECKO_WJ_GSRT")) {  // store-only: actual/expected shape at deopt
        uintptr_t aa = uintptr_t(static_cast<void*>(&gWJGSActual));
        uintptr_t ea = uintptr_t(static_cast<void*>(&gWJGSExpect));
        uintptr_t ha = uintptr_t(static_cast<void*>(&gWJGSHits));
        if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(aa)) ||
            !GetOp(e, be, g->object()) || !e.writeOp(Op::I32Load) ||
            !e.writeVarU32(2) || !e.writeVarU32(shapeOff) ||
            !e.writeOp(Op::I32Store) || !e.writeVarU32(2) || !e.writeVarU32(0))
          return false;
        if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(ea)) ||
            !e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(slot)) ||
            !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0) ||
            !e.writeOp(Op::I32Store) || !e.writeVarU32(2) || !e.writeVarU32(0))
          return false;
        if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(ha)) ||
            !e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(ha)) ||
            !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0) ||
            !e.writeOp(Op::I32Const) || !e.writeVarS32(1) || !e.writeOp(Op::I32Add) ||
            !e.writeOp(Op::I32Store) || !e.writeVarU32(2) || !e.writeVarU32(0))
          return false;
      }
      if (!EmitDeopt(e, be)) return false;
      if (!e.writeOp(Op::End)) return false;
      return GetOp(e, be, g->object());  // result = the (current) object
    }
    case MDefinition::Opcode::GuardNoDenseElements: {
      // Passthrough the object; deopt if obj->elements->initializedLength != 0
      // (Ion visitGuardNoDenseElements). Used by SetProp/AddSlot IC stubs to ensure
      // a named-property add can't collide with dense indexed elements. Bailing it
      // dropped crypto-sha1's hot core_sha1 entirely to PBL (1.17x). The empty-
      // elements header is shared+valid (initLen 0) so the load is always safe.
      MGuardNoDenseElements* g = ins->toGuardNoDenseElements();
      if (!GetOp(e, be, g->object())) return false;  // obj
      if (!e.writeOp(Op::I32Load) || !e.writeVarU32(2) ||
          !e.writeVarU32(uint32_t(js::NativeObject::offsetOfElements())))
        return false;  // obj->elements_ (offsetOfElements positive -> memarg OK)
      // elements->initializedLength: offsetOfInitializedLength() is NEGATIVE (it is
      // element-data-relative: offsetof - sizeof(ObjectElements)). A wasm memarg
      // offset is UNSIGNED, so baking -12 there gives 0xFFFFFFF4 -> addr wraps far
      // past memory -> OOB trap (this was crypto-sha1's elusive core_sha1 OOB, and a
      // latent landmine for any executed GuardNoDenseElements). Add it explicitly.
      if (!e.writeOp(Op::I32Const) ||
          !e.writeVarS32(js::ObjectElements::offsetOfInitializedLength()) ||
          !e.writeOp(Op::I32Add) || !e.writeOp(Op::I32Load) || !e.writeVarU32(2) ||
          !e.writeVarU32(0))
        return false;  // elements->initializedLength (nonzero -> dense present)
      if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
      if (!EmitDeopt(e, be)) return false;
      if (!e.writeOp(Op::End)) return false;
      return GetOp(e, be, g->object());  // passthrough the (current) object
    }
    case MDefinition::Opcode::GuardIndexIsNotDenseElement: {
      // Passthrough the index; deopt iff obj[index] IS a present dense element
      // (index < initLength && element is not a hole). Mirrors
      // visitGuardIndexIsNotDenseElement. op0 = object, op1 = index (Int32).
      if (WJGETENV("GECKO_WJ_NO_GINDE")) return false;
      MDefinition* obj = ins->getOperand(0);
      MDefinition* idx = ins->getOperand(1);
      int32_t io = be.local(idx);
      if (io < 0) return false;
      uint32_t offElems = uint32_t(js::NativeObject::offsetOfElements());
      // index u< initLength ?
      if (!GetLocal(e, uint32_t(io))) return false;
      if (!GetOp(e, be, obj) || !e.writeOp(Op::I32Load) || !e.writeVarU32(2) ||
          !e.writeVarU32(offElems) || !e.writeOp(Op::I32Const) ||
          !e.writeVarS32(js::ObjectElements::offsetOfInitializedLength()) ||
          !e.writeOp(Op::I32Add) || !e.writeOp(Op::I32Load) || !e.writeVarU32(2) ||
          !e.writeVarU32(0))
        return false;  // initLength
      if (!e.writeOp(Op::I32LtU)) return false;
      if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
      // in-bounds: load elem[index] (i64 box); deopt unless it is a hole magic.
      if (!GetOp(e, be, obj) || !e.writeOp(Op::I32Load) || !e.writeVarU32(2) ||
          !e.writeVarU32(offElems))
        return false;  // elements ptr
      if (!GetLocal(e, uint32_t(io)) || !e.writeOp(Op::I32Const) ||
          !e.writeVarS32(8) || !e.writeOp(Op::I32Mul) || !e.writeOp(Op::I32Add))
        return false;  // &elem[index]
      if (!e.writeOp(Op::I64Load) || !e.writeVarU32(3) || !e.writeVarU32(0))
        return false;  // element value
      if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(be.unboxScratch)) return false;
      // isHole = (tag == MAGIC) && (payload == JS_ELEMENTS_HOLE)
      if (!GetLocal(e, be.unboxScratch) || !e.writeOp(Op::I64Const) ||
          !e.writeVarS64(32) || !e.writeOp(Op::I64ShrU) || !e.writeOp(Op::I32WrapI64) ||
          !e.writeOp(Op::I32Const) ||
          !e.writeVarS32(int32_t(TagWord(JSVAL_TYPE_MAGIC))) || !e.writeOp(Op::I32Eq))
        return false;
      if (!GetLocal(e, be.unboxScratch) || !e.writeOp(Op::I32WrapI64) ||
          !e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(JS_ELEMENTS_HOLE)) ||
          !e.writeOp(Op::I32Eq))
        return false;
      if (!e.writeOp(Op::I32And) || !e.writeOp(Op::I32Eqz)) return false;  // !isHole
      if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
      if (!EmitDeopt(e, be)) return false;
      if (!e.writeOp(Op::End)) return false;  // end !isHole
      if (!e.writeOp(Op::End)) return false;  // end in-bounds
      return GetLocal(e, uint32_t(io));       // passthrough index
    }
    case MDefinition::Opcode::GuardMultipleShapes: {
      // Polymorphic shape guard backed by a runtime ListObject of shapes. Deopt
      // unless obj->shape() is in the list (helper scans the dense elements).
      // Passthrough the object. Mirrors visitGuardMultipleShapes.
      if (WJGETENV("GECKO_WJ_NO_GMS")) return false;
      MGuardMultipleShapes* g = ins->toGuardMultipleShapes();
      int32_t l = be.local(g->object());
      if (l < 0) return false;
      if (!EmitStageScratch(e, be, g->object(), 0)) return false;
      if (!EmitStageScratch(e, be, g->shapeList(), 1)) return false;
      if (!EmitHelperCallResult(e, be, ins, WJH_INSHAPELIST, 0)) return false;
      if (!EmitHelperResultAsType(e, be, jit::MIRType::Boolean)) return false;
      if (!e.writeOp(Op::I32Eqz)) return false;  // not in list -> deopt
      if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
      if (!EmitDeopt(e, be)) return false;
      if (!e.writeOp(Op::End)) return false;
      return GetLocal(e, uint32_t(l));
    }
    case MDefinition::Opcode::GuardMultipleShapesToOffset: {
      // Polymorphic property access via a runtime ListObject of [shape,offset]
      // pairs -> the matched shape's slot byte offset (Int32), consumed by a
      // {Load,Store}DynamicSlotFromOffset. Helper scans the list (mirrors
      // MacroAssembler::branchTestObjShapeListSetOffset) and returns the offset,
      // or -1 -> deopt. Sibling of GuardShapeListToOffset (inline shapes).
      if (WJGETENV("GECKO_WJ_NO_GMSO")) return false;
      MGuardMultipleShapesToOffset* g = ins->toGuardMultipleShapesToOffset();
      if (be.local(g->object()) < 0) return false;
      if (!EmitStageScratch(e, be, g->object(), 0)) return false;
      if (!EmitStageScratch(e, be, g->shapeList(), 1)) return false;
      if (!EmitHelperCallResult(e, be, ins, WJH_INSHAPELISTOFFSET, 0)) return false;
      if (!EmitHelperResultAsType(e, be, jit::MIRType::Int32)) return false;  // [off]
      // off < 0 => obj->shape() not in the list => deopt. Stash off first so it
      // survives the compare, then re-push it as this op's Int32 result. Use the
      // i32 propShapeLocal as the temp (unboxScratch is an i64 local; not otherwise
      // live in this op).
      if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(be.propShapeLocal)) return false;
      if (!GetLocal(e, be.propShapeLocal) || !e.writeOp(Op::I32Const) ||
          !e.writeVarS32(0) || !e.writeOp(Op::I32LtS))
        return false;
      if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
      if (!EmitDeopt(e, be)) return false;
      if (!e.writeOp(Op::End)) return false;
      return GetLocal(e, be.propShapeLocal);
    }
    case MDefinition::Opcode::GuardShapeList: {
      // Polymorphic shape guard: deopt unless obj->shape() is one of the recorded
      // shapes (up to 4). Each shape goes through the GC-traced pool (relocation-
      // safe). Passthrough the object. Lets polymorphic property access (e.g.
      // splay `this.root_` Node-or-null, deltablue constraint subtypes) compile.
      MGuardShapeList* g = ins->toGuardShapeList();
      if (WJGETENV("GECKO_WJ_NOSHAPEGUARD")) return GetOp(e, be, g->object());  // probe
      // Megamorphic recompile: a GuardShapeList DEOPTS on any shape outside its
      // captured set -> storms when a new subtype appears (deltablue 414). If its
      // uses are convertible reads, passthrough (no deopt) and let the reads
      // self-guard via multi-shape EmitPropIC (helper fallback handles ANY shape).
      if (be.forceMega) {
        bool conv = WJMegaConvertibleGuard(ins, /*allowStores=*/true, /*stormProven=*/true);
        if (WJGETENV("GECKO_WJ_MEGADBG")) {
          js::Shape* sh = WJGuardRepShape(ins);
          const char* bad = conv ? "CONV" : (!sh || !sh->isShared()) ? "notshared" : "?";
          if (!conv && sh && sh->isShared()) {
            for (jit::MUseIterator u = g->usesBegin(); u != g->usesEnd(); u++) {
              if (!u->consumer()->isDefinition()) continue;
              jit::MDefinition* d = u->consumer()->toDefinition();
              jit::MDefinition::Opcode o = d->op();
              if (o == jit::MDefinition::Opcode::LoadFixedSlot ||
                  o == jit::MDefinition::Opcode::LoadFixedSlotAndUnbox) {
                uint64_t k;
                if (!WJDerivePropKey(sh, o == jit::MDefinition::Opcode::LoadFixedSlot
                    ? uint32_t(d->toLoadFixedSlot()->slot())
                    : uint32_t(d->toLoadFixedSlotAndUnbox()->slot()), &k)) {
                  bad = "nokey-read"; break;
                }
                continue;
              }
              bad = WJOpName(o); break;
            }
          }
          JSScript* ms = be.info ? be.info->script() : nullptr;
          fprintf(stderr, "[wj-mega] %s:%u GuardShapeList conv=%d bad=%s\n",
                  ms && ms->filename() ? ms->filename() : "?",
                  ms ? unsigned(ms->lineno()) : 0, conv, bad);
        }
        if (conv) return GetOp(e, be, g->object());
      }
      // Shape-hybrid (read-only, default-on): a GuardShapeList whose uses are all
      // convertible slot READS becomes a passthrough -- the reads self-guard via a
      // by-name EmitPropIC that handles ANY shape, so NO deopt on a receiver outside
      // the captured 4-shape set. Kills polymorphic-receiver GuardShapeList storms
      // (ubo's astFromTemplate: 175k deopts on >4 AST-template subtypes). Mirrors the
      // GuardShape hybrid above. (Store-feeding lists still deopt -> store-hybrid.)
      if (WJShapeHybrid() && WJMegaConvertibleGuard(ins, /*allowStores=*/false, /*stormProven=*/be.forceMega)) {
        return GetOp(e, be, g->object());
      }
      int32_t objLocal = be.local(g->object());
      if (objLocal < 0) return false;
      uint32_t shapeOff = uint32_t(offsetof(JS::shadow::Object, shape));
      const auto& shapes = g->shapeList()->shapes();
      bool any = false;
      for (size_t i = 0; i < 4; i++) {  // ShapeListSnapshot::NumShapes
        js::Shape* s = shapes[i];
        if (!s) continue;
        uintptr_t slot = WJInternShape(uintptr_t(s));
        if (!slot) return false;
        // push (obj->shape == shapes[i])
        if (!GetLocal(e, uint32_t(objLocal)) || !e.writeOp(Op::I32Load) ||
            !e.writeVarU32(2) || !e.writeVarU32(shapeOff) ||
            !e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(slot)) ||
            !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0) ||
            !e.writeOp(Op::I32Eq))
          return false;
        if (any && !e.writeOp(Op::I32Or)) return false;
        any = true;
      }
      if (!any) return false;  // empty list shouldn't happen; bail to be safe
      // deopt unless matched: !matched -> deopt
      if (!e.writeOp(Op::I32Eqz)) return false;
      if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
      if (!EmitDeopt(e, be)) return false;
      if (!e.writeOp(Op::End)) return false;
      return GetLocal(e, uint32_t(objLocal));
    }
    case MDefinition::Opcode::GuardShapeListToOffset: {
      // Polymorphic property access -> slot offset. deopt unless obj->shape() is
      // one of the recorded (shape, offset) pairs; result is the matched shape's
      // byte offset (Int32), consumed by LoadDynamicSlotFromOffset. Mirrors Ion's
      // visitGuardShapeListToOffset. Two passes: a guard at nesting==1 (OOL-deopt
      // safe), then a deopt-free select for the offset. Relocation-safe interned
      // shapes (GC-traced pool). (splay insert/remove `node.left/right`.)
      auto* g = ins->toGuardShapeListToOffset();
      // Megamorphic recompile: if every use is a LoadDynamicSlotFromOffset that
      // converts to EmitPropIC (which self-guards + helper-falls-back, ignoring
      // this offset), drop the DEOPT and emit a dummy offset. This kills the
      // GuardShapeListToOffset storm on an unlisted shape (deltablue 414).
      if (be.forceMega) {
        jit::MDefinition* o = nullptr;
        uint64_t k = 0;
        bool allConv = g->hasUses() && WJMegaGSLTO(ins, &o, &k) &&
                       o->type() == MIRType::Object;
        for (jit::MUseIterator u = g->usesBegin(); allConv && u != g->usesEnd();
             u++) {
          if (!u->consumer()->isDefinition()) continue;
          if (!u->consumer()->toDefinition()->isLoadDynamicSlotFromOffset())
            allConv = false;
        }
        if (allConv) return e.writeOp(Op::I32Const) && e.writeVarS32(0);
      }
      int32_t objLocal = be.local(g->object());
      if (objLocal < 0) return false;
      uint32_t shapeOff = uint32_t(offsetof(JS::shadow::Object, shape));
      const auto& shapes = g->shapeList()->shapes();
      const auto& offsets = g->shapeList()->offsets();
      uintptr_t shapeSlot[4];
      uint32_t offs[4];
      int count = 0;
      for (size_t i = 0; i < 4; i++) {  // ShapeListSnapshot::NumShapes
        js::Shape* s = shapes[i];
        if (!s) continue;
        uintptr_t slot = WJInternShape(uintptr_t(s));
        if (!slot) return false;
        shapeSlot[count] = slot;
        offs[count] = offsets[i];
        count++;
      }
      if (count == 0) return false;
      // obj->shape -> propShapeLocal (reused across guard + select).
      if (!GetLocal(e, uint32_t(objLocal)) || !e.writeOp(Op::I32Load) ||
          !e.writeVarU32(2) || !e.writeVarU32(shapeOff) ||
          !e.writeOp(Op::LocalSet) || !e.writeVarU32(be.propShapeLocal))
        return false;
      auto pushShapeEq = [&](int k) -> bool {
        return GetLocal(e, be.propShapeLocal) && e.writeOp(Op::I32Const) &&
               e.writeVarS32(int32_t(shapeSlot[k])) && e.writeOp(Op::I32Load) &&
               e.writeVarU32(2) && e.writeVarU32(0) && e.writeOp(Op::I32Eq);
      };
      // guard: deopt unless OR(propShape == shape[k]).
      for (int k = 0; k < count; k++) {
        if (!pushShapeEq(k)) return false;
        if (k && !e.writeOp(Op::I32Or)) return false;
      }
      if (!e.writeOp(Op::I32Eqz) || !e.writeOp(Op::If) || !e.writeFixedU8(0x40))
        return false;
      if (!EmitDeopt(e, be)) return false;
      if (!e.writeOp(Op::End)) return false;
      // offset select (deopt-free; guard guarantees a match -> last is the else).
      for (int k = 0; k + 1 < count; k++) {
        if (!pushShapeEq(k)) return false;
        if (!e.writeOp(Op::If) || !e.writeFixedU8(uint8_t(TypeCode::I32)))
          return false;
        if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(offs[k])))
          return false;
        if (!e.writeOp(Op::Else)) return false;
      }
      if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(offs[count - 1])))
        return false;
      for (int k = 0; k + 1 < count; k++) {
        if (!e.writeOp(Op::End)) return false;
      }
      return true;
    }
    case MDefinition::Opcode::HasShape: {
      // result = (obj->shape() == shape) as a Boolean. Non-deopting shape test
      // (vs GuardShape). Interned shape ptr (GC-traced pool). Mirrors Ion's
      // visitHasShape (loadObjShapeUnsafe + cmpPtrSet).
      MHasShape* h = ins->toHasShape();
      uintptr_t slot = WJInternShape(uintptr_t(h->shape()));
      if (!slot) return false;
      uint32_t shapeOff = uint32_t(offsetof(JS::shadow::Object, shape));
      if (!GetOp(e, be, h->object())) return false;             // [obj]
      if (!e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(shapeOff))
        return false;                                           // [obj->shape]
      if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(slot)))
        return false;                                           // [obj->shape, &internedShape]
      if (!e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0))
        return false;                                           // [obj->shape, internedShape]
      return e.writeOp(Op::I32Eq);                              // [bool]
    }
    case MDefinition::Opcode::OptimizeGetIteratorCache: {
      // for-of / spread fast-path check: does `value` have the default iterator?
      // js::OptimizeGetIterator (pure, no throw) via helper -> Boolean. Unblocks
      // for-of/spread-heavy real code. GECKO_WJ_NOOPTGETITER reverts.
      if (WJGETENV("GECKO_WJ_NOOPTGETITER")) return false;
      if (!EmitStageScratch(e, be, ins->toOptimizeGetIteratorCache()->value(), 0))
        return false;
      if (!EmitHelperCallResult(e, be, ins, WJH_OPTGETITER, 0)) return false;
      return EmitHelperResultAsType(e, be, jit::MIRType::Boolean);
    }
    case MDefinition::Opcode::IsPackedArray: {
      // obj is a dense hole-free ArrayObject? Pure bool read (js::IsPackedArray) used by
      // array fast paths (spread/Reflect/Array.from). Helper (cheap read, no GC/throw) --
      // strictly better than bailing the whole fn.
      if (!EmitStageScratch(e, be, ins->getOperand(0), 0)) return false;  // object
      if (!EmitHelperCallResult(e, be, ins, WJH_ISPACKEDARRAY, 0)) return false;
      return EmitHelperResultAsType(e, be, jit::MIRType::Boolean);
    }
    case MDefinition::Opcode::NewIterator: {
      // for-of / spread iterator object (`for (x of arr)`, `[...arr]`, `[...str]`).
      // 3 variants via m->type(); cold alloc -> VM call (js::NewArrayIterator etc.),
      // passing the type as the helper `site`. Result Object.
      uint32_t ty = uint32_t(ins->toNewIterator()->type());
      if (!EmitHelperCallResult(e, be, ins, WJH_NEWITERATOR, ty)) return false;
      return EmitHelperResultAsType(e, be, jit::MIRType::Object);
    }
    case MDefinition::Opcode::NewMapObject: {
      // `new Map()`: cold alloc (Map's internal hashtable can't be inline bump-alloc'd)
      // -> js::MapObject::create via helper. No operands (templateObject baked; nullptr
      // proto = default Map.prototype, matches Ion OOL).
      if (!EmitHelperCallResult(e, be, ins, WJH_NEWMAP, 0)) return false;
      return EmitHelperResultAsType(e, be, jit::MIRType::Object);
    }
    case MDefinition::Opcode::NewSetObject: {
      // `new Set()`: same pattern as NewMapObject.
      if (!EmitHelperCallResult(e, be, ins, WJH_NEWSET, 0)) return false;
      return EmitHelperResultAsType(e, be, jit::MIRType::Object);
    }
    case MDefinition::Opcode::HasOwnCache: {
      // `id in obj` / obj.hasOwnProperty(id): js::HasOwnProperty (boxed value+key ->
      // bool). May GC/throw (ToPropertyKey/ToObject) -> the helper propagates a throw.
      // value()=obj in scratch[0], idval()=key in scratch[1] (order matches
      // DoHasOwnFallback). Very common on real sites (feature detection) -> was the top
      // real-site root bail (reddit/stackoverflow).
      if (!EmitStageScratch(e, be, ins->toHasOwnCache()->value(), 0)) return false;
      if (!EmitStageScratch(e, be, ins->toHasOwnCache()->idval(), 1)) return false;
      if (!EmitHelperCallResult(e, be, ins, WJH_HASOWN, 0)) return false;
      return EmitHelperResultAsType(e, be, jit::MIRType::Boolean);
    }
    case MDefinition::Opcode::OptimizeSpreadCallCache: {
      // `f(...value)` spread-call fast-path: js::OptimizeSpreadCall -> the array to
      // spread (Value) or undefined. May GC/throw (arguments-spread path) -> the
      // helper propagates a throw. Unblocks spread-call-heavy real code (33 sites).
      if (!EmitStageScratch(e, be, ins->toOptimizeSpreadCallCache()->value(), 0))
        return false;
      if (!EmitHelperCallResult(e, be, ins, WJH_OPTSPREADCALL, 0)) return false;
      return EmitHelperResultAsType(e, be, jit::MIRType::Value);
    }
    case MDefinition::Opcode::Rest: {
      // Rest parameter `function f(a, ...rest)`: build the array from actuals
      // [numFormals..argc). Stage the kWJMaxArgs actuals (wasm params), numFormals
      // (const) and the argc snapshot into scratch, then WJH_REST ->
      // js::jit::InitRestParameter. The function-entry argc>kWJMaxArgs guard
      // (needsArgcGuard) ensures the body only runs with argc<=kWJMaxArgs, so the
      // 8-param staging captures every actual. Wider calls return flag-2 -> PBL.
      // 14 real-tool bail sites.
      if (ins->block()->info().script() != be.info->script()) return false;
      if (!be.usesArgc) return false;
      if (be.info && be.info->nargs() > kWJMaxArgs) return false;
      uint32_t numFormals = ins->toRest()->numFormals();
      if (numFormals > uint32_t(kWJMaxArgs)) return false;
      for (uint32_t i = 0; i < kWJMaxArgs; i++) {  // scratch[i] = actual arg i
        if (!GetLocal(e, be.scratchBase)) return false;
        if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(kWJArg0Param + i)) return false;
        if (!e.writeOp(Op::I64Store) || !e.writeVarU32(3) || !e.writeVarU32(i * 8))
          return false;
      }
      if (!GetLocal(e, be.scratchBase) ||                          // scratch[8]=numFormals
          !e.writeOp(Op::I64Const) || !e.writeVarS64(int64_t(numFormals)) ||
          !e.writeOp(Op::I64Store) || !e.writeVarU32(3) || !e.writeVarU32(8 * 8))
        return false;
      if (!GetLocal(e, be.scratchBase) ||                          // scratch[9]=argc (raw)
          !GetLocal(e, be.actualArgcLocal) || !e.writeOp(Op::I64ExtendI32U) ||
          !e.writeOp(Op::I64Store) || !e.writeVarU32(3) || !e.writeVarU32(9 * 8))
        return false;
      if (!EmitHelperCallResult(e, be, ins, WJH_REST, 0)) return false;
      return EmitHelperResultAsType(e, be, jit::MIRType::Object);
    }
    case MDefinition::Opcode::CallGetSparseElement: {
      // obj[index] slow path for sparse/holey arrays (js::GetSparseElementHelper).
      // Cold, but eliminating the bail keeps the containing fn in JIT.
      MCallGetSparseElement* c = ins->toCallGetSparseElement();
      if (!EmitStageScratch(e, be, c->object(), 0)) return false;
      if (!EmitStageScratch(e, be, c->index(), 1)) return false;
      if (!EmitHelperCallResult(e, be, ins, WJH_GETSPARSEELEM, 0)) return false;
      return EmitHelperResultAsType(e, be, jit::MIRType::Value);
    }
    case MDefinition::Opcode::LoadSlotByIteratorIndexIndexed: {
      // Indexed for-in body `obj[k]` read (explicit iter index operand). The
      // non-indexed variant is served by WJH_LOADSLOTBYITER below; this one had no
      // handler (16-site real-tool bail) -> inline the pointer arithmetic.
      MLoadSlotByIteratorIndexIndexed* l =
          ins->toLoadSlotByIteratorIndexIndexed();
      return EmitIterSlotLoad(e, be, l->object(), l->iterator(), l->index());
    }
    case MDefinition::Opcode::ArrayBufferByteLength: {
      // result = arraybuffer->byteLength (IntPtr == i32 on wasm32). Stored as a
      // PrivateValue in fixed slot BYTE_LENGTH_SLOT: on 64-bit NUNBOX its raw
      // asBits_ == the length integer, so load the i64 slot and wrap to i32.
      // Mirrors Ion's loadArrayBufferByteLengthIntPtr (loadPrivate).
      MArrayBufferByteLength* a = ins->toArrayBufferByteLength();
      uint32_t off = uint32_t(js::ArrayBufferObject::offsetOfByteLengthSlot());
      if (!GetOp(e, be, a->object())) return false;             // [obj]
      if (!e.writeOp(Op::I64Load) || !e.writeVarU32(3) || !e.writeVarU32(off))
        return false;                                           // [i64 byteLength]
      return e.writeOp(Op::I32WrapI64);                         // [i32 IntPtr]
    }
    case MDefinition::Opcode::GuardSpecificFunction: {
      // Passthrough operand 0 (the function); deopt if it != the expected
      // function (operand 1, an object pointer). Both are i32 object ptrs.
      MDefinition* obj = ins->getOperand(0);
      MDefinition* expected = ins->getOperand(1);
      int32_t ol = be.local(obj), xl = be.local(expected);
      if (ol < 0 || xl < 0) return false;
      if (WJGETENV("GECKO_WJ_GSFPASS")) return GetLocal(e, uint32_t(ol));
      if (!GetLocal(e, uint32_t(ol)) || !GetLocal(e, uint32_t(xl)) ||
          !e.writeOp(Op::I32Ne)) {
        return false;
      }
      if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
      if (!EmitDeopt(e, be)) return false;
      if (!e.writeOp(Op::End)) return false;
      return GetLocal(e, uint32_t(ol));
    }
    case MDefinition::Opcode::GuardObjectIdentity: {
      // Passthrough operand 0; deopt if (obj==expected) when bailOnEquality, else
      // if (obj!=expected). Both are i32 object pointers.
      MGuardObjectIdentity* g = ins->toGuardObjectIdentity();
      int32_t ol = be.local(g->object()), xl = be.local(g->expected());
      if (ol < 0 || xl < 0) return false;
      if (!GetLocal(e, uint32_t(ol)) || !GetLocal(e, uint32_t(xl))) return false;
      if (!e.writeOp(g->bailOnEquality() ? Op::I32Eq : Op::I32Ne)) return false;
      if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
      if (!EmitDeopt(e, be)) return false;
      if (!e.writeOp(Op::End)) return false;
      return GetLocal(e, uint32_t(ol));
    }
    case MDefinition::Opcode::ConstantProto: {
      // Wraps a constant prototype object; its value is just that constant.
      int32_t l = be.local(ins->getOperand(0));
      if (l < 0) return false;
      return GetLocal(e, uint32_t(l));
    }
    case MDefinition::Opcode::GuardValue: {
      // Passthrough the boxed value operand; deopt unless it equals the expected
      // (baked) value (raw 64-bit box compare). Nursery-index expected (a movable
      // baked object) is rare -- bail to PBL for it.
      MGuardValue* g = ins->toGuardValue();
      if (WJGETENV("GECKO_WJ_NO_GVAL")) return false;
      if (!g->expected().isValue()) return false;
      int32_t l = be.local(g->value());
      if (l < 0) return false;
      uint64_t bits = g->expected().asRawBits();
      if (!GetLocal(e, uint32_t(l)) || !e.writeOp(Op::I64Const) ||
          !e.writeVarS64(int64_t(bits)) || !e.writeOp(Op::I64Ne))
        return false;
      if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
      if (!EmitDeopt(e, be)) return false;
      if (!e.writeOp(Op::End)) return false;
      return GetLocal(e, uint32_t(l));
    }
    case MDefinition::Opcode::ArgumentsLength: {
      // arguments.length / actual arg count. Snapshotted into actualArgcLocal at
      // function entry from gWJCallArgc (see prologue, gated on be.usesArgc which the
      // op scan set when it saw this op). Result is Int32.
      // INLINED frame -> bail: the snapshot/staged params belong to the OUTER
      // function; an inlined callee's arguments.* must come from its call site
      // (overRest-in-lodash read argc=0 from the top-level script). Same check
      // as Rest.
      if (ins->block()->info().script() != be.info->script()) { fprintf(stderr, "[wj-inlargs] ArgumentsLength inlined\n"); return false; }
      return GetLocal(e, be.actualArgcLocal);
    }
    case MDefinition::Opcode::Clz: {
      // Math.clz32 / count-leading-zeros on an Int32 operand -> wasm i32.clz (exact;
      // i32.clz(0)==32 matches Math.clz32(0)). Operand must be i32-repr.
      MDefinition* a = ins->getOperand(0);
      int32_t l = be.local(a);
      if (l < 0 || WJValType(a->type()) != uint8_t(TypeCode::I32)) return false;
      if (!GetLocal(e, uint32_t(l))) return false;
      return e.writeOp(Op::I32Clz);
    }
    case MDefinition::Opcode::GuardSpecificSymbol: {
      // Passthrough the symbol operand; deopt unless its pointer equals the baked
      // expected JS::Symbol*. Symbols are interned/unique so identity IS pointer
      // identity -- a bare pointer compare is exact (no char-equal fallback needed).
      // A stale baked pointer (compacting GC moved the symbol) makes the compare fail
      // -> deopt to PBL, which re-checks correctly: fail-safe, same as GuardShape bakes.
      MGuardSpecificSymbol* g = ins->toGuardSpecificSymbol();
      MDefinition* sym = ins->getOperand(0);
      int32_t l = be.local(sym);
      if (l < 0 || WJValType(sym->type()) != uint8_t(TypeCode::I32)) return false;
      if (!GetLocal(e, uint32_t(l))) return false;
      if (!e.writeOp(Op::I32Const) ||
          !e.writeVarS32(int32_t(uintptr_t(static_cast<void*>(g->expected())))))
        return false;
      if (!e.writeOp(Op::I32Ne)) return false;  // ptr != expected -> deopt
      if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
      if (!EmitDeopt(e, be)) return false;
      if (!e.writeOp(Op::End)) return false;
      return GetLocal(e, uint32_t(l));
    }
    case MDefinition::Opcode::GuardSpecificInt32: {
      // Passthrough the Int32 operand; deopt unless it equals the baked expected value.
      // (Mirrors GuardSpecificSymbol/Atom for an int32.) Operand must be i32-repr.
      MDefinition* n = ins->getOperand(0);
      int32_t l = be.local(n);
      if (l < 0 || WJValType(n->type()) != uint8_t(TypeCode::I32)) return false;
      if (!GetLocal(e, uint32_t(l))) return false;
      if (!e.writeOp(Op::I32Const) ||
          !e.writeVarS32(ins->toGuardSpecificInt32()->expected()))
        return false;
      if (!e.writeOp(Op::I32Ne)) return false;  // != expected -> deopt
      if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
      if (!EmitDeopt(e, be)) return false;
      if (!e.writeOp(Op::End)) return false;
      return GetLocal(e, uint32_t(l));
    }
    case MDefinition::Opcode::NewBoundFunction: {
      // fn.bind fast path: allocate a BoundFunctionObject from the baked template.
      // Cold (alloc) -> helper (js::BoundFunctionObject::createWithTemplate).
      uint64_t tbits =
          JS::ObjectValue(*ins->toNewBoundFunction()->templateObj()).asRawBits();
      if (!EmitStageConstBoxed(e, be, tbits, 0)) return false;
      if (!EmitHelperCallResult(e, be, ins, WJH_NEWBOUNDFN, 0)) return false;
      return EmitHelperResultAsType(e, be, jit::MIRType::Object);
    }
    case MDefinition::Opcode::GetIteratorCache: {
      // for-in iterator setup (JSOp::Iter) -> js::ValueToIterator (alloc + enumerate
      // own/proto keys). Cold + uninlinable -> helper. Operand is ValueOrObject.
      if (!EmitStageScratch(e, be, ins->getOperand(0), 0)) return false;  // value
      if (!EmitHelperCallResult(e, be, ins, WJH_GETITER, 0)) return false;
      return EmitHelperResultAsType(e, be, jit::MIRType::Object);
    }
    case MDefinition::Opcode::IteratorMore: {
      // for-in advance: next enumerated key (String) or MagicValue(JS_NO_ITER_VALUE)
      // when exhausted. NativeIterator cursor walk (no GC/throw) -> minimal helper.
      if (!EmitStageScratch(e, be, ins->getOperand(0), 0)) return false;  // iterator
      if (!EmitHelperCallResult(e, be, ins, WJH_ITERMORE, 0)) return false;
      return EmitHelperResultAsType(e, be, jit::MIRType::Value);
    }
    case MDefinition::Opcode::IsNoIter: {
      // true iff the value is MagicValue(JS_NO_ITER_VALUE) -- the for-in done sentinel.
      // That magic is a unique 64-bit NaN-box pattern, so a bare i64 compare is exact.
      MDefinition* v = ins->getOperand(0);
      int32_t l = be.local(v);
      if (l < 0 || WJValType(v->type()) != uint8_t(TypeCode::I64)) return false;
      if (!GetLocal(e, uint32_t(l))) return false;
      if (!e.writeOp(Op::I64Const) ||
          !e.writeVarS64(int64_t(JS::MagicValue(JS_NO_ITER_VALUE).asRawBits())))
        return false;
      return e.writeOp(Op::I64Eq);  // -> i32 boolean
    }
    case MDefinition::Opcode::LoadSlotByIteratorIndex: {
      // for-in `obj[key]` optimized read: load object's slot/element at the iterator's
      // current PropertyIndex. Cold + needs NativeIterator index internals -> helper.
      if (!EmitStageScratch(e, be, ins->getOperand(0), 0)) return false;  // object
      if (!EmitStageScratch(e, be, ins->getOperand(1), 1)) return false;  // iterator
      if (!EmitHelperCallResult(e, be, ins, WJH_LOADSLOTBYITER, 0)) return false;
      return EmitHelperResultAsType(e, be, jit::MIRType::Value);
    }
    case MDefinition::Opcode::IteratorHasIndices: {
      // for-in fast-path eligibility: NativeIterator has indices available AND the
      // object's shape still matches the iterator's stored shape. Cold -> helper.
      if (!EmitStageScratch(e, be, ins->getOperand(0), 0)) return false;  // object
      if (!EmitStageScratch(e, be, ins->getOperand(1), 1)) return false;  // iterator
      if (!EmitHelperCallResult(e, be, ins, WJH_ITERHASINDICES, 0)) return false;
      return EmitHelperResultAsType(e, be, jit::MIRType::Boolean);
    }
    case MDefinition::Opcode::IteratorLength: {
      // MIteratorLength: the NativeIterator's ownPropertyCount. Cold (iterator-helper
      // pre-size) -> minimal helper (reads the field; no GC/throw).
      if (!EmitStageScratch(e, be, ins->getOperand(0), 0)) return false;  // iterator
      if (!EmitHelperCallResult(e, be, ins, WJH_ITERLENGTH, 0)) return false;
      return EmitHelperResultAsType(e, be, jit::MIRType::Int32);
    }
    case MDefinition::Opcode::ObjectKeysFromIterator: {
      // Object.keys fast path from an already-built iterator -> js::ObjectKeysFromIterator
      // (alloc result array). Cold -> helper.
      if (!EmitStageScratch(e, be, ins->getOperand(0), 0)) return false;  // iterator
      if (!EmitHelperCallResult(e, be, ins, WJH_OBJKEYSITER, 0)) return false;
      return EmitHelperResultAsType(e, be, jit::MIRType::Object);
    }
    case MDefinition::Opcode::GetDOMProperty: {
      // DOM getter (JSJitGetterOp ABI): call js::jit::CallDOMGetter (the exact Ion VM
      // fallback -- loads DOM_OBJECT_SLOT + invokes info->getter) via a cold helper. The
      // object (operand 0) is the GUARDED DOM object (upstream GuardShape-family ops ensure
      // the class; if a guard bails the whole fn is PBL). fun() is the static (non-GC)
      // JSJitGetterOp -- baked as a wasm32 ptr (fn-table index) into scratch[1]; the helper
      // replicates CallDOMGetter (load DOM_OBJECT_SLOT + call the getter). info() is protected,
      // so bake the public getter directly. Default-on; GECKO_WJ_NOGETDOMPROP.
      if (WJGETENV("GECKO_WJ_NOGETDOMPROP")) return false;
      MGetDOMProperty* g = ins->toGetDOMProperty();
      if (!EmitStageScratch(e, be, g->getOperand(0), 0)) return false;  // DOM object
      uint32_t getterAddr = uint32_t(uintptr_t(reinterpret_cast<void*>(g->fun())));
      if (!GetLocal(e, be.scratchBase) || !e.writeOp(Op::I64Const) ||
          !e.writeVarS64(int64_t(uint64_t(getterAddr))) || !e.writeOp(Op::I64Store) ||
          !e.writeVarU32(3) || !e.writeVarU32(1 * 8))
        return false;
      if (!EmitHelperCallResult(e, be, ins, WJH_GETDOMPROP, 0)) return false;
      return EmitHelperResultAsType(e, be, jit::MIRType::Value);
    }
    case MDefinition::Opcode::LoadIteratorElement: {
      // ObjectKeysReplacer (ScalarReplacement): load the index-th iterated key STRING
      // directly from the NativeIterator's property array -- `Object.keys(o)[i]` without
      // materializing the keys array. iter=op0 (PropertyIteratorObject), index=op1 (Int32,
      // in-bounds by the pass's invariant, matching Ion's unchecked iteratorLoadElement).
      // Pure read (no GC/throw) -> cold helper.
      if (WJGETENV("GECKO_WJ_NOLOADITERELEM")) return false;
      if (!EmitStageScratch(e, be, ins->getOperand(0), 0)) return false;  // iterator
      if (!EmitStageScratch(e, be, ins->getOperand(1), 1)) return false;  // index (Int32)
      if (!EmitHelperCallResult(e, be, ins, WJH_LOADITERELEM, 0)) return false;
      return EmitHelperResultAsType(e, be, jit::MIRType::String);
    }
    case MDefinition::Opcode::BindFunction: {
      // Function.prototype.bind: stage target (operand 0) + the bound args (operands
      // 1..numStackArgs; args[0] is the new `this`) and call BoundFunctionObject::
      // functionBindImpl in a cold helper (alloc + arg-copy). Pass maybeBound=null so
      // C++ does the alloc (the template fast-path is just an optimization). site=argc.
      MBindFunction* b = ins->toBindFunction();
      uint32_t argc = b->numStackArgs();
      if (uint32_t(1) + argc > kWJScratchSlots) return false;  // too many bound args -> PBL
      if (!EmitStageScratch(e, be, b->target(), 0)) return false;  // target (Object)
      for (uint32_t i = 0; i < argc; i++) {
        if (!EmitStageScratch(e, be, b->getArg(i), 1 + i)) return false;  // bound arg i
      }
      if (!EmitHelperCallResult(e, be, ins, WJH_BINDFUNCTION, argc)) return false;
      return EmitHelperResultAsType(e, be, jit::MIRType::Object);
    }
    case MDefinition::Opcode::CharCodeConvertCase: {
      // String.prototype.to{Lower,Upper}Case on a single char code -> 1-char string.
      // Cold (unicode case fold + string alloc) -> helper; site = case (0 lower/1 upper).
      MCharCodeConvertCase* m = ins->toCharCodeConvertCase();
      if (!EmitStageScratch(e, be, ins->getOperand(0), 0)) return false;  // code (Int32)
      uint32_t mode = (m->stringCase() == jit::StringCase::Lower) ? 0u : 1u;
      if (!EmitHelperCallResult(e, be, ins, WJH_CHARCASE, mode)) return false;
      return EmitHelperResultAsType(e, be, jit::MIRType::String);
    }
    case MDefinition::Opcode::BoundFunctionNumArgs: {
      // A bound function's stored arg count: unbox Int32 from the FlagsSlot and
      // shift right past the is-constructor bit. Immutable slot -> pure inline read.
      MDefinition* obj = ins->getOperand(0);
      int32_t l = be.local(obj);
      if (l < 0 || WJValType(obj->type()) != uint8_t(TypeCode::I32)) return false;
      if (!GetLocal(e, uint32_t(l))) return false;  // bound fn obj ptr (i32)
      if (!e.writeOp(Op::I64Load) || !e.writeVarU32(3) ||
          !e.writeVarU32(uint32_t(js::BoundFunctionObject::offsetOfFlagsSlot())))
        return false;                               // boxed Int32 flags Value
      if (!e.writeOp(Op::I32WrapI64)) return false;  // low32 = flags int
      if (!e.writeOp(Op::I32Const) ||
          !e.writeVarS32(int32_t(js::BoundFunctionObject::NumBoundArgsShift)))
        return false;
      return e.writeOp(Op::I32ShrS);  // flags >> NumBoundArgsShift = numBoundArgs
    }
    case MDefinition::Opcode::LoadWrapperTarget: {
      // Proxy target = the proxy's private slot (an Object). Pure inline slot read.
      // If fallible (revoked proxies possible), the slot may not be an Object ->
      // tag-guard and deopt to PBL (which bails for revoked proxies).
      MLoadWrapperTarget* m = ins->toLoadWrapperTarget();
      MDefinition* obj = ins->getOperand(0);
      int32_t l = be.local(obj);
      if (l < 0 || WJValType(obj->type()) != uint8_t(TypeCode::I32)) return false;
      if (!GetLocal(e, uint32_t(l))) return false;  // proxy obj ptr (i32)
      if (!e.writeOp(Op::I64Load) || !e.writeVarU32(3) ||
          !e.writeVarU32(uint32_t(js::ProxyObject::offsetOfPrivateSlot())))
        return false;                               // boxed target Value
      if (m->fallible()) {
        if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(be.unboxScratch))
          return false;
        if (!EmitTagGuardLocal(e, be, be.unboxScratch, TagWord(JSVAL_TYPE_OBJECT)))
          return false;  // not an object (revoked) -> deopt
        if (!GetLocal(e, be.unboxScratch)) return false;
      }
      return e.writeOp(Op::I32WrapI64);  // low32 = target object ptr
    }
    case MDefinition::Opcode::ApplyArray: {
      // fn.apply(thisArg, denseArray): spread the array's elements as call args.
      // Variable-argc -> not expressible as a fixed wasm call; delegate the whole
      // call to a cold helper (JS::Call). Stage the ARRAY OBJECT (the MElements'
      // source object), NOT the raw Elements ptr, so it stays GC-rooted and the
      // helper's getDenseElement handles shifted elements.
      MApplyArray* a = ins->toApplyArray();
      MDefinition* elems = a->getElements();
      if (!elems->isElements()) return false;  // unexpected elements producer -> PBL
      MDefinition* arrObj = elems->toElements()->object();
      if (!EmitStageScratch(e, be, a->getFunction(), 0)) return false;  // callee
      if (!EmitStageScratch(e, be, a->getThis(), 1)) return false;      // thisArg
      if (!EmitStageScratch(e, be, arrObj, 2)) return false;            // args array obj
      if (!EmitHelperCallResult(e, be, ins, WJH_APPLYARRAY, 0)) return false;
      return EmitHelperResultAsType(e, be, jit::MIRType::Value);
    }
    case MDefinition::Opcode::CheckPrivateFieldCache: {
      // `#x in obj` / `obj.#x` brand check. The ThrowCondition is read from the bytecode
      // pc (not carried on the MIR). The OP'S OWN bytecode pc is baked into scratch[2] and
      // the helper calls js::CheckPrivateFieldOperation.
      // NOTE: ins->trackedSite()->pc() is BLOCK-GRANULAR in this backend (returns block-entry,
      // NOT the CheckPrivateField op's pc -- it reads as a wrong opcode), so it would decode
      // the WRONG ThrowCondition. We instead SCAN the op's own script for the single
      // JSOp::CheckPrivateField pc and bail when absent/ambiguous (>1 in the script). Validated:
      // `#x in o` (OnlyCheckRhs), `obj.#x` get (ThrowHasNot), `#x=v` init (ThrowHas) all
      // JIT==PBL (/tmp/privfield_all.js PFALL=32400000). Multi-CPF fns bail to PBL (correct).
      const jit::BytecodeSite* site = ins->trackedSite();
      if (!site) return false;
      // trackedSite()->pc() is BLOCK-GRANULAR here (returns the block-entry pc, not the
      // CheckPrivateField op's own pc) -- verified: it reads as off=0/op!=CheckPrivateField,
      // so GetCheckPrivateFieldOperands would decode the WRONG bytecode's ThrowCondition.
      // Recover the op's exact pc by scanning the op's own script (site->script() handles
      // inlining) for the single JSOp::CheckPrivateField; bail if absent or ambiguous (>1).
      JSScript* sc = site->script();
      if (!sc) sc = be.info ? be.info->script() : nullptr;
      if (!sc) return false;
      // Prefer THIS node's op-accurate pc from its resume point (rp->pc() is the
      // bytecode-exact resume pc used for deopt, unlike the block-granular
      // trackedSite). This lets multi-#field scripts (a class using `#x` in >1
      // method -- the common real-world case) JIT each CheckPrivateField with its
      // OWN ThrowCondition. Fall back to the unique-scan only if no resume pc lands
      // on a CheckPrivateField op (then require exactly one, as before).
      auto pcIsCPF = [&](jsbytecode* p) -> bool {
        return p && p >= sc->code() && p < sc->codeEnd() &&
               JSOp(*p) == JSOp::CheckPrivateField;
      };
      jsbytecode* opPc = nullptr;
      if (MResumePoint* rp = ins->resumePoint()) {
        if (pcIsCPF(rp->pc())) opPc = rp->pc();
      }
      if (!opPc && be.curRp && pcIsCPF(be.curRp->pc())) opPc = be.curRp->pc();
      if (!opPc) {
        int cpfCount = 0;
        for (jsbytecode* p = sc->code(); p < sc->codeEnd();
             p += js::GetBytecodeLength(p)) {
          if (JSOp(*p) == JSOp::CheckPrivateField) { opPc = p; cpfCount++; }
        }
        if (cpfCount != 1 || !opPc) return false;
      }
      if (WJGETENV("GECKO_WJ_CPFCDBG")) {
        js::ThrowCondition c; js::ThrowMsgKind k;
        js::GetCheckPrivateFieldOperands(opPc, &c, &k);
        fprintf(stderr, "[cpfc-pc] fnLine=%u cond=%d off=%ld\n",
                unsigned(sc->lineno()), int(c), long(opPc - sc->code()));
      }
      if (!EmitStageScratch(e, be, ins->getOperand(0), 0)) return false;  // value
      if (!EmitStageScratch(e, be, ins->getOperand(1), 1)) return false;  // idval (private-name Symbol)
      uint32_t pc = uint32_t(uintptr_t(opPc));
      if (!GetLocal(e, be.scratchBase) || !e.writeOp(Op::I64Const) ||
          !e.writeVarS64(int64_t(uint64_t(pc))) || !e.writeOp(Op::I64Store) ||
          !e.writeVarU32(3) || !e.writeVarU32(2 * 8))
        return false;
      if (!EmitHelperCallResult(e, be, ins, WJH_CHECKPRIVFIELD, 0)) return false;
      return EmitHelperResultAsType(e, be, jit::MIRType::Boolean);
    }
    case MDefinition::Opcode::GuardSpecificAtom: {
      // Passthrough the string operand; deopt unless it is pointer- or char-equal
      // to the baked atom. The helper does the pointer-compare fast path itself.
      MGuardSpecificAtom* g = ins->toGuardSpecificAtom();
      if (WJGETENV("GECKO_WJ_NO_GATOM")) return false;
      int32_t l = be.local(g->str());
      if (l < 0) return false;
      if (!EmitStageScratch(e, be, g->str(), 0)) return false;
      uint64_t atomBits =
          JS::StringValue(reinterpret_cast<JSString*>(g->atom())).asRawBits();
      if (!EmitStageConstBoxed(e, be, atomBits, 1)) return false;
      if (!EmitHelperCallResult(e, be, ins, WJH_STREQATOM, 0)) return false;
      if (!EmitHelperResultAsType(e, be, jit::MIRType::Boolean)) return false;
      if (!e.writeOp(Op::I32Eqz)) return false;  // !match -> deopt
      if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
      if (!EmitDeopt(e, be)) return false;
      if (!e.writeOp(Op::End)) return false;
      return GetLocal(e, uint32_t(l));
    }
    case MDefinition::Opcode::ObjectStaticProto: {
      // Load obj->shape()->base()->proto() (the static, non-lazy prototype). Mirrors
      // visitObjectStaticProto (masm.loadObjProto). result = proto object pointer.
      if (WJGETENV("GECKO_WJ_NO_OSPROTO")) return false;
      if (!GetOp(e, be, ins->getOperand(0))) return false;
      if (!e.writeOp(Op::I32Load) || !e.writeVarU32(2) ||
          !e.writeVarU32(uint32_t(offsetof(JS::shadow::Object, shape))))
        return false;  // shape
      if (!e.writeOp(Op::I32Load) || !e.writeVarU32(2) ||
          !e.writeVarU32(uint32_t(js::Shape::offsetOfBaseShape())))
        return false;  // baseShape
      if (!e.writeOp(Op::I32Load) || !e.writeVarU32(2) ||
          !e.writeVarU32(uint32_t(js::BaseShape::offsetOfProto())))
        return false;  // proto (TaggedProto; object ptr for a static proto)
      return true;
    }
    case MDefinition::Opcode::GetPropSuperCache: {
      // `super.prop` / `super[expr]`: read `idval` off `object` (the super base =
      // HomeObjectSuperBase result) but invoke getters with `receiver` (this). Helper:
      // ToPropertyKey(idval) + js::GetProperty(obj, receiver, id). Residual super bail
      // (vuejs.org) after HomeObjectSuperBase un-park. May GC/throw.
      MGetPropSuperCache* g = ins->toGetPropSuperCache();
      if (!EmitStageScratch(e, be, g->object(), 0)) return false;    // super base (Object)
      if (!EmitStageScratch(e, be, g->receiver(), 1)) return false;  // this (Value)
      if (!EmitStageScratch(e, be, g->idval(), 2)) return false;     // key (Value)
      if (!EmitHelperCallResult(e, be, ins, WJH_GETPROPSUPER, 0)) return false;
      return EmitHelperResultAsType(e, be, jit::MIRType::Value);
    }
    case MDefinition::Opcode::AsyncResolve: {
      // MAsyncResolve: resolve an async fn's promise at its return/resolve point via
      // js::AsyncFunctionResolve(cx, generator, value) -> the promise Object. Whole async
      // fns bailed to PBL here (reddit.com x3). Stage generator (Object) + value (boxed);
      // may GC/OOM.
      auto* a = ins->toAsyncResolve();
      if (!EmitStageScratch(e, be, a->generator(), 0)) return false;  // generator (Object)
      if (!EmitStageScratch(e, be, a->value(), 1)) return false;  // value (Value)
      if (!EmitHelperCallResult(e, be, ins, WJH_ASYNCRESOLVE, 0)) return false;
      return EmitHelperResultAsType(e, be, jit::MIRType::Object);
    }
    case MDefinition::Opcode::CanSkipAwait: {
      // Await-skip optimization: MCanSkipAwait -> Boolean via js::CanSkipAwait (true if
      // the awaited value is not a thenable, so await resolves synchronously). Pairs with
      // MaybeExtractAwaitValue below; both bailed async fns to PBL (reddit.com). May GC.
      if (!EmitStageScratch(e, be, ins->getOperand(0), 0)) return false;
      if (!EmitHelperCallResult(e, be, ins, WJH_CANSKIPAWAIT, 0)) return false;
      return EmitHelperResultAsType(e, be, jit::MIRType::Boolean);
    }
    case MDefinition::Opcode::MaybeExtractAwaitValue: {
      // if canSkip -> js::ExtractAwaitValue(value) (unwrap the resolved value); else pass
      // `value` through unchanged. Mirrors Ion visitMaybeExtractAwaitValue. Result Value.
      auto* m = ins->toMaybeExtractAwaitValue();
      int32_t cl = be.local(m->canSkip()), vl = be.local(m->value());
      if (cl < 0 || vl < 0) return false;
      if (!GetLocal(e, uint32_t(cl))) return false;                 // canSkip (i32 bool)
      if (!e.writeOp(Op::If) || !e.writeFixedU8(0x7E)) return false;  // result i64 (Value)
      if (!EmitStageScratch(e, be, m->value(), 0) ||
          !EmitHelperCallResult(e, be, ins, WJH_EXTRACTAWAITVALUE, 0) ||
          !EmitHelperResultAsType(e, be, jit::MIRType::Value))
        return false;
      if (!e.writeOp(Op::Else)) return false;
      if (!GetLocal(e, uint32_t(vl))) return false;                 // passthrough value
      return e.writeOp(Op::End);
    }
    case MDefinition::Opcode::GuardProto: {
      // Guard obj's static prototype == expected; deopt on mismatch. Mirrors
      // visitGuardProto (loadObjProto + branch-not-equal bailout). No helper;
      // passthrough the guarded object as the result (MIRType::Object).
      if (WJGETENV("GECKO_WJ_NO_GUARDPROTO")) return false;
      MGuardProto* g = ins->toGuardProto();
      if (WJDbg()) fprintf(stderr, "[wb-be] GuardProto inline\n");
      if (!GetOp(e, be, g->object())) return false;
      if (!e.writeOp(Op::I32Load) || !e.writeVarU32(2) ||
          !e.writeVarU32(uint32_t(offsetof(JS::shadow::Object, shape))))
        return false;
      if (!e.writeOp(Op::I32Load) || !e.writeVarU32(2) ||
          !e.writeVarU32(uint32_t(js::Shape::offsetOfBaseShape())))
        return false;
      if (!e.writeOp(Op::I32Load) || !e.writeVarU32(2) ||
          !e.writeVarU32(uint32_t(js::BaseShape::offsetOfProto())))
        return false;
      if (!GetOp(e, be, g->expected())) return false;
      if (!e.writeOp(Op::I32Ne)) return false;
      if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
      if (!EmitDeopt(e, be)) return false;
      if (!e.writeOp(Op::End)) return false;
      return GetOp(e, be, g->object());
    }
    case MDefinition::Opcode::GuardIsNotProxy: {
      // Deopt if obj is a proxy. Mirrors masm.branchTestObjectIsProxy: a proxy's
      // Shape has kind == Proxy (== 0), so (shape->immutableFlags & (kindMask <<
      // kindShift)) == 0 IFF the object is a proxy. Passthrough object. No helper.
      if (WJGETENV("GECKO_WJ_NO_GUARDNOTPROXY")) return false;
      MDefinition* obj = ins->getOperand(0);
      if (!GetOp(e, be, obj)) return false;
      if (!e.writeOp(Op::I32Load) || !e.writeVarU32(2) ||
          !e.writeVarU32(uint32_t(offsetof(JS::shadow::Object, shape))))
        return false;
      if (!e.writeOp(Op::I32Load) || !e.writeVarU32(2) ||
          !e.writeVarU32(uint32_t(js::Shape::offsetOfImmutableFlags())))
        return false;
      if (!e.writeOp(Op::I32Const) ||
          !e.writeVarS32(int32_t(js::Shape::kindMask() << js::Shape::kindShift())))
        return false;
      if (!e.writeOp(Op::I32And)) return false;
      if (!e.writeOp(Op::I32Eqz)) return false;  // (flags & mask) == 0 => is proxy
      if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
      if (!EmitDeopt(e, be)) return false;
      if (!e.writeOp(Op::End)) return false;
      return GetOp(e, be, obj);
    }
    case MDefinition::Opcode::GuardFunctionFlags: {
      // Deopt unless fun's flags match: at least one expectedFlags bit set AND no
      // unexpectedFlags bit set. Mirrors masm.branchTestFunctionFlags (test32 of
      // the low-16 flags in flagsAndArgCount). Passthrough function. No helper.
      if (WJGETENV("GECKO_WJ_NO_GUARDFNFLAGS")) return false;
      MGuardFunctionFlags* g = ins->toGuardFunctionFlags();
      uint32_t expected = g->expectedFlags();
      uint32_t unexpected = g->unexpectedFlags();
      if (expected) {
        if (!GetOp(e, be, g->function())) return false;
        if (!e.writeOp(Op::I32Load) || !e.writeVarU32(2) ||
            !e.writeVarU32(uint32_t(JSFunction::offsetOfFlagsAndArgCount())))
          return false;
        if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(expected)) ||
            !e.writeOp(Op::I32And) || !e.writeOp(Op::I32Eqz))
          return false;  // (flags & expected) == 0 => missing expected => deopt
        if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
        if (!EmitDeopt(e, be)) return false;
        if (!e.writeOp(Op::End)) return false;
      }
      if (unexpected) {
        if (!GetOp(e, be, g->function())) return false;
        if (!e.writeOp(Op::I32Load) || !e.writeVarU32(2) ||
            !e.writeVarU32(uint32_t(JSFunction::offsetOfFlagsAndArgCount())))
          return false;
        if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(unexpected)) ||
            !e.writeOp(Op::I32And))
          return false;  // (flags & unexpected) != 0 => deopt
        if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
        if (!EmitDeopt(e, be)) return false;
        if (!e.writeOp(Op::End)) return false;
      }
      return GetOp(e, be, g->function());
    }
    case MDefinition::Opcode::GuardToFunction: {
      // Passthrough the object operand; function-ness is established by the
      // following GuardFunctionScript / call-site guards.
      int32_t l = be.local(ins->getOperand(0));
      if (l < 0) return false;
      return GetLocal(e, uint32_t(l));
    }
    case MDefinition::Opcode::GuardFunctionIsNonBuiltinCtor: {
      // Deopt unless the function is a non-builtin constructor: (flags & mask) ==
      // (BASESCRIPT|CONSTRUCTOR), i.e. it has BASESCRIPT+CONSTRUCTOR and is NOT
      // SELF_HOSTED. Mirrors MacroAssembler::branchIfNotFunctionIsNonBuiltinCtor
      // (load the 32-bit flagsAndArgCount word, flags in the low bits). Passthrough
      // the function object. (discord.com/login bailed a fn here.)
      int32_t l = be.local(ins->getOperand(0));
      if (l < 0) return false;
      constexpr int32_t mask = js::FunctionFlags::BASESCRIPT |
                               js::FunctionFlags::SELF_HOSTED |
                               js::FunctionFlags::CONSTRUCTOR;
      constexpr int32_t expected =
          js::FunctionFlags::BASESCRIPT | js::FunctionFlags::CONSTRUCTOR;
      if (!GetLocal(e, uint32_t(l)) || !e.writeOp(Op::I32Load) || !e.writeVarU32(2) ||
          !e.writeVarU32(uint32_t(JSFunction::offsetOfFlagsAndArgCount())) ||
          !e.writeOp(Op::I32Const) || !e.writeVarS32(mask) || !e.writeOp(Op::I32And) ||
          !e.writeOp(Op::I32Const) || !e.writeVarS32(expected) ||
          !e.writeOp(Op::I32Ne))
        return false;
      if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
      if (!EmitDeopt(e, be)) return false;
      if (!e.writeOp(Op::End)) return false;
      return GetLocal(e, uint32_t(l));  // passthrough the function object
    }
    case MDefinition::Opcode::LoadGetterSetterFunction: {
      // getterSetter is a Value boxing a GetterSetter* (JSVAL_TYPE_PRIVATE_GCTHING);
      // its payload is the raw ptr in the low 32 bits. Load the getter or setter
      // JSFunction* at the fixed offset, DEOPT if null. Mirrors Ion
      // visitLoadGetterSetterFunction. needsClassGuard (the accessor may be a
      // non-function -- rare) -> bail; the object-is-function class check isn't
      // inlined here. (discord.com/login bailed 3 fns here.)
      auto* lgs = ins->toLoadGetterSetterFunction();
      if (lgs->needsClassGuard()) return false;
      int32_t l = be.local(lgs->getterSetter());
      if (l < 0) return false;
      uint32_t off = uint32_t(lgs->isGetter() ? js::GetterSetter::offsetOfGetter()
                                              : js::GetterSetter::offsetOfSetter());
      // gs* = low32(box); fn = *(gs+off) (zero-extended into unboxScratch)
      if (!GetLocal(e, uint32_t(l)) || !e.writeOp(Op::I32WrapI64) ||
          !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(off) ||
          !e.writeOp(Op::I64ExtendI32U) || !e.writeOp(Op::LocalSet) ||
          !e.writeVarU32(be.unboxScratch))
        return false;
      // fn == null -> deopt
      if (!GetLocal(e, be.unboxScratch) || !e.writeOp(Op::I64Eqz)) return false;
      if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
      if (!EmitDeopt(e, be)) return false;
      if (!e.writeOp(Op::End)) return false;
      return GetLocal(e, be.unboxScratch) && e.writeOp(Op::I32WrapI64);  // fn (Object)
    }
    case MDefinition::Opcode::GuardNullOrUndefined: {
      // Deopt unless the value's tag is NULL or UNDEFINED; passthrough the value.
      // (splay `x == null` fast paths.) tag = high 32 bits of the NUNBOX box.
      int32_t l = be.local(ins->getOperand(0));
      if (l < 0) return false;
      auto pushTag = [&]() -> bool {
        return GetLocal(e, uint32_t(l)) && e.writeOp(Op::I64Const) &&
               e.writeVarS64(32) && e.writeOp(Op::I64ShrU) &&
               e.writeOp(Op::I32WrapI64);
      };
      if (!pushTag() || !e.writeOp(Op::I32Const) ||
          !e.writeVarS32(int32_t(TagWord(JSVAL_TYPE_NULL))) || !e.writeOp(Op::I32Eq))
        return false;
      if (!pushTag() || !e.writeOp(Op::I32Const) ||
          !e.writeVarS32(int32_t(TagWord(JSVAL_TYPE_UNDEFINED))) ||
          !e.writeOp(Op::I32Eq))
        return false;
      if (!e.writeOp(Op::I32Or) || !e.writeOp(Op::I32Eqz)) return false;
      if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
      if (!EmitDeopt(e, be)) return false;
      if (!e.writeOp(Op::End)) return false;
      return GetLocal(e, uint32_t(l));  // passthrough the (null/undefined) value
    }
    case MDefinition::Opcode::GuardArrayIsPacked: {
      if (WJGETENV("GECKO_WJ_NOGUARDPACKED")) return false;
      // Deopt unless the array's dense elements are PACKED (no holes); passthrough the
      // array object. Ion: load elements header flags, check NON_PACKED bit clear.
      int32_t l = be.local(ins->toGuardArrayIsPacked()->array());
      if (l < 0) return false;
      // elements = array->elements_ ; flags = elements->flags. offsetOfFlags() is
      // NEGATIVE (elem-relative: the ObjectElements header is BEFORE elements_), so
      // it must be I32Add'd as a signed const -- NOT used as an unsigned wasm load
      // offset (which would be a huge OOB address). Same fix pattern as ArrayLength.
      if (!GetLocal(e, uint32_t(l)) || !e.writeOp(Op::I32Load) || !e.writeVarU32(2) ||
          !e.writeVarU32(uint32_t(js::NativeObject::offsetOfElements())))
        return false;
      if (!e.writeOp(Op::I32Const) ||
          !e.writeVarS32(js::ObjectElements::offsetOfFlags()) ||
          !e.writeOp(Op::I32Add) || !e.writeOp(Op::I32Load) || !e.writeVarU32(2) ||
          !e.writeVarU32(0))
        return false;
      if (!e.writeOp(Op::I32Const) ||
          !e.writeVarS32(int32_t(js::ObjectElements::NON_PACKED)) ||
          !e.writeOp(Op::I32And))
        return false;
      if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;  // non-packed
      if (!EmitDeopt(e, be)) return false;
      if (!e.writeOp(Op::End)) return false;
      return GetLocal(e, uint32_t(l));  // passthrough the array
    }
    case MDefinition::Opcode::InCache: {
      // MInCache (`key in obj`) -> WJH_INCACHE (js::jit::OperatorIn, general). Operands:
      // key(Value, op0), object(Object, op1). Eliminates the InCache bail.
      if (WJGETENV("GECKO_WJ_NOINCACHE")) return false;
      MDefinition* keyD = ins->getOperand(0);
      MDefinition* objD = ins->getOperand(1);
      if (objD->type() != MIRType::Object) return false;
      if (!EmitStageScratch(e, be, keyD, 0)) return false;
      if (!EmitStageScratch(e, be, objD, 1)) return false;
      if (!EmitHelperCallResult(e, be, ins, WJH_INCACHE, 0)) return false;
      return EmitHelperResultAsType(e, be, ins->type());  // Boolean
    }
    case MDefinition::Opcode::ArrayJoin: {
      // MArrayJoin (array.join(sep)) -> WJH_ARRAYJOIN (js::jit::ArrayJoin handles any
      // array). Eliminates the ArrayJoin bail (uBlock string-building). Operands:
      // array(Object), separator(String).
      if (WJGETENV("GECKO_WJ_NOARRAYJOIN")) return false;
      MArrayJoin* aj = ins->toArrayJoin();
      if (aj->array()->type() != MIRType::Object ||
          aj->separator()->type() != MIRType::String)
        return false;
      if (!EmitStageScratch(e, be, aj->array(), 0)) return false;
      if (!EmitStageScratch(e, be, aj->separator(), 1)) return false;
      if (!EmitHelperCallResult(e, be, ins, WJH_ARRAYJOIN, 0)) return false;
      return EmitHelperResultAsType(e, be, ins->type());  // String
    }
    case MDefinition::Opcode::ArraySlice: {
      // MArraySlice (array.slice(begin,end)): packed-guard + deopt (Ion does
      // branchArrayIsNotPacked->bail; ArraySliceDense asserts IsPackedArray), then
      // WJH_ARRAYSLICE allocates+copies the slice. Unblocks uBlock's getNodeTypes.
      if (WJGETENV("GECKO_WJ_NOARRAYSLICE")) return false;
      MArraySlice* as = ins->toArraySlice();
      int32_t ol = be.local(as->object());
      if (ol < 0 || as->object()->type() != MIRType::Object) return false;
      // packed guard: deopt unless dense elements are PACKED. offsetOfFlags() is
      // NEGATIVE (elem-relative) -> I32Add as signed const, not an unsigned offset.
      if (!GetLocal(e, uint32_t(ol)) || !e.writeOp(Op::I32Load) ||
          !e.writeVarU32(2) ||
          !e.writeVarU32(uint32_t(js::NativeObject::offsetOfElements())) ||
          !e.writeOp(Op::I32Const) ||
          !e.writeVarS32(js::ObjectElements::offsetOfFlags()) ||
          !e.writeOp(Op::I32Add) || !e.writeOp(Op::I32Load) || !e.writeVarU32(2) ||
          !e.writeVarU32(0) || !e.writeOp(Op::I32Const) ||
          !e.writeVarS32(int32_t(js::ObjectElements::NON_PACKED)) ||
          !e.writeOp(Op::I32And))
        return false;
      if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
      if (!EmitDeopt(e, be)) return false;
      if (!e.writeOp(Op::End)) return false;
      if (!EmitStageScratch(e, be, as->object(), 0)) return false;
      if (!EmitStageScratch(e, be, as->begin(), 1)) return false;
      if (!EmitStageScratch(e, be, as->end(), 2)) return false;
      if (!EmitHelperCallResult(e, be, ins, WJH_ARRAYSLICE, 0)) return false;
      return EmitHelperResultAsType(e, be, ins->type());  // Object
    }
    case MDefinition::Opcode::GuardIsNotObject: {
      if (WJGETENV("GECKO_WJ_NOGUARDNOTOBJ")) return false;
      // Passthrough the Value; deopt if its tag is OBJECT.
      int32_t l = be.local(ins->getOperand(0));
      if (l < 0) return false;
      if (!GetLocal(e, uint32_t(l)) || !e.writeOp(Op::I64Const) ||
          !e.writeVarS64(32) || !e.writeOp(Op::I64ShrU) ||
          !e.writeOp(Op::I32WrapI64) || !e.writeOp(Op::I32Const) ||
          !e.writeVarS32(int32_t(TagWord(JSVAL_TYPE_OBJECT))) || !e.writeOp(Op::I32Eq))
        return false;
      if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
      if (!EmitDeopt(e, be)) return false;
      if (!e.writeOp(Op::End)) return false;
      return GetLocal(e, uint32_t(l));  // passthrough Value
    }
    case MDefinition::Opcode::HomeObjectSuperBase: {
      // Result = the home object's [[Prototype]] (where super.x lookup begins).
      // A bare box of the home object would resolve super.x against the method's
      // OWN class, not the superclass -- so go through GetPrototype (helper).
      // 2026-07-04 default-on: the old
      // parking reason was a LATENT downstream "compile of null" miscompile in ubo
      // when super methods compiled fully -- that was a stale/dropped-null-callee
      // (rooting-miss class), now cleared by this session's rooting fixes (string-
      // rooting, ROOTLIVE, precise dataflow rooting). RE-VALIDATED: ubo 3/3 OK,
      // super.js JIT==PBL==7200000, octane no-ERR, babel clean. Unblocks super.x
      // (super methods/getters) on class-heavy real sites (surfaced as a
      // GetPropSuperCache cascade bail on vuejs.org; HomeObjectSuperBase is the root).
      MDefinition* ho = ins->toHomeObjectSuperBase()->homeObject();
      if (!EmitStageScratch(e, be, ho, 0)) return false;
      if (!EmitHelperCallResult(e, be, ins, WJH_HOMEPROTO, 0)) return false;
      return EmitHelperResultAsType(e, be, ins->type());  // Object
    }
    case MDefinition::Opcode::HomeObject: {
      // Load the method's [[HomeObject]] from its FunctionExtended slot (a boxed
      // ObjectValue; low 32 bits = the object pointer on wasm32). Default-on
      // 2026-07-03 (validated class/super correctness JIT==PBL).
      MDefinition* fn = ins->toHomeObject()->function();
      if (!GetOp(e, be, fn) || !e.writeOp(Op::I64Load) || !e.writeVarU32(3) ||
          !e.writeVarU32(uint32_t(js::FunctionExtended::offsetOfMethodHomeObjectSlot())))
        return false;
      return e.writeOp(Op::I32WrapI64);
    }
    case MDefinition::Opcode::GuardFunctionScript: {
      // Deopt unless the function's BaseScript matches the expected one. This is
      // the real guard behind polymorphic-call INLINING (each inlined target has a
      // distinct script). BUT if every use of this guard is just the CALLEE of a
      // real MCall (not an inlined body), the guard is redundant: our call path
      // dispatches ANY function via its polymorphic IC / wjhelp fallback, so a
      // mismatched callee does NOT need a deopt. Monomorphic GuardFunctionScript on
      // a genuinely polymorphic indirect call (gbemu memoryRead's
      // `this.memoryReader[address](...)` -- 65536 distinct handlers) was deopt-
      // storming EVERY call (3699 resumes -> the whole emulator memory path in PBL).
      // Passthrough (no deopt) when safe; keep the deopt when it protects an inlined
      // region (a use that is NOT a call callee). GECKO_WJ_NOGSFCALLPASS reverts.
      {
        bool allCallCallee = ins->hasUses();
        for (jit::MUseIterator u = ins->usesBegin(); u != ins->usesEnd(); u++) {
          jit::MNode* c = u->consumer();
          if (!c->isDefinition()) continue;  // resume points: shape-agnostic
          jit::MDefinition* d = c->toDefinition();
          if (d->isCall() && d->toCall()->getCallee() == ins) continue;
          allCallCallee = false;
          break;
        }
        // OPT-IN (GECKO_WJ_GSFCALLPASS): the passthrough KILLS the memoryRead
        // deopt storm (gbemu 4000->88 resumes) BUT exposes a latent correctness
        // bug somewhere in gbemu (ERR=Gameboy) when applied globally. The flag may
        // be a LINE NUMBER to scope the passthrough to one function (for bisection)
        // or "1" to apply everywhere. Keep the deopt by default until fixed.
        const char* gsfEnv = WJGETENV("GECKO_WJ_GSFCALLPASS");
        bool gsfOn = false;
        if (gsfEnv && allCallCallee) {
          int wantLine = atoi(gsfEnv);
          uint32_t fnLine =
              be.info && be.info->script() ? be.info->script()->lineno() : 0;
          gsfOn = (wantLine <= 1) || (uint32_t(wantLine) == fnLine);
        }
        if (gsfOn) {
          int32_t pl = be.local(ins->getOperand(0));
          if (pl < 0) return false;
          return GetLocal(e, uint32_t(pl));  // passthrough; call IC dispatches
        }
      }
      int32_t fl = be.local(ins->getOperand(0));
      if (fl < 0) return false;
      uintptr_t expected =
          uintptr_t(static_cast<void*>(ins->toGuardFunctionScript()->expected()));
      uint32_t off = JSFunction::offsetOfJitInfoOrScript();
      if (!GetLocal(e, uint32_t(fl))) return false;
      if (!e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(off))
        return false;
      if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(expected)) ||
          !e.writeOp(Op::I32Ne)) {
        return false;
      }
      if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
      if (!EmitDeopt(e, be)) return false;
      if (!e.writeOp(Op::End)) return false;
      return GetLocal(e, uint32_t(fl));
    }
    case MDefinition::Opcode::Callee: {
      // The running function object. For the OUTER frame this MUST be the
      // RUNTIME callee (rooted at envRootIdx+1 by the prologue): baking the
      // script's canonical function is WRONG for closures sharing a script --
      // an NFE self-call (acorn-walk `c`) then invoked the canonical function,
      // whose ->environment() is not the running closure's CallObject, and all
      // captured-var reads returned garbage. For an INLINED frame the callee is
      // pinned to a specific function by the inlining guard, so baking the
      // frame's function stays correct.
      bool outer = ins->block()->info().script() == be.info->script();
      if (outer && be.usesCallee && be.useEnvRoot) {
        uintptr_t rootsBase = uintptr_t(static_cast<void*>(&gWJCallRoots[0]));
        return e.writeOp(Op::LocalGet) && e.writeVarU32(be.envRootIdx) &&
               e.writeOp(Op::I32Const) && e.writeVarS32(8) && e.writeOp(Op::I32Mul) &&
               e.writeOp(Op::I32Const) && e.writeVarS32(int32_t(rootsBase)) &&
               e.writeOp(Op::I32Add) && e.writeOp(Op::I64Load) && e.writeVarU32(3) &&
               e.writeVarU32(8) && e.writeOp(Op::I32WrapI64);
      }
      JSScript* cs = ins->block()->info().script();
      JSFunction* fun = cs ? cs->function() : nullptr;
      if (!fun) return false;
      uintptr_t slot = WJInternConstant(JS::ObjectValue(*fun).asRawBits());
      if (!slot) return false;
      return e.writeOp(Op::I32Const) && e.writeVarS32(int32_t(slot)) &&
             e.writeOp(Op::I64Load) && e.writeVarU32(3) && e.writeVarU32(0) &&
             e.writeOp(Op::I32WrapI64);
    }
    case MDefinition::Opcode::FunctionEnvironment: {
      // result = function->environment(). Mirrors Ion's LFunctionEnvironment,
      // which loads the env from the SPECIFIC function operand -- crucial for
      // INLINED frames, whose function differs from the compiled (outer) one.
      // For the outer running function the operand is MCallee; its runtime
      // closure env isn't a compile-time value, so use envLocal (snapshotted
      // from gWJCurrentEnv at entry = the caller-provided runtime env). For any
      // other function operand (an inlined closure, materialized as a constant
      // or value) load ->environment() directly, like Ion.
      // CORRECTNESS: the only sound env we have is envLocal (snapshotted from
      // gWJCurrentEnv at entry = the caller-provided runtime env), valid ONLY for
      // the outer running function (operand == MCallee, not inlined). For inlined
      // frames or any other operand the env value is unsound -- loading
      // fn->environment() off a non-Callee operand read a wild pointer (splay OOB)
      // and mis-resolved closure vars (navier set_bnd, deltablue c.execute). So
      // BAIL those to PBL. Proper fix: env-PARAM ABI (pass each frame's env).
      if (WJGETENV("GECKO_WJ_NOFE")) return false;
      MDefinition* fn = ins->toFunctionEnvironment()->function();
      bool inlined = ins->block()->info().script() != be.info->script();
      if (WJGETENV("GECKO_WJ_FEDBG")) {
        JSScript* s = be.info->script();
        fprintf(stderr, "[wb-fedbg] FE fnOp=%s inlined=%d %s:%u\n",
                WJOpName(fn->op()), inlined, s ? s->filename() : "?",
                s ? s->lineno() : 0);
      }
      // Outer running function: rematerialize from the persistent GC-rooted env
      // slot (GetOp handles the same; this value-node store is mostly dead since
      // GetOp intercepts every use, but kept correct). Inlined/non-Callee env:
      // a different frame's env we don't persistently root -> bail to PBL.
      if (fn->isCallee() && !inlined) return GetOp(e, be, ins);
      // Any non-MCallee function operand (inlined callee, OR an outer-frame
      // GuardSpecificFunction referencing a specific closure -- navier): load that
      // function's env slot directly (GC-fresh). Keyed on the operand, not `inlined`.
      return EmitInlinedFEEnv(e, be, ins);  // false if operand not GetOp-able -> PBL
    }
    case MDefinition::Opcode::NurseryObject: {
      // A reference to a specific snapshot-held object; pool it (traced) and load
      // the live pointer like an object constant.
      if (!be.snapshot) return false;
      uint32_t idx = ins->toNurseryObject()->nurseryObjectIndex();
      if (idx >= be.snapshot->nurseryObjects().length()) return false;
      JSObject* obj = be.snapshot->nurseryObjects()[idx];
      if (!obj) return false;
      uintptr_t slot = WJInternConstant(JS::ObjectValue(*obj).asRawBits());
      if (!slot) return false;
      be.nurseryObjSlot[ins->id()] = slot;  // GetOp rematerializes from this traced slot
      return e.writeOp(Op::I32Const) && e.writeVarS32(int32_t(slot)) &&
             e.writeOp(Op::I64Load) && e.writeVarU32(3) && e.writeVarU32(0) &&
             e.writeOp(Op::I32WrapI64);
    }
    case MDefinition::Opcode::ArrayLength: {
      int32_t off = js::ObjectElements::offsetOfLength();  // negative (elem-relative)
      if (!GetOp(e, be, ins->getOperand(0)) || !e.writeOp(Op::I32Const) ||
          !e.writeVarS32(off) || !e.writeOp(Op::I32Add)) {
        return false;
      }
      return e.writeOp(Op::I32Load) && e.writeVarU32(2) && e.writeVarU32(0);
    }
    case MDefinition::Opcode::InitializedLength: {
      int32_t off = js::ObjectElements::offsetOfInitializedLength();
      if (!GetOp(e, be, ins->getOperand(0)) || !e.writeOp(Op::I32Const) ||
          !e.writeVarS32(off) || !e.writeOp(Op::I32Add)) {
        return false;
      }
      return e.writeOp(Op::I32Load) && e.writeVarU32(2) && e.writeVarU32(0);
    }
    case MDefinition::Opcode::BoundsCheck: {
      MBoundsCheck* bc = ins->toBoundsCheck();
      int32_t il = be.local(bc->index()), ll = be.local(bc->length());
      if (il < 0 || ll < 0) return false;
      // OOB-safe: skip the deopt; the dependent Value LoadElement self-bounds.
      if (WJBoundsCheckOOBSafe(ins)) return GetLocal(e, uint32_t(il));
      // PROBE (UNSAFE, correctness-ignored): skip ALL BoundsCheck deopts to measure
      // the perf ceiling if bounds-check deopts were free (gbemu's 357K hoisted-form
      // deopt storm). GECKO_WJ_NOBCDEOPT.
      if (WJGETENV("GECKO_WJ_NOBCDEOPT")) return GetLocal(e, uint32_t(il));
      int32_t mn = bc->minimum(), mx = bc->maximum();
      if (mn == 0 && mx == 0) {
        // Common case: deopt unless (uint32)index < (uint32)length.
        if (!GetLocal(e, uint32_t(il)) || !GetLocal(e, uint32_t(ll)) ||
            !e.writeOp(Op::I32GeU))
          return false;
        if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
        if (WJGETENV("GECKO_WJ_BCRT")) {  // store-only idx/len/hits at taken deopt
          uintptr_t ia = uintptr_t(static_cast<void*>(&gWJBCDbgIdx));
          uintptr_t la = uintptr_t(static_cast<void*>(&gWJBCDbgLen));
          uintptr_t ha = uintptr_t(static_cast<void*>(&gWJBCDbgHits));
          if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(ia)) ||
              !GetLocal(e, uint32_t(il)) || !e.writeOp(Op::I32Store) ||
              !e.writeVarU32(2) || !e.writeVarU32(0))
            return false;
          if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(la)) ||
              !GetLocal(e, uint32_t(ll)) || !e.writeOp(Op::I32Store) ||
              !e.writeVarU32(2) || !e.writeVarU32(0))
            return false;
          if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(ha)) ||
              !e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(ha)) ||
              !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0) ||
              !e.writeOp(Op::I32Const) || !e.writeVarS32(1) || !e.writeOp(Op::I32Add) ||
              !e.writeOp(Op::I32Store) || !e.writeVarU32(2) || !e.writeVarU32(0))
            return false;
        }
        if (!EmitDeopt(e, be)) return false;
        if (!e.writeOp(Op::End)) return false;
        return GetLocal(e, uint32_t(il));
      }
      // Hoisted range check: deopt unless (index+min) >= 0 && (index+max) < length
      // (signed; length is non-negative). i.e. deopt if (index+min < 0) ||
      // (index+max >= length).
      if (!GetLocal(e, uint32_t(il)) || !e.writeOp(Op::I32Const) ||
          !e.writeVarS32(mn) || !e.writeOp(Op::I32Add) ||
          !e.writeOp(Op::I32Const) || !e.writeVarS32(0) || !e.writeOp(Op::I32LtS))
        return false;  // (index+min) < 0
      if (!GetLocal(e, uint32_t(il)) || !e.writeOp(Op::I32Const) ||
          !e.writeVarS32(mx) || !e.writeOp(Op::I32Add) ||
          !GetLocal(e, uint32_t(ll)) || !e.writeOp(Op::I32GeS) ||
          !e.writeOp(Op::I32Or))
        return false;  // OR (index+max) >= length
      if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
      if (!EmitDeopt(e, be)) return false;
      if (!e.writeOp(Op::End)) return false;
      return GetLocal(e, uint32_t(il));  // passthrough index
    }
    // IntPtr <-> Int32 conversions: on wasm32 IntPtr == i32, so these are no-ops
    // (the framework stores the operand value into this node's local). The guard
    // form is a passthrough -- the typed-array BoundsCheck already validates range.
    case MDefinition::Opcode::Int32ToIntPtr:
    case MDefinition::Opcode::NonNegativeIntPtrToInt32:
    case MDefinition::Opcode::GuardInt32IsNonNegative:
      return GetOp(e, be, ins->getOperand(0));
    // Typed-array / DataView data pointer (see GetOp). Emit it so the node gets a
    // local; GetOp rematerializes on every use to stay GC-current.
    case MDefinition::Opcode::ArrayBufferViewElements:
      return GetOp(e, be, ins);
    // Typed-array length: a PrivateValue (raw size_t) in fixed slot LENGTH_SLOT;
    // the wasm32 value is the low 32 bits at lengthOffset() -- a plain i32 load.
    // Result IntPtr (== i32). gbemu reads `memory.length` in its hot fns.
    case MDefinition::Opcode::ArrayBufferViewLength: {
      if (!GetOp(e, be, ins->toArrayBufferViewLength()->object())) return false;
      return e.writeOp(Op::I32Load) && e.writeVarU32(2) &&
             e.writeVarU32(uint32_t(js::ArrayBufferViewObject::lengthOffset()));
    }
    // Sign-extend an 8/16-bit int held in an i32 to full 32-bit (e.g. `(x<<24)>>24`
    // patterns / Int8Array reads that Warp lowers to MSignExtendInt32).
    case MDefinition::Opcode::SignExtendInt32: {
      if (!GetOp(e, be, ins->getOperand(0))) return false;
      return ins->toSignExtendInt32()->mode() == jit::MSignExtendInt32::Byte
                 ? e.writeOp(Op::I32Extend8S)
                 : e.writeOp(Op::I32Extend16S);
    }
    // Typed-array element load: addr = data + index * byteSize(storageType), then a
    // sized scalar load. Mirrors Ion visitLoadUnboxedScalar. gbemu's GameBoy memory
    // is Uint8Array/Int8Array -> I32Load8U/S; this kept its hot fns in PBL before.
    case MDefinition::Opcode::LoadUnboxedScalar: {
      auto* l = ins->toLoadUnboxedScalar();
      Scalar::Type st = l->storageType();
      int32_t il = be.local(l->index());
      if (il < 0) return false;
      uint32_t esz = uint32_t(Scalar::byteSize(st));
      if (!GetOp(e, be, l->elements()) || !GetLocal(e, uint32_t(il)) ||
          !e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(esz)) ||
          !e.writeOp(Op::I32Mul) || !e.writeOp(Op::I32Add))
        return false;  // addr on stack
      MIRType rty = ins->type();
      switch (st) {
        case Scalar::Int8:
          return e.writeOp(Op::I32Load8S) && e.writeVarU32(0) && e.writeVarU32(0);
        case Scalar::Uint8:
        case Scalar::Uint8Clamped:
          return e.writeOp(Op::I32Load8U) && e.writeVarU32(0) && e.writeVarU32(0);
        case Scalar::Int16:
          return e.writeOp(Op::I32Load16S) && e.writeVarU32(1) && e.writeVarU32(0);
        case Scalar::Uint16:
          return e.writeOp(Op::I32Load16U) && e.writeVarU32(1) && e.writeVarU32(0);
        case Scalar::Int32:
          return e.writeOp(Op::I32Load) && e.writeVarU32(2) && e.writeVarU32(0);
        case Scalar::Uint32:
          // Result is Int32 (raw bits, may be negative) or Double (unsigned value).
          if (!e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0))
            return false;
          if (rty == MIRType::Double)
            return e.writeOp(Op::F64ConvertI32U);  // unsigned -> double
          return true;                              // Int32 raw bits
        case Scalar::Float32:
          if (!e.writeOp(Op::F32Load) || !e.writeVarU32(2) || !e.writeVarU32(0))
            return false;
          return e.writeOp(Op::F64PromoteF32);  // -> f64 local (Double/Float32)
        case Scalar::Float64:
          return e.writeOp(Op::F64Load) && e.writeVarU32(3) && e.writeVarU32(0);
        default:
          return WJBAIL("LoadUnboxedScalar storage type unsupported\n");
      }
    }
    case MDefinition::Opcode::LoadElementHole: {
      // arr[i] where i may be a hole or past initializedLength (Ion
      // visitLoadElementHole). Result is a boxed Value: in-bounds non-hole -> the
      // element; in-bounds hole OR OOB -> undefined; OOB with a negative index ->
      // deopt (needsNegativeIntCheck). Mirrors the OOB-safe LoadElement path below.
      // crypto-sha1 core_sha1 reads x[i+j] where str2binb leaves sparse holes.
      // DEFAULT-ON (GECKO_WJ_NOLEHOLE reverts to bailing): the elusive core_sha1 OOB
      // was a separate GuardNoDenseElements negative-memarg bug (now fixed), not this.
      if (WJGETENV("GECKO_WJ_NOLEHOLE")) return false;
      MDefinition* elemsD = ins->getOperand(0);
      MDefinition* idxD = ins->getOperand(1);
      int32_t il = be.local(idxD);
      if (il < 0) return false;
      bool negCheck = ins->toLoadElementHole()->needsNegativeIntCheck();
      int64_t undefBits = int64_t(JS::UndefinedValue().asRawBits());
      // inBounds = (uint32)index < (uint32)initializedLength(elements). GetOp(elemsD)
      // rematerializes obj->elements_ fresh on each use (GetOpInner never caches an
      // Elements pointer), so it is always GC-current -- no manual refetch needed.
      if (!GetLocal(e, uint32_t(il)) || !GetOp(e, be, elemsD) ||
          !e.writeOp(Op::I32Const) ||
          !e.writeVarS32(js::ObjectElements::offsetOfInitializedLength()) ||
          !e.writeOp(Op::I32Add) || !e.writeOp(Op::I32Load) || !e.writeVarU32(2) ||
          !e.writeVarU32(0) || !e.writeOp(Op::I32LtU))
        return false;
      if (!e.writeOp(Op::If) || !e.writeFixedU8(uint8_t(TypeCode::I64)))
        return false;
      // in-bounds: load boxed elem; map the hole magic to undefined.
      if (!GetOp(e, be, elemsD) || !GetLocal(e, uint32_t(il)) ||
          !e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(sizeof(JS::Value))) ||
          !e.writeOp(Op::I32Mul) || !e.writeOp(Op::I32Add) ||
          !e.writeOp(Op::I64Load) || !e.writeVarU32(3) || !e.writeVarU32(0) ||
          !e.writeOp(Op::LocalTee) || !e.writeVarU32(be.unboxScratch))
        return false;
      if (!e.writeOp(Op::I64Const) ||
          !e.writeVarS64(int64_t(JS::MagicValue(JS_ELEMENTS_HOLE).asRawBits())) ||
          !e.writeOp(Op::I64Eq))
        return false;
      if (!e.writeOp(Op::If) || !e.writeFixedU8(uint8_t(TypeCode::I64)))
        return false;
      if (!e.writeOp(Op::I64Const) || !e.writeVarS64(undefBits)) return false;
      if (!e.writeOp(Op::Else)) return false;
      if (!GetLocal(e, be.unboxScratch)) return false;
      if (!e.writeOp(Op::End)) return false;  // boxed-or-undefined
      if (!e.writeOp(Op::Else)) return false;
      // OOB: Ion deopts on a negative index (needsNegativeIntCheck); else undefined.
      if (negCheck) {
        if (!GetLocal(e, uint32_t(il)) || !e.writeOp(Op::I32Const) ||
            !e.writeVarS32(0) || !e.writeOp(Op::I32LtS))
          return false;
        if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
        if (!EmitDeopt(e, be)) return false;
        if (!e.writeOp(Op::End)) return false;
      }
      if (!e.writeOp(Op::I64Const) || !e.writeVarS64(undefBits)) return false;  // OOB
      return e.writeOp(Op::End);
    }
    case MDefinition::Opcode::LoadElement:
    case MDefinition::Opcode::LoadElementAndUnbox: {
      bool andUnbox = ins->op() == MDefinition::Opcode::LoadElementAndUnbox;
      MDefinition* elemsD = ins->getOperand(0);
      MDefinition* idxD = ins->getOperand(1);
      bool holeCheck = andUnbox ? false : ins->toLoadElement()->needsHoleCheck();
      int32_t il = be.local(idxD);
      if (il < 0) return false;
      // OOB-SAFE self-bounded load: the index's BoundsCheck was deopt-skipped
      // (GECKO_WJ_OOBLOAD). Bound here and return undefined on OOB (and on an
      // in-bounds hole) -- MLoadElementHole semantics, no deopt.
      if (!andUnbox && ins->type() == MIRType::Value &&
          idxD->isBoundsCheck() && WJBoundsCheckOOBSafe(idxD)) {
        int64_t undefBits = int64_t(JS::UndefinedValue().asRawBits());
        // inBounds = (uint32)index < (uint32)initializedLength(elements)
        if (!GetLocal(e, uint32_t(il)) || !GetOp(e, be, elemsD) ||
            !e.writeOp(Op::I32Const) ||
            !e.writeVarS32(js::ObjectElements::offsetOfInitializedLength()) ||
            !e.writeOp(Op::I32Add) || !e.writeOp(Op::I32Load) || !e.writeVarU32(2) ||
            !e.writeVarU32(0) || !e.writeOp(Op::I32LtU))
          return false;
        if (!e.writeOp(Op::If) || !e.writeFixedU8(uint8_t(TypeCode::I64)))
          return false;
        // in-bounds: load boxed elem; map the hole magic to undefined too.
        if (!GetOp(e, be, elemsD) || !GetLocal(e, uint32_t(il)) ||
            !e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(sizeof(JS::Value))) ||
            !e.writeOp(Op::I32Mul) || !e.writeOp(Op::I32Add) ||
            !e.writeOp(Op::I64Load) || !e.writeVarU32(3) || !e.writeVarU32(0) ||
            !e.writeOp(Op::LocalTee) || !e.writeVarU32(be.unboxScratch))
          return false;  // [boxed], unboxScratch=boxed
        if (!e.writeOp(Op::I64Const) ||
            !e.writeVarS64(int64_t(JS::MagicValue(JS_ELEMENTS_HOLE).asRawBits())) ||
            !e.writeOp(Op::I64Eq))
          return false;  // [boxed, isHole]
        if (!e.writeOp(Op::If) || !e.writeFixedU8(uint8_t(TypeCode::I64)))
          return false;
        if (!e.writeOp(Op::I64Const) || !e.writeVarS64(undefBits)) return false;
        if (!e.writeOp(Op::Else)) return false;
        if (!GetLocal(e, be.unboxScratch)) return false;
        if (!e.writeOp(Op::End)) return false;  // boxed-or-undefined
        if (!e.writeOp(Op::Else)) return false;
        if (!e.writeOp(Op::I64Const) || !e.writeVarS64(undefBits)) return false;  // OOB
        return e.writeOp(Op::End);
      }
      // OOB-SAFE Int32 LoadElementAndUnbox (gbemu RAM bytes): OOB index -> 0
      // (== ToInt32(undefined)); in-bounds keeps the fallible unbox (deopt only on a
      // genuinely non-int32 in-bounds value, rare). Eliminates the 357K OOB-index
      // BoundsCheck deopt-storm-into-PBL.
      if (andUnbox && ins->type() == MIRType::Int32 && idxD->isBoundsCheck() &&
          WJBoundsCheckOOBSafe(idxD)) {
        if (!GetLocal(e, uint32_t(il)) || !GetOp(e, be, elemsD) ||
            !e.writeOp(Op::I32Const) ||
            !e.writeVarS32(js::ObjectElements::offsetOfInitializedLength()) ||
            !e.writeOp(Op::I32Add) || !e.writeOp(Op::I32Load) || !e.writeVarU32(2) ||
            !e.writeVarU32(0) || !e.writeOp(Op::I32LtU))
          return false;
        if (!e.writeOp(Op::If) || !e.writeFixedU8(uint8_t(TypeCode::I32)))
          return false;
        // in-bounds: load boxed -> unboxScratch -> fallible Int32 unbox.
        if (!GetOp(e, be, elemsD) || !GetLocal(e, uint32_t(il)) ||
            !e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(sizeof(JS::Value))) ||
            !e.writeOp(Op::I32Mul) || !e.writeOp(Op::I32Add) ||
            !e.writeOp(Op::I64Load) || !e.writeVarU32(3) || !e.writeVarU32(0) ||
            !e.writeOp(Op::LocalSet) || !e.writeVarU32(be.unboxScratch))
          return false;
        if (!EmitUnboxLocal(e, be, be.unboxScratch, MIRType::Int32, /*fallible=*/true))
          return false;
        if (!e.writeOp(Op::Else)) return false;
        if (!e.writeOp(Op::I32Const) || !e.writeVarS32(0)) return false;  // OOB -> 0
        return e.writeOp(Op::End);
      }
      // addr = elements + index * sizeof(Value); load the boxed element.
      if (!GetOp(e, be, elemsD) || !GetLocal(e, uint32_t(il)) ||
          !e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(sizeof(JS::Value))) ||
          !e.writeOp(Op::I32Mul) || !e.writeOp(Op::I32Add)) {
        return false;
      }
      if (!e.writeOp(Op::I64Load) || !e.writeVarU32(3) || !e.writeVarU32(0))
        return false;
      MIRType ty = ins->type();
      static int noHoleDeopt = WJGETENV("GECKO_WJ_NOHOLEDEOPT") ? 1 : 0;
      if (noHoleDeopt) holeCheck = false;  // PROBE: skip hole-check deopt
      if (!holeCheck && ty == MIRType::Value) return true;  // boxed Value on stack
      if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(be.unboxScratch)) return false;
      if (holeCheck) {
        // Deopt if the element is a hole (MagicValue(JS_ELEMENTS_HOLE)).
        if (!GetLocal(e, be.unboxScratch) || !e.writeOp(Op::I64Const) ||
            !e.writeVarS64(int64_t(JS::MagicValue(JS_ELEMENTS_HOLE).asRawBits())) ||
            !e.writeOp(Op::I64Eq)) {
          return false;
        }
        if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
        if (!EmitDeopt(e, be)) return false;
        if (!e.writeOp(Op::End)) return false;
      }
      if (ty == MIRType::Value) return GetLocal(e, be.unboxScratch);
      return EmitUnboxLocal(e, be, be.unboxScratch, ty, /*fallible=*/true);
    }
    case MDefinition::Opcode::Slots: {
      int32_t objLocal = be.local(ins->toSlots()->object());
      if (objLocal < 0) return false;
      uint32_t slotsOff = uint32_t(js::NativeObject::offsetOfSlots());
      if (!GetLocal(e, uint32_t(objLocal))) return false;
      return e.writeOp(Op::I32Load) && e.writeVarU32(2) && e.writeVarU32(slotsOff);
    }
    case MDefinition::Opcode::LoadFixedSlot: {
      MLoadFixedSlot* l = ins->toLoadFixedSlot();
      // Megamorphic recompile / shape-hybrid: this read is guarded by a mega-
      // convertible GuardShape (now a passthrough). Re-derive the property key and
      // emit a baked multi-shape EmitPropIC (bare shape-compare fast path + helper
      // fallback, NO deopt) on the real receiver -- a stale/poly/dictionary shape
      // never deopts. The PropIC self-guards, so losing the GuardShape is sound.
      if ((be.forceMega || WJShapeHybrid()) && WJIsShapeGuard(l->object()) &&
          WJMegaConvertibleGuard(l->object(), /*allowStores=*/be.forceMega, /*stormProven=*/be.forceMega)) {
        // Guard removed -> read MUST convert by-name or BAIL (a fixed-slot read
        // under a removed guard is a correctness bug for a polymorphic receiver).
        jit::MDefinition* gs = l->object();
        uint64_t keyBits = 0;
        if (WJDerivePropKey(WJGuardRepShape(gs), uint32_t(l->slot()), &keyBits)) {
          uint32_t site = WJPropSite(be);
          if (site != 0 && WJGuardObject(gs)->type() == MIRType::Object) {
            if (!EmitPropIC(e, be, ins, WJGuardObject(gs), keyBits, site,
                            WJGuardRepShape(gs),
                            uint32_t(js::NativeObject::getFixedSlotOffset(l->slot())),
                            /*bakedFixed=*/true))
              return false;
            return EmitHelperResultAsType(e, be, l->type(), /*guardTag=*/true);
          }
        }
        return WJBAIL("mega read couldn't convert (guard removed)\n");
      }
      uint32_t off = uint32_t(js::NativeObject::getFixedSlotOffset(l->slot()));
      if (WJGETENV("GECKO_WJ_SLOTDBG2"))
        fprintf(stderr, "[wb-slot] LoadFixedSlot slot=%zu off=%u objOp=%s type=%d\n",
                l->slot(), off, WJOpName(l->object()->op()), int(l->type()));
      // Read the object via GetOp (rematerializes Unbox/GuardShape from the rooted
      // source) -- a cached object local goes stale across a GC-ing call.
      if (!GetOp(e, be, l->object())) return false;
      if (!e.writeOp(Op::I64Load) || !e.writeVarU32(3) || !e.writeVarU32(off))
        return false;
      // Usually the result is a boxed Value (i64 local). But Warp can TYPE a
      // LoadFixedSlot (e.g. an Object base feeding a nested LoadFixedSlotAndUnbox,
      // as in navier's Field accessors): then the node's local is i32/f64, so the
      // raw i64 box must be converted (infallible -- Warp proved the type). Without
      // this the host rejects the module (local.set i32 <- i64.load).
      if (l->type() == MIRType::Value) return true;
      if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(be.unboxScratch)) return false;
      return EmitUnboxLocal(e, be, be.unboxScratch, l->type(), /*fallible=*/false);
    }
    case MDefinition::Opcode::LoadFixedSlotAndUnbox: {
      MLoadFixedSlotAndUnbox* l = ins->toLoadFixedSlotAndUnbox();
      // Megamorphic recompile / shape-hybrid (see LoadFixedSlot): self-guarding load.
      if ((be.forceMega || WJShapeHybrid()) && WJIsShapeGuard(l->object()) &&
          WJMegaConvertibleGuard(l->object(), /*allowStores=*/be.forceMega, /*stormProven=*/be.forceMega)) {
        // Guard removed -> read MUST convert by-name or BAIL (a fixed-slot read
        // under a removed guard is a correctness bug for a polymorphic receiver).
        jit::MDefinition* gs = l->object();
        uint64_t keyBits = 0;
        if (WJDerivePropKey(WJGuardRepShape(gs), uint32_t(l->slot()), &keyBits)) {
          uint32_t site = WJPropSite(be);
          if (site != 0 && WJGuardObject(gs)->type() == MIRType::Object) {
            if (!EmitPropIC(e, be, ins, WJGuardObject(gs), keyBits, site,
                            WJGuardRepShape(gs),
                            uint32_t(js::NativeObject::getFixedSlotOffset(l->slot())),
                            /*bakedFixed=*/true))
              return false;
            return EmitHelperResultAsType(e, be, l->type(), /*guardTag=*/true);
          }
        }
        return WJBAIL("mega read couldn't convert (guard removed)\n");
      }
      uint32_t off = uint32_t(js::NativeObject::getFixedSlotOffset(l->slot()));
      if (WJGETENV("GECKO_WJ_SLOTDBG2"))
        fprintf(stderr, "[wb-slot] LoadFixedSlotAndUnbox slot=%zu off=%u objOp=%s type=%d\n",
                l->slot(), off, WJOpName(l->object()->op()), int(l->type()));
      if (!GetOp(e, be, l->object())) return false;  // rematerialized object
      if (!e.writeOp(Op::I64Load) || !e.writeVarU32(3) || !e.writeVarU32(off))
        return false;
      if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(be.unboxScratch))
        return false;
      return EmitUnboxLocal(e, be, be.unboxScratch, l->type(),
                            WJSlotFallible(l->type(), l->mode()));
    }
    case MDefinition::Opcode::LoadDynamicSlotAndUnbox: {
      MLoadDynamicSlotAndUnbox* l = ins->toLoadDynamicSlotAndUnbox();
      if (be.forceMega || WJShapeHybrid()) {
        if (jit::MDefinition* gs = WJMegaSlotsGuard(l->slots(), /*allowStores=*/be.forceMega, /*stormProven=*/be.forceMega)) {
          // The shape guard WILL be passthrough'd (removed -- WJMegaConvertibleGuard
          // matched). So this read MUST become a self-guarding by-name EmitPropIC;
          // a bare fixed-slot load would read the WRONG slot for a polymorphic
          // receiver that no longer hits the (removed) guard. If we can't convert
          // (no derivable key / site pool exhausted), BAIL the whole function --
          // emitting the fixed-slot load here is a CORRECTNESS bug (deltablue 741
          // "Cycle": late valve-recompile exhausted the prop-site pool, guard
          // removed but read stayed fixed-slot).
          uint32_t nf = WJGuardRepShape(gs)->asShared().numFixedSlots();
          uint64_t keyBits = 0;
          if (WJDerivePropKey(WJGuardRepShape(gs), nf + uint32_t(l->slot()), &keyBits) &&
              WJGuardObject(gs)->type() == MIRType::Object) {
            uint32_t site = WJPropSite(be);
            if (site != 0) {
              if (!EmitPropIC(e, be, ins, WJGuardObject(gs), keyBits, site,
                              WJGuardRepShape(gs),
                              uint32_t(l->slot() * sizeof(JS::Value)),
                              /*bakedFixed=*/false))
                return false;
              return EmitHelperResultAsType(e, be, l->type(), /*guardTag=*/true);
            }
          }
          return WJBAIL("mega read couldn't convert (guard removed)\n");
        }
      }
      uint32_t off = uint32_t(l->slot() * sizeof(JS::Value));
      if (!GetOp(e, be, l->slots())) return false;
      if (!e.writeOp(Op::I64Load) || !e.writeVarU32(3) || !e.writeVarU32(off))
        return false;
      if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(be.unboxScratch))
        return false;
      return EmitUnboxLocal(e, be, be.unboxScratch, l->type(),
                            WJSlotFallible(l->type(), l->mode()));
    }
    case MDefinition::Opcode::LoadDynamicSlot: {
      MLoadDynamicSlot* l = ins->toLoadDynamicSlot();
      if (be.forceMega || WJShapeHybrid()) {
        if (jit::MDefinition* gs = WJMegaSlotsGuard(l->slots(), /*allowStores=*/be.forceMega, /*stormProven=*/be.forceMega)) {
          // Guard removed -> read MUST convert by-name or BAIL (see
          // LoadDynamicSlotAndUnbox: a fixed-slot read under a removed guard is a
          // correctness bug on a polymorphic receiver).
          uint32_t nf = WJGuardRepShape(gs)->asShared().numFixedSlots();
          uint64_t keyBits = 0;
          if (WJDerivePropKey(WJGuardRepShape(gs), nf + uint32_t(l->slot()), &keyBits) &&
              WJGuardObject(gs)->type() == MIRType::Object) {
            uint32_t site = WJPropSite(be);
            if (site != 0) {
              if (!EmitPropIC(e, be, ins, WJGuardObject(gs), keyBits, site,
                              WJGuardRepShape(gs),
                              uint32_t(l->slot() * sizeof(JS::Value)),
                              /*bakedFixed=*/false))
                return false;
              return EmitHelperResultAsType(e, be, l->type(), /*guardTag=*/true);
            }
          }
          return WJBAIL("mega read couldn't convert (guard removed)\n");
        }
      }
      uint32_t off = uint32_t(l->slot() * sizeof(JS::Value));
      if (WJGETENV("GECKO_WJ_SLOTDBG2"))
        fprintf(stderr, "[wb-slot] LoadDynamicSlot slot=%u off=%u slotsOp=%s\n",
                uint32_t(l->slot()), off, WJOpName(l->slots()->op()));
      if (!GetOp(e, be, l->slots())) return false;
      if (!e.writeOp(Op::I64Load) || !e.writeVarU32(3) || !e.writeVarU32(off))
        return false;
      // As LoadFixedSlot: convert when Warp typed the slot (else i64 box stays).
      if (l->type() == MIRType::Value) return true;
      if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(be.unboxScratch)) return false;
      return EmitUnboxLocal(e, be, be.unboxScratch, l->type(), /*fallible=*/false);
    }
    case MDefinition::Opcode::LoadDynamicSlotFromOffset: {
      // result = loadValue(slots + offset). offset is a runtime byte offset
      // (Ion: BaseIndex(slots, offset, TimesOne)). Result is a boxed Value.
      MLoadDynamicSlotFromOffset* l = ins->toLoadDynamicSlotFromOffset();
      // Megamorphic recompile: the offset comes from a GuardShapeListToOffset that
      // DEOPTS on an unlisted shape (deltablue 414 storm). Convert to a no-deopt
      // multi-shape EmitPropIC on the real receiver (helper fallback for any shape).
      if (be.forceMega) {
        jit::MDefinition* obj = nullptr;
        uint64_t keyBits = 0;
        if (WJMegaGSLTO(l->offset(), &obj, &keyBits) &&
            obj->type() == MIRType::Object) {
          uint32_t site = WJPropSite(be);
          if (site != 0) {
            if (!EmitPropIC(e, be, ins, obj, keyBits, site)) return false;
            return EmitHelperResultAsType(e, be, l->type(), /*guardTag=*/true);
          }
          // The GuardShapeListToOffset may have been converted to a DUMMY offset (0)
          // -- then GetOp(offset) reads slots+0 (the WRONG slot). Can't convert (no
          // site) -> BAIL rather than read the wrong slot. Conservative (also bails
          // the rare case the GSLTO kept its real offset), but always correct.
          return WJBAIL("mega GSLTO read couldn't convert\n");
        }
      }
      if (!GetOp(e, be, l->slots())) return false;       // i32 slots ptr
      if (!GetOp(e, be, l->offset())) return false;       // i32 byte offset
      if (!e.writeOp(Op::I32Add)) return false;
      return e.writeOp(Op::I64Load) && e.writeVarU32(3) && e.writeVarU32(0);
    }
    case MDefinition::Opcode::LoadFixedSlotFromOffset: {
      // result = loadValue(object + offset). Fixed slots are inline in the
      // object, so the base is the object pointer (not a separate slots buffer).
      MLoadFixedSlotFromOffset* l = ins->toLoadFixedSlotFromOffset();
      if (!GetOp(e, be, l->object())) return false;       // i32 object ptr
      if (!GetOp(e, be, l->offset())) return false;        // i32 byte offset
      if (!e.writeOp(Op::I32Add)) return false;
      if (!e.writeOp(Op::I64Load) || !e.writeVarU32(3) || !e.writeVarU32(0))
        return false;
      if (l->type() == MIRType::Value) return true;
      if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(be.unboxScratch)) return false;
      return EmitUnboxLocal(e, be, be.unboxScratch, l->type(), /*fallible=*/false);
    }
    case MDefinition::Opcode::StrictConstantCompareInt32: {
      // value === k (or !==): a strict int32 compare on a boxed Value. The value
      // may be stored int32-boxed OR double-boxed, so compare the raw i64 bits
      // against both representations of k (and -0.0 when k==0), mirroring Ion's
      // visitStrictConstantCompareInt32. Result is an i32 boolean.
      static int noScc = WJGETENV("GECKO_WJ_NOSCC") ? 1 : 0;
      if (noScc) return false;
      if (WJGETENV("GECKO_WJ_NOSCC")) return false;  // diagnostic gate (bisect only)
      MStrictConstantCompareInt32* c = ins->toStrictConstantCompareInt32();
      bool eq = c->jsop() == JSOp::StrictEq;
      int32_t k = c->constant();
      uint64_t intBits = JS::Int32Value(k).asRawBits();
      uint64_t dblBits = JS::DoubleValue(double(k)).asRawBits();
      auto cmp = [&](uint64_t bits) -> bool {
        return GetLocal(e, be.unboxScratch) && e.writeOp(Op::I64Const) &&
               e.writeVarS64(int64_t(bits)) &&
               e.writeOp(eq ? Op::I64Eq : Op::I64Ne);
      };
      if (!GetOp(e, be, c->value())) return false;  // boxed i64 value
      if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(be.unboxScratch)) return false;
      if (!cmp(intBits)) return false;
      if (!cmp(dblBits)) return false;
      if (!e.writeOp(eq ? Op::I32Or : Op::I32And)) return false;
      if (k == 0) {
        if (!cmp(JS::DoubleValue(-0.0).asRawBits())) return false;
        if (!e.writeOp(eq ? Op::I32Or : Op::I32And)) return false;
      }
      return true;
    }
    case MDefinition::Opcode::ToDouble: {
      // ToDouble result is a FULL Double (f64). Int32 -> f64 EXACTLY
      // (f64.convert_i32_s). The old code rounded through f32 (F32ConvertI32S ->
      // F64PromoteF32), which LOSES precision for |int32| > 2^24 -- e.g. crypto's
      // bignum limbs up to 2^28 got truncated to f32's 24-bit mantissa -> wrong
      // double -> wrong reduction -> ~50% wrong crypto checksum (the long-standing
      // "deopt-resume" heisenbug was actually THIS: a number-typed variant of the
      // bignum/RC4 fns uses ToDouble, the int32 variant doesn't). Float32 input is
      // held as a (already-f32-rounded) f64 -> no-op; Double -> no-op.
      MDefinition* in = ins->getOperand(0);
      if (!GetOp(e, be, in)) return false;
      if (in->type() == MIRType::Int32) return e.writeOp(Op::F64ConvertI32S);
      if (in->type() == MIRType::Double || in->type() == MIRType::Float32)
        return true;  // already an f64 local
      return false;
    }
    case MDefinition::Opcode::ToFloat32: {
      // Result is Float32, held as f64 but ROUNDED to f32 precision (value -> f32
      // -> f64), matching JS Math.fround / Float32 store semantics.
      MDefinition* in = ins->getOperand(0);
      if (!GetOp(e, be, in)) return false;
      if (in->type() == MIRType::Int32) {
        if (!e.writeOp(Op::F32ConvertI32S)) return false;
      } else if (in->type() == MIRType::Double ||
                 in->type() == MIRType::Float32) {
        if (!e.writeOp(Op::F32DemoteF64)) return false;
      } else {
        return false;
      }
      return e.writeOp(Op::F64PromoteF32);  // back to the f64 local
    }
    case MDefinition::Opcode::Add:
    case MDefinition::Opcode::Sub:
    case MDefinition::Opcode::Mul: {
      bool d = ins->type() == MIRType::Double;
      bool i = ins->type() == MIRType::Int32;
      if (d) {
        if (!GetOp(e, be, ins->getOperand(0)) ||
            !GetOp(e, be, ins->getOperand(1)))
          return false;
        return e.writeOp(ins->isAdd()   ? Op::F64Add
                         : ins->isSub() ? Op::F64Sub
                                        : Op::F64Mul);
      }
      if (ins->type() == MIRType::Float32) {
        // Float32 Add/Sub/Mul: operands held as f64 (exact f32 values). f32{+,-,*}f32
        // is EXACT in f64 (the result mantissa fits in 52 bits), so do the f64 op then
        // round the result to f32 precision (demote+promote) = the exact IEEE-f32 /
        // Math.fround result. (Div is NOT sound this way -- double-rounding -- and is
        // not in this Add/Sub/Mul case.) Operands must be f64-repr; else bail clean.
        if (WJValType(ins->getOperand(0)->type()) != uint8_t(TypeCode::F64) ||
            WJValType(ins->getOperand(1)->type()) != uint8_t(TypeCode::F64))
          return false;
        if (!GetOp(e, be, ins->getOperand(0)) ||
            !GetOp(e, be, ins->getOperand(1)))
          return false;
        if (!e.writeOp(ins->isAdd()   ? Op::F64Add
                       : ins->isSub() ? Op::F64Sub
                                      : Op::F64Mul))
          return false;
        return e.writeOp(Op::F32DemoteF64) && e.writeOp(Op::F64PromoteF32);
      }
      if (!i) return false;
      // Operands MUST be i32-repr in wasm. An Int32-result op can have a non-i32
      // operand (a Value/Int64 local) -- emitting i32.add/sub/mul on an i64
      // local.get is INVALID wasm and V8 rejects the WHOLE module (gbemu
      // executeIteration: `i32.sub expected i32, found local.get i64` -> the entire
      // 100KB function fell to PBL). Guard the operand repr; if either isn't i32,
      // bail this function cleanly to PBL (correct) instead of poisoning the module.
      if (WJValType(ins->getOperand(0)->type()) != uint8_t(TypeCode::I32) ||
          WJValType(ins->getOperand(1)->type()) != uint8_t(TypeCode::I32)) {
        if (WJGETENV("GECKO_WJ_ARITHDBG"))
          fprintf(stderr, "[wj-arith-badop] %s op0=%s:%s op1=%s:%s\n",
                  WJOpName(ins->op()), WJOpName(ins->getOperand(0)->op()),
                  StringFromMIRType(ins->getOperand(0)->type()),
                  WJOpName(ins->getOperand(1)->op()),
                  StringFromMIRType(ins->getOperand(1)->type()));
        js::wasm::gWJBailReason = "int-arith-nonI32-operand";
        return false;
      }
      // int32 +/-/* overflow MUST bail to a double (Ion does): a non-truncated
      // int32 op that overflows 2^31 and then feeds a shift/compare/sum (crypto's
      // bignum carry `c=(l>>28)+...`, hash-map's keySum accumulator 4.05e9) gets a
      // WRONG result from a wrapping i32.add (negative wrap). Compute in i64
      // (sign-extended operands), deopt if the result doesn't fit int32, else wrap.
      //
      // DEFAULT (2026-06-28): emit the overflow check ONLY for FALLIBLE ops, exactly
      // as Ion does. MAdd/MSub/MMul::fallible() is false when range analysis proved
      // int32 bounds or the op is truncated -- those CANNOT overflow / want wrapping,
      // so a plain i32 op is correct AND fast (loop counters, indices, `x|0` chains).
      // Only genuinely-fallible ops pay the i64 check, so the suite-wide perf cost is
      // minimal while wrong-answers (the old wrapping default was latently wrong on
      // every overflowing non-truncated op) are eliminated. The earlier blanket-on
      // version's crypto ERRs were a separate (now-fixed) resume bug; gating on
      // fallible() shrinks the checked set to where it matters.
      //   GECKO_WJ_NOOVFLCHECK -> old wrapping-always default (no check anywhere).
      //   GECKO_WJ_OVFLCHECK   -> force the check on ALL int ops (debug/A-B).
      bool fallible;
      if (ins->isAdd())      fallible = ins->toAdd()->fallible();
      else if (ins->isSub()) fallible = ins->toSub()->fallible();
      else                   fallible = ins->toMul()->fallible();
      // NEGATIVE ZERO (Mul only): int32 `a*b` producing 0 is -0 (a double, not +0) when
      // exactly one operand is negative -- e.g. `-x` = `x*-1` gives -0 for x==0, so
      // `1/(-0)` = -Infinity, Object.is(-0,0)=false. Ion bails via MMul::canBeNegativeZero;
      // the JIT emitted a plain i32.mul -> +0 (silent wrong result: -0 lost). Add the
      // deopt below, and force such a Mul OFF the plain fast path. (Int32 Add/Sub never
      // produce -0: 0+0=+0, 0-0=+0.) canBeNegativeZero is false where Warp proved -0
      // irrelevant (result feeds bitwise/compare/sum), so hot loops pay nothing.
      bool negZero = ins->isMul() && ins->toMul()->canBeNegativeZero();
      static int noOvfl  = WJGETENV("GECKO_WJ_NOOVFLCHECK") ? 1 : 0;
      static int allOvfl = WJGETENV("GECKO_WJ_OVFLCHECK") ? 1 : 0;
      if (!negZero && (noOvfl || (!fallible && !allOvfl))) {
        if (!GetOp(e, be, ins->getOperand(0)) ||
            !GetOp(e, be, ins->getOperand(1)))
          return false;
        return e.writeOp(ins->isAdd()   ? Op::I32Add
                         : ins->isSub() ? Op::I32Sub
                                        : Op::I32Mul);
      }
      // i64 sum/diff/prod of sign-extended operands.
      if (!GetOp(e, be, ins->getOperand(0)) ||
          !e.writeOp(Op::I64ExtendI32S) ||
          !GetOp(e, be, ins->getOperand(1)) || !e.writeOp(Op::I64ExtendI32S))
        return false;
      if (!e.writeOp(ins->isAdd()   ? Op::I64Add
                     : ins->isSub() ? Op::I64Sub
                                    : Op::I64Mul))
        return false;
      if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(be.unboxScratch)) return false;
      // overflow if sext(wrap(s)) != s -> deopt (PBL redoes it as a double)
      if (!GetLocal(e, be.unboxScratch) || !e.writeOp(Op::I32WrapI64) ||
          !e.writeOp(Op::I64ExtendI32S) || !GetLocal(e, be.unboxScratch) ||
          !e.writeOp(Op::I64Ne))
        return false;
      if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
      if (!EmitDeopt(e, be)) return false;
      if (!e.writeOp(Op::End)) return false;
      // Negative-zero deopt: result==0 && (lhs<0 || rhs<0) -> the product is -0 (double).
      if (negZero) {
        if (!GetLocal(e, be.unboxScratch) || !e.writeOp(Op::I32WrapI64) ||
            !e.writeOp(Op::I32Eqz))
          return false;  // result == 0
        if (!GetOp(e, be, ins->getOperand(0)) || !e.writeOp(Op::I32Const) ||
            !e.writeVarS32(0) || !e.writeOp(Op::I32LtS))
          return false;  // lhs < 0
        if (!GetOp(e, be, ins->getOperand(1)) || !e.writeOp(Op::I32Const) ||
            !e.writeVarS32(0) || !e.writeOp(Op::I32LtS))
          return false;  // rhs < 0
        if (!e.writeOp(Op::I32Or) || !e.writeOp(Op::I32And)) return false;
        if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
        if (!EmitDeopt(e, be)) return false;
        if (!e.writeOp(Op::End)) return false;
      }
      return GetLocal(e, be.unboxScratch) && e.writeOp(Op::I32WrapI64);
    }
    case MDefinition::Opcode::Div: {
      if (ins->type() == MIRType::Double) {
        if (!GetOp(e, be, ins->getOperand(0)) ||
            !GetOp(e, be, ins->getOperand(1))) {
          return false;
        }
        return e.writeOp(Op::F64Div);
      }
      if (ins->type() == MIRType::Int32) {
        // Integer division. wasm i32.div_s traps on /0 and INT32_MIN/-1, and a
        // non-exact or negative-zero result must be a double -- deopt in all
        // those cases (matching Ion's MDiv bailouts) so we never produce a wrong
        // int32 or trap.
        MDiv* d = ins->toDiv();
        int32_t ll = be.local(d->lhs()), rl = be.local(d->rhs());
        if (ll < 0 || rl < 0) return false;
        auto gl = [&](int32_t l) { return GetLocal(e, uint32_t(l)); };
        auto deoptIf = [&]() -> bool {
          return e.writeOp(Op::If) && e.writeFixedU8(0x40) && EmitDeopt(e, be) &&
                 e.writeOp(Op::End);
        };
        // rhs == 0 -> deopt
        if (!gl(rl) || !e.writeOp(Op::I32Eqz) || !deoptIf()) return false;
        // lhs == INT32_MIN && rhs == -1 -> deopt (overflow)
        if (!gl(ll) || !e.writeOp(Op::I32Const) || !e.writeVarS32(INT32_MIN) ||
            !e.writeOp(Op::I32Eq) || !gl(rl) || !e.writeOp(Op::I32Const) ||
            !e.writeVarS32(-1) || !e.writeOp(Op::I32Eq) || !e.writeOp(Op::I32And) ||
            !deoptIf())
          return false;
        // negative zero: lhs == 0 && rhs < 0 -> result is -0.0, must be double
        if (d->canBeNegativeZero()) {
          if (!gl(ll) || !e.writeOp(Op::I32Eqz) || !gl(rl) ||
              !e.writeOp(Op::I32Const) || !e.writeVarS32(0) ||
              !e.writeOp(Op::I32LtS) || !e.writeOp(Op::I32And) || !deoptIf())
            return false;
        }
        // non-exact (true quotient not integral) -> must be double
        if (!d->isTruncated()) {
          if (!gl(ll) || !gl(rl) || !e.writeOp(Op::I32RemS) ||
              !e.writeOp(Op::I32Const) || !e.writeVarS32(0) ||
              !e.writeOp(Op::I32Ne) || !deoptIf())
            return false;
        }
        return gl(ll) && gl(rl) && e.writeOp(Op::I32DivS);
      }
      return false;
    }
    case MDefinition::Opcode::Mod: {
      if (ins->type() == MIRType::Double) {
        // JS `a % b` on doubles == C fmod (result has sign of `a`, |result|<|b|).
        // wasm has no f64.rem, but fmod(a,b) == a - b*trunc(a/b) exactly (handles
        // inf/0/NaN edge cases identically: trunc(NaN)=NaN propagates). Compute
        // a/b once into a temp would need a local; recompute is fine (GetOp is
        // cheap for locals/consts).
        MDefinition* a = ins->getOperand(0);
        MDefinition* b = ins->getOperand(1);
        if (!GetOp(e, be, a)) return false;                 // a
        if (!GetOp(e, be, b)) return false;                 // a b
        if (!GetOp(e, be, a) || !GetOp(e, be, b)) return false;  // a b a b
        if (!e.writeOp(Op::F64Div)) return false;           // a b (a/b)
        if (!e.writeOp(Op::F64Trunc)) return false;         // a b trunc(a/b)
        if (!e.writeOp(Op::F64Mul)) return false;           // a (b*trunc)
        if (!e.writeOp(Op::F64Sub)) return false;           // a - b*trunc
        // fmod's result carries the DIVIDEND's sign -- including ±0 (fmod(-1,1)
        // = -0), which the subtraction formula loses (x - x = +0). copysign is
        // an identity for every correct nonzero result and fixes the zero case
        // (jit-test cacheir/binaryarith-mod-int32 via the Double-typed Mod).
        return GetOp(e, be, a) && e.writeOp(Op::F64CopySign);
      }
      if (ins->type() == MIRType::Int32) {
        MMod* m = ins->toMod();
        int32_t ll = be.local(m->lhs()), rl = be.local(m->rhs());
        if (ll < 0 || rl < 0) return false;
        auto gl = [&](int32_t l) { return GetLocal(e, uint32_t(l)); };
        auto deoptIf = [&]() -> bool {
          return e.writeOp(Op::If) && e.writeFixedU8(0x40) && EmitDeopt(e, be) &&
                 e.writeOp(Op::End);
        };
        // rhs == 0 -> NaN result (double) -> deopt.
        if (!gl(rl) || !e.writeOp(Op::I32Eqz) || !deoptIf()) return false;
        // Negative-zero: a negative dividend with a zero remainder yields -0.0,
        // which must be a double -> deopt (unless the result is truncated to int).
        // i32.rem_s(INT32_MIN,-1)==0 in wasm (no trap), and that case has lhs<0 so
        // it's covered here too.
        if (!m->isTruncated() && m->canBeNegativeDividend()) {
          if (!gl(ll) || !e.writeOp(Op::I32Const) || !e.writeVarS32(0) ||
              !e.writeOp(Op::I32LtS) ||                       // lhs < 0
              !gl(ll) || !gl(rl) || !e.writeOp(Op::I32RemS) ||
              !e.writeOp(Op::I32Eqz) ||                       // rem == 0
              !e.writeOp(Op::I32And) || !deoptIf())
            return false;
        }
        return gl(ll) && gl(rl) && e.writeOp(Op::I32RemS);
      }
      return false;
    }
    case MDefinition::Opcode::Pow: {
      // Math.pow / `**`. No wasm pow instruction -> route through WJH_BINARYARITH
      // (which does the *Values VM op for JSOp::Pow == js::ecmaPow). Operands are
      // staged as boxed Values; result converted to the node's type (Double).
      // (raytrace rayTrace's specular Math.pow(dot, shininess) bailed here.)
      if (!EmitStageScratch(e, be, ins->getOperand(0), 0)) return false;
      if (!EmitStageScratch(e, be, ins->getOperand(1), 1)) return false;
      if (!EmitHelperCallResult(e, be, ins, WJH_BINARYARITH, uint32_t(JSOp::Pow)))
        return false;
      // An Int32-typed Pow is SPECULATIVE: Warp saw only small NON-NEGATIVE integer
      // exponents in warmup, but the helper's true result can be fractional
      // (2**-1 = 0.5), overflow int32 (2**31 = 2147483648), or NaN -- all DOUBLE-tagged.
      // Without guardTag the Int32 coercion blindly wraps the double box's low 32 bits to
      // garbage (2**-1 -> 0, 2**31 -> 0, 3**-1 -> 1431655765). guardTag DEOPTS to PBL on a
      // non-int32 box (PBL produces the correct double). Double-typed Pow (raytrace's
      // specular Math.pow) is unaffected -- guardTag only applies to Int32/Boolean. Same
      // speculative-int32-edge class as [[ursh-negative-int32-fix]]/[[negzero-mul-fix]].
      return EmitHelperResultAsType(e, be, ins->type(), /*guardTag=*/true);
    }
    case MDefinition::Opcode::PowHalf: {
      // Math.pow(x, 0.5) == sqrt(x), EXCEPT pow(-Infinity,0.5)==+Infinity (sqrt is
      // NaN) and pow(-0,0.5)==+0 (sqrt is -0). Mirrors Ion visitPowHalf:
      // (x == -Inf) ? +Inf : sqrt(x + 0.0)   [the +0.0 maps -0 -> +0].
      const uint8_t kF64bt = 0x7C;
      double negInf = -std::numeric_limits<double>::infinity();
      double posInf = std::numeric_limits<double>::infinity();
      if (!GetOp(e, be, ins->getOperand(0)) || !e.writeOp(Op::F64Const) ||
          !e.writeFixedF64(negInf) || !e.writeOp(Op::F64Eq))
        return false;
      if (!e.writeOp(Op::If) || !e.writeFixedU8(kF64bt)) return false;
      if (!e.writeOp(Op::F64Const) || !e.writeFixedF64(posInf)) return false;
      if (!e.writeOp(Op::Else)) return false;
      if (!GetOp(e, be, ins->getOperand(0)) || !e.writeOp(Op::F64Const) ||
          !e.writeFixedF64(0.0) || !e.writeOp(Op::F64Add) || !e.writeOp(Op::F64Sqrt))
        return false;
      return e.writeOp(Op::End);
    }
    case MDefinition::Opcode::Sqrt: {
      if (ins->type() != MIRType::Double) return false;
      if (!GetOp(e, be, ins->getOperand(0))) return false;
      return e.writeOp(Op::F64Sqrt);
    }
    case MDefinition::Opcode::MathFunction: {
      // Math.sin/cos/tan/log/exp/acos/asin/atan/log2/log10/cbrt/sinh/.../floor/ceil/
      // round (Double). wasm has no transcendentals -> call the SAME fdlibm/native fn
      // Ion uses (GetUnaryMathFunctionPtr), via WJH_MATHFN with the function id as site.
      if (WJGETENV("GECKO_WJ_NO_MATHFN")) return false;
      MMathFunction* m = ins->toMathFunction();
      if (!EmitStageScratch(e, be, m->input(), 0)) return false;
      if (!EmitHelperCallResult(e, be, ins, WJH_MATHFN, uint32_t(m->function())))
        return false;
      return EmitHelperResultAsType(e, be, ins->type());  // Double
    }
    case MDefinition::Opcode::TypedArrayElementSize: {
      // typedArray's bytesPerElement (1/2/4/8). Per-type constant; cold -> helper
      // (avoids the inline clasp-table pointer arithmetic Ion uses).
      if (!EmitStageScratch(e, be, ins->getOperand(0), 0)) return false;  // typed array
      if (!EmitHelperCallResult(e, be, ins, WJH_TYPEDARRELEMSIZE, 0)) return false;
      return EmitHelperResultAsType(e, be, jit::MIRType::Int32);
    }
    case MDefinition::Opcode::NewTypedArray: {
      // new TypedArray(constLen) / new TypedArray() -- FIXED-length ctor: the length is
      // a compile-time constant baked in the template (FixedLengthTypedArrayObject).
      // Reuse the DynamicLength helper (js::NewTypedArrayWithTemplateAndLength) by
      // staging the constant length as a boxed Int32. Common on real sites (canvas/
      // WebGL/binary/crypto -> MDN surfaced this bail).
      MNewTypedArray* m = ins->toNewTypedArray();
      int32_t n = int32_t(m->templateObject()->length());
      if (!EmitStageConstBoxed(e, be, JS::Int32Value(n).asRawBits(), 0)) return false;  // length
      uint64_t tBits = JS::ObjectValue(*m->templateObject()).asRawBits();
      if (!EmitStageConstBoxed(e, be, tBits, 1)) return false;  // template (traced)
      if (!EmitHelperCallResult(e, be, ins, WJH_NEWTYPEDARRDYN, 0)) return false;
      return EmitHelperResultAsType(e, be, jit::MIRType::Object);
    }
    case MDefinition::Opcode::NewTypedArrayDynamicLength: {
      // new TypedArray(length). Alloc helper (js::NewTypedArrayWithTemplateAndLength).
      // Unlike NADL, typed arrays aren't built via deep recursion (cdjs's recursion
      // is plain arrays), so default-on; revert-to-bail if a recursive
      // typed-array fn ever trips the V8-stack landmine.
      MNewTypedArrayDynamicLength* m = ins->toNewTypedArrayDynamicLength();
      if (!EmitStageScratch(e, be, m->length(), 0)) return false;  // length (Int32)
      uint64_t tBits = JS::ObjectValue(*m->templateObject()).asRawBits();
      if (!EmitStageConstBoxed(e, be, tBits, 1)) return false;  // template (traced)
      if (!EmitHelperCallResult(e, be, ins, WJH_NEWTYPEDARRDYN, 0)) return false;
      return EmitHelperResultAsType(e, be, jit::MIRType::Object);
    }
    case MDefinition::Opcode::NewTypedArrayFromArrayBuffer: {
      // new TypedArray(buffer, byteOffset, length) with the baked template.
      // VM call js::NewTypedArrayWithTemplateAndBuffer (byteOffset/length are
      // boxed Values, may be undefined).
      MNewTypedArrayFromArrayBuffer* m = ins->toNewTypedArrayFromArrayBuffer();
      if (!EmitStageScratch(e, be, m->arrayBuffer(), 0)) return false;  // buffer (Object)
      if (!EmitStageScratch(e, be, m->byteOffset(), 1)) return false;   // byteOffset (Value)
      if (!EmitStageScratch(e, be, m->length(), 2)) return false;       // length (Value)
      uint64_t tBits = JS::ObjectValue(*m->templateObject()).asRawBits();
      if (!EmitStageConstBoxed(e, be, tBits, 3)) return false;          // template (traced)
      if (!EmitHelperCallResult(e, be, ins, WJH_NEWTYPEDARRFROMBUF, 0)) return false;
      return EmitHelperResultAsType(e, be, jit::MIRType::Object);
    }
    case MDefinition::Opcode::RegExpHasCaptureGroups: {
      // re.hasCaptureGroups against input. VM fn (parses RegExpShared) -> helper.
      if (!EmitStageScratch(e, be, ins->getOperand(0), 0)) return false;  // regexp
      if (!EmitStageScratch(e, be, ins->getOperand(1), 1)) return false;  // input
      if (!EmitHelperCallResult(e, be, ins, WJH_REGEXPHASCAPS, 0)) return false;
      return EmitHelperResultAsType(e, be, jit::MIRType::Boolean);
    }
    case MDefinition::Opcode::RegExpMatcher: {
      // RegExp op family (pre-
      // 2026-07-02 behavior) to isolate whether these ops cause a failure.
      // re.exec-style match -> match array (Value) or null. Same VM fallback Ion
      // uses (nullptr matches). Keeps regexp-heavy real-site fns (acorn/marked)
      // in JIT instead of a whole-function bail.
      if (!EmitStageScratch(e, be, ins->getOperand(0), 0)) return false;  // regexp (Object)
      if (!EmitStageScratch(e, be, ins->getOperand(1), 1)) return false;  // string (String)
      if (!EmitStageScratch(e, be, ins->getOperand(2), 2)) return false;  // lastIndex (Int32)
      if (!EmitHelperCallResult(e, be, ins, WJH_REGEXPMATCHER, 0)) return false;
      return EmitHelperResultAsType(e, be, jit::MIRType::Value);
    }
    case MDefinition::Opcode::RegExpSearcher: {
      // String.prototype.search fast path -> match start (Int32) or -1. Also sets
      // cx->regExpSearcherLastLimit, consumed by an immediately-following
      // RegExpSearcherLastLimit (MIR order guarantees no intervening regexp op).
      if (!EmitStageScratch(e, be, ins->getOperand(0), 0)) return false;  // regexp (Object)
      if (!EmitStageScratch(e, be, ins->getOperand(1), 1)) return false;  // string (String)
      if (!EmitStageScratch(e, be, ins->getOperand(2), 2)) return false;  // lastIndex (Int32)
      if (!EmitHelperCallResult(e, be, ins, WJH_REGEXPSEARCHER, 0)) return false;
      return EmitHelperResultAsType(e, be, jit::MIRType::Int32);
    }
    case MDefinition::Opcode::RegExpSearcherLastLimit: {
      // Reads cx->regExpSearcherLastLimit (set by the preceding RegExpSearcher).
      if (!EmitHelperCallResult(e, be, ins, WJH_REGEXPSEARCHERLASTLIMIT, 0)) return false;
      return EmitHelperResultAsType(e, be, jit::MIRType::Int32);
    }
    case MDefinition::Opcode::RegExpExecMatch: {
      // RegExp.prototype.exec fast path -> match array (Value) or null.
      if (!EmitStageScratch(e, be, ins->getOperand(0), 0)) return false;  // regexp (RegExpObject)
      if (!EmitStageScratch(e, be, ins->getOperand(1), 1)) return false;  // string (String)
      if (!EmitHelperCallResult(e, be, ins, WJH_REGEXPEXECMATCH, 0)) return false;
      return EmitHelperResultAsType(e, be, jit::MIRType::Value);
    }
    case MDefinition::Opcode::RegExpExecTest: {
      // RegExp.prototype.test fast path -> Boolean.
      if (!EmitStageScratch(e, be, ins->getOperand(0), 0)) return false;  // regexp (RegExpObject)
      if (!EmitStageScratch(e, be, ins->getOperand(1), 1)) return false;  // string (String)
      if (!EmitHelperCallResult(e, be, ins, WJH_REGEXPEXECTEST, 0)) return false;
      return EmitHelperResultAsType(e, be, jit::MIRType::Boolean);
    }
    case MDefinition::Opcode::GetFirstDollarIndex: {
      // String.prototype.replace substitution: first '$' index or -1.
      if (!EmitStageScratch(e, be, ins->getOperand(0), 0)) return false;  // string (String)
      if (!EmitHelperCallResult(e, be, ins, WJH_GETFIRSTDOLLARINDEX, 0)) return false;
      return EmitHelperResultAsType(e, be, jit::MIRType::Int32);
    }
    case MDefinition::Opcode::DeleteProperty: {
      // delete obj.name. Cold -> helper (js::DelPropOperation<strict>). The name
      // (PropertyName*) is baked as a StringValue via the traced const pool.
      MDeleteProperty* d = ins->toDeleteProperty();
      if (!EmitStageScratch(e, be, d->value(), 0)) return false;  // object (Value)
      uint64_t nameBits = JS::StringValue(d->name()).asRawBits();
      if (!EmitStageConstBoxed(e, be, nameBits, 1)) return false;  // name (traced)
      if (!EmitHelperCallResult(e, be, ins, WJH_DELPROP, d->strict() ? 1u : 0u))
        return false;
      return EmitHelperResultAsType(e, be, jit::MIRType::Boolean);
    }
    case MDefinition::Opcode::Random: {
      // Math.random() -> Double in [0,1). The realm's XorShift128+ RNG. Helper
      // (RNG state read/advance); not typically hot in these workloads.
      if (!EmitHelperCallResult(e, be, ins, WJH_RANDOM, 0)) return false;
      return EmitHelperResultAsType(e, be, jit::MIRType::Double);
    }
    case MDefinition::Opcode::IsConstructor: {
      // IsConstructor(obj). Leaf inspection (bound-fn/proxy/class-ctor logic) ->
      // helper (jit::ObjectIsConstructor); not hot.
      if (!EmitStageScratch(e, be, ins->getOperand(0), 0)) return false;  // object
      if (!EmitHelperCallResult(e, be, ins, WJH_ISCONSTRUCTOR, 0)) return false;
      return EmitHelperResultAsType(e, be, jit::MIRType::Boolean);
    }
    case MDefinition::Opcode::ObjectKeys: {
      // Object.keys(obj). Cold + uninlinable (enumerate own keys + alloc array) -> helper.
      if (!EmitStageScratch(e, be, ins->getOperand(0), 0)) return false;  // object
      if (!EmitHelperCallResult(e, be, ins, WJH_OBJECTKEYS, 0)) return false;
      return EmitHelperResultAsType(e, be, jit::MIRType::Object);
    }
    case MDefinition::Opcode::DeleteElement: {
      // delete obj[index]. Cold -> helper (js::DelElemOperation<strict>).
      MDeleteElement* d = ins->toDeleteElement();
      if (!EmitStageScratch(e, be, d->value(), 0)) return false;  // object (Value)
      if (!EmitStageScratch(e, be, d->index(), 1)) return false;  // index (Value)
      if (!EmitHelperCallResult(e, be, ins, WJH_DELELEM, d->strict() ? 1u : 0u))
        return false;
      return EmitHelperResultAsType(e, be, jit::MIRType::Boolean);
    }
    case MDefinition::Opcode::StringReplace: {
      // str.replace(pattern, repl) for string pattern/repl. Native string op,
      // uninlinable + cold -> minimal helper (jit::StringReplace / flat variant).
      MStringReplace* m = ins->toStringReplace();
      if (!EmitStageScratch(e, be, m->string(), 0)) return false;
      if (!EmitStageScratch(e, be, m->pattern(), 1)) return false;
      if (!EmitStageScratch(e, be, m->replacement(), 2)) return false;
      if (!EmitHelperCallResult(e, be, ins, WJH_STRINGREPLACE,
                                m->isFlatReplacement() ? 1u : 0u))
        return false;
      return EmitHelperResultAsType(e, be, jit::MIRType::String);
    }
    case MDefinition::Opcode::StringSplit: {
      // str.split(sep). Native op + array alloc, cold -> minimal helper.
      if (!EmitStageScratch(e, be, ins->getOperand(0), 0)) return false;  // string
      if (!EmitStageScratch(e, be, ins->getOperand(1), 1)) return false;  // separator
      if (!EmitHelperCallResult(e, be, ins, WJH_STRINGSPLIT, 0)) return false;
      return EmitHelperResultAsType(e, be, jit::MIRType::Object);
    }
    case MDefinition::Opcode::GuardStringToInt32: {
      // Speculation guard: string -> exact Int32, else deopt. Uses the exact VM fn
      // Ion's fallback uses (jit::GetInt32FromStringPure, pure/no-GC). Minimal helper.
      if (!EmitStageScratch(e, be, ins->getOperand(0), 0)) return false;  // string
      if (!EmitHelperCallResult(e, be, ins, WJH_STRINGTONUMBER, 0)) return false;
      return EmitHelperResultAsType(e, be, jit::MIRType::Int32);
    }
    case MDefinition::Opcode::GuardStringToDouble: {
      // Speculation guard: string -> Double, else deopt (js::StringToNumberPure).
      if (!EmitStageScratch(e, be, ins->getOperand(0), 0)) return false;  // string
      if (!EmitHelperCallResult(e, be, ins, WJH_STRINGTONUMBER, 1)) return false;
      return EmitHelperResultAsType(e, be, jit::MIRType::Double);
    }
    case MDefinition::Opcode::StringTrimStartIndex: {
      // String.prototype.trim() helper: first non-whitespace index. Leaf C++ scan
      // (jit::StringTrimStartIndex, no GC) -> minimal helper. Removes the bail.
      if (!EmitStageScratch(e, be, ins->getOperand(0), 0)) return false;  // string
      if (!EmitHelperCallResult(e, be, ins, WJH_STRINGTRIMINDEX, 0)) return false;
      return EmitHelperResultAsType(e, be, jit::MIRType::Int32);
    }
    case MDefinition::Opcode::StringTrimEndIndex: {
      // trim() end index from a start offset. Leaf jit::StringTrimEndIndex(str,start).
      if (!EmitStageScratch(e, be, ins->getOperand(0), 0)) return false;  // string
      if (!EmitStageScratch(e, be, ins->getOperand(1), 1)) return false;  // start Int32
      if (!EmitHelperCallResult(e, be, ins, WJH_STRINGTRIMINDEX, 1)) return false;
      return EmitHelperResultAsType(e, be, jit::MIRType::Int32);
    }
    case MDefinition::Opcode::Int32ToStringWithBase: {
      // n.toString(base) for an Int32 n. Cold + genuinely uninlinable in wasm
      // (digit conversion + string allocation) -> minimal helper calling
      // js::Int32ToStringWithBase<CanGC>. Removes the whole-function bail.
      MInt32ToStringWithBase* m = ins->toInt32ToStringWithBase();
      if (!EmitStageScratch(e, be, m->input(), 0)) return false;
      if (!EmitStageScratch(e, be, m->base(), 1)) return false;
      uint32_t lower = (m->stringCase() == jit::StringCase::Lower) ? 1u : 0u;
      if (!EmitHelperCallResult(e, be, ins, WJH_INT32TOSTRINGBASE, lower))
        return false;
      return EmitHelperResultAsType(e, be, ins->type());  // String
    }
    case MDefinition::Opcode::LimitedTruncate: {
      // Truncation BARRIER (MLimitedTruncate): a movable no-op that protects its
      // input from follow-up truncation optimizations. Operand is Int32 (its
      // ConvertToInt32Policy converts if needed) and result is Int32 -- pure
      // passthrough at codegen. Mirrors Ion lowering it to its input register.
      return GetOp(e, be, ins->getOperand(0));
    }
    case MDefinition::Opcode::Abs: {
      MAbs* a = ins->toAbs();
      MDefinition* in = a->input();
      if (ins->type() == MIRType::Double) {
        // Math.abs(double/float32) -> F64Abs (operands held as f64; exact).
        if (in->type() != MIRType::Double && in->type() != MIRType::Float32)
          return false;
        if (!GetOp(e, be, in)) return false;
        return e.writeOp(Op::F64Abs);
      }
      // Int32 abs: branchless (x ^ (x>>31)) - (x>>31). abs(INT32_MIN) overflows
      // int32 -> when fallible() deopt on INT32_MIN (matches visitAbsI's bailout);
      // when !fallible() the op is implicit-truncate so the wrap is intended.
      if (ins->type() == MIRType::Int32 && in->type() == MIRType::Int32 &&
          !WJGETENV("GECKO_WJ_NO_ABSI32")) {
        if (a->fallible()) {
          if (!GetOp(e, be, in)) return false;
          if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(0x80000000)) ||
              !e.writeOp(Op::I32Eq))
            return false;
          if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
          if (!EmitDeopt(e, be)) return false;
          if (!e.writeOp(Op::End)) return false;
        }
        if (!GetOp(e, be, in)) return false;  // x
        if (!GetOp(e, be, in) || !e.writeOp(Op::I32Const) || !e.writeVarS32(31) ||
            !e.writeOp(Op::I32ShrS))
          return false;  // x, t=(x>>31)
        if (!e.writeOp(Op::I32Xor)) return false;  // x^t
        if (!GetOp(e, be, in) || !e.writeOp(Op::I32Const) || !e.writeVarS32(31) ||
            !e.writeOp(Op::I32ShrS))
          return false;  // x^t, t
        return e.writeOp(Op::I32Sub);  // (x^t) - t = abs(x)
      }
      return false;
    }
    case MDefinition::Opcode::Floor:
    case MDefinition::Opcode::Ceil:
    case MDefinition::Opcode::Trunc: {
      // Math.floor/ceil/trunc(double) -> int32 (Warp-typed Int32). Apply the
      // f64 rounding, then deopt unless the (integral) result is exactly
      // representable as int32 -- mirrors Ion's MFloor/MCeil/MTrunc, which bail
      // on out-of-range/NaN. The deopt MUST sit at nesting==1 (its own `if`).
      MDefinition* in = ins->getOperand(0);
      if (ins->type() != MIRType::Int32) return false;
      if (in->type() != MIRType::Double && in->type() != MIRType::Float32) {
        if (in->type() == MIRType::Int32) return GetOp(e, be, in);  // already int
        return false;
      }
      Op round = ins->isFloor()  ? Op::F64Floor
                 : ins->isCeil() ? Op::F64Ceil
                                 : Op::F64Trunc;
      auto pushRounded = [&]() -> bool {
        return GetOp(e, be, in) && e.writeOp(round);
      };
      // deopt unless (rounded >= -2147483648.0 && rounded <= 2147483647.0).
      // NaN makes both comparisons false -> deopts (correct).
      if (!pushRounded() || !e.writeOp(Op::F64Const) ||
          !e.writeFixedF64(-2147483648.0) || !e.writeOp(Op::F64Ge))
        return false;
      if (!pushRounded() || !e.writeOp(Op::F64Const) ||
          !e.writeFixedF64(2147483647.0) || !e.writeOp(Op::F64Le) ||
          !e.writeOp(Op::I32And) || !e.writeOp(Op::I32Eqz))
        return false;  // !(inRange)
      if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
      if (!EmitDeopt(e, be)) return false;
      if (!e.writeOp(Op::End)) return false;
      // In range: convert the integral rounded value to int32 (exact, no trap).
      return pushRounded() && e.writeOp(Op::I32TruncF64S);
    }
    case MDefinition::Opcode::MinMax: {
      MMinMax* m = ins->toMinMax();
      if (!GetOp(e, be, m->lhs()) || !GetOp(e, be, m->rhs())) return false;
      if (m->type() == MIRType::Double) {
        return e.writeOp(m->isMax() ? Op::F64Max : Op::F64Min);
      }
      if (m->type() == MIRType::Int32) {
        // i32 has no min/max: select on a signed compare. Stack: [a, b].
        if (!GetOp(e, be, m->lhs()) || !GetOp(e, be, m->rhs())) return false;
        if (!e.writeOp(m->isMax() ? Op::I32GtS : Op::I32LtS)) return false;
        return e.writeOp(Op::SelectNumeric);  // a (gt/lt) b ? a : b
      }
      return false;
    }
    case MDefinition::Opcode::Not: {
      MDefinition* in = ins->toNot()->input();
      MIRType t = in->type();
      if (t == MIRType::Boolean || t == MIRType::Int32) {
        return GetOp(e, be, in) && e.writeOp(Op::I32Eqz);
      }
      if (t == MIRType::Object) {  // non-null object is always truthy
        return e.writeOp(Op::I32Const) && e.writeVarS32(0);
      }
      if (t == MIRType::Null || t == MIRType::Undefined) {
        return e.writeOp(Op::I32Const) && e.writeVarS32(1);
      }
      if (t == MIRType::Double) {  // !d == (d == 0 || isNaN(d))
        if (!GetOp(e, be, in) || !e.writeOp(Op::F64Const) ||
            !e.writeFixedF64(0.0) || !e.writeOp(Op::F64Ne))
          return false;  // d != 0
        if (!GetOp(e, be, in) || !GetOp(e, be, in) || !e.writeOp(Op::F64Eq))
          return false;  // d == d (false for NaN)
        return e.writeOp(Op::I32And) && e.writeOp(Op::I32Eqz);
      }
      if (t == MIRType::Value) {
        int32_t l = be.local(in);
        if (l < 0) return false;
        return EmitValueTruthy(e, be, uint32_t(l)) && e.writeOp(Op::I32Eqz);
      }
      return false;
    }
    case MDefinition::Opcode::StringLength: {
      int32_t sl = be.local(ins->toStringLength()->string());
      if (sl < 0) return false;
      if (!GetLocal(e, uint32_t(sl))) return false;
      return e.writeOp(Op::I32Load) && e.writeVarU32(2) &&
             e.writeVarU32(uint32_t(JSString::offsetOfLength()));
    }
    case MDefinition::Opcode::IsObject: {
      // result = (typeof v is object). For a boxed Value: tag == OBJECT.
      MDefinition* v = ins->getOperand(0);
      MIRType t = v->type();
      if (t == MIRType::Object) return e.writeOp(Op::I32Const) && e.writeVarS32(1);
      if (t != MIRType::Value)
        return e.writeOp(Op::I32Const) && e.writeVarS32(0);  // any other typed prim
      if (!GetOp(e, be, v) || !e.writeOp(Op::I64Const) || !e.writeVarS64(32) ||
          !e.writeOp(Op::I64ShrU) || !e.writeOp(Op::I32WrapI64) ||
          !e.writeOp(Op::I32Const) ||
          !e.writeVarS32(int32_t(TagWord(JSVAL_TYPE_OBJECT))) ||
          !e.writeOp(Op::I32Eq))
        return false;
      return true;
    }
    case MDefinition::Opcode::IsNullOrUndefined: {
      MDefinition* v = ins->getOperand(0);
      MIRType t = v->type();
      if (t == MIRType::Null || t == MIRType::Undefined)
        return e.writeOp(Op::I32Const) && e.writeVarS32(1);
      if (t != MIRType::Value)
        return e.writeOp(Op::I32Const) && e.writeVarS32(0);
      // (tag == NULL) | (tag == UNDEFINED)
      auto tag = [&]() -> bool {
        return GetOp(e, be, v) && e.writeOp(Op::I64Const) && e.writeVarS64(32) &&
               e.writeOp(Op::I64ShrU) && e.writeOp(Op::I32WrapI64);
      };
      if (!tag() || !e.writeOp(Op::I32Const) ||
          !e.writeVarS32(int32_t(TagWord(JSVAL_TYPE_NULL))) || !e.writeOp(Op::I32Eq))
        return false;
      if (!tag() || !e.writeOp(Op::I32Const) ||
          !e.writeVarS32(int32_t(TagWord(JSVAL_TYPE_UNDEFINED))) ||
          !e.writeOp(Op::I32Eq) || !e.writeOp(Op::I32Or))
        return false;
      return true;
    }
    case MDefinition::Opcode::CheckObjCoercible: {
      // Passthrough the value; THROW (deopt to PBL, which re-runs + throws the proper
      // TypeError) if it is null/undefined. Result type is Value (boxed i64).
      MDefinition* v = ins->getOperand(0);
      if (v->type() != MIRType::Value) return false;  // typed-operand cases rare; bail
      int32_t l = be.local(v);
      if (l < 0) return false;
      auto tag = [&]() -> bool {
        return GetLocal(e, uint32_t(l)) && e.writeOp(Op::I64Const) &&
               e.writeVarS64(32) && e.writeOp(Op::I64ShrU) && e.writeOp(Op::I32WrapI64);
      };
      if (!tag() || !e.writeOp(Op::I32Const) ||
          !e.writeVarS32(int32_t(TagWord(JSVAL_TYPE_NULL))) || !e.writeOp(Op::I32Eq))
        return false;
      if (!tag() || !e.writeOp(Op::I32Const) ||
          !e.writeVarS32(int32_t(TagWord(JSVAL_TYPE_UNDEFINED))) ||
          !e.writeOp(Op::I32Eq) || !e.writeOp(Op::I32Or))
        return false;
      if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
      if (!EmitDeopt(e, be)) return false;
      if (!e.writeOp(Op::End)) return false;
      return GetLocal(e, uint32_t(l));  // passthrough (i64)
    }
    case MDefinition::Opcode::LexicalCheck: {
      // Passthrough; THROW (deopt -> PBL throws ReferenceError) on the TDZ sentinel
      // MagicValue(JS_UNINITIALIZED_LEXICAL). Result type Value.
      MDefinition* v = ins->getOperand(0);
      if (v->type() != MIRType::Value) return false;
      int32_t l = be.local(v);
      if (l < 0) return false;
      int64_t magic = int64_t(JS::MagicValue(JS_UNINITIALIZED_LEXICAL).asRawBits());
      if (!GetLocal(e, uint32_t(l)) || !e.writeOp(Op::I64Const) ||
          !e.writeVarS64(magic) || !e.writeOp(Op::I64Eq))
        return false;
      if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
      if (!EmitDeopt(e, be)) return false;
      if (!e.writeOp(Op::End)) return false;
      return GetLocal(e, uint32_t(l));  // passthrough (i64)
    }
    case MDefinition::Opcode::HasClass: {
      // result = (obj->clasp == class_). obj is MIRType::Object (i32 ptr).
      MDefinition* obj = ins->getOperand(0);
      if (obj->type() != MIRType::Object) return false;
      const JSClass* clasp = ins->toHasClass()->getClass();
      uint32_t shapeOff = uint32_t(offsetof(JS::shadow::Object, shape));
      if (!GetOp(e, be, obj) || !e.writeOp(Op::I32Load) || !e.writeVarU32(2) ||
          !e.writeVarU32(shapeOff) || !e.writeOp(Op::I32Load) || !e.writeVarU32(2) ||
          !e.writeVarU32(uint32_t(js::Shape::offsetOfBaseShape())) ||
          !e.writeOp(Op::I32Load) || !e.writeVarU32(2) ||
          !e.writeVarU32(uint32_t(js::BaseShape::offsetOfClasp())))
        return false;
      return e.writeOp(Op::I32Const) &&
             e.writeVarS32(int32_t(uintptr_t(clasp))) && e.writeOp(Op::I32Eq);
    }
    case MDefinition::Opcode::GuardToClass: {
      // Passthrough obj; deopt if obj->clasp != class_. Result type Object (i32 ptr).
      MDefinition* obj = ins->getOperand(0);
      if (obj->type() != MIRType::Object) return false;
      const JSClass* clasp = ins->toGuardToClass()->getClass();
      uint32_t shapeOff = uint32_t(offsetof(JS::shadow::Object, shape));
      if (!GetOp(e, be, obj) || !e.writeOp(Op::I32Load) || !e.writeVarU32(2) ||
          !e.writeVarU32(shapeOff) || !e.writeOp(Op::I32Load) || !e.writeVarU32(2) ||
          !e.writeVarU32(uint32_t(js::Shape::offsetOfBaseShape())) ||
          !e.writeOp(Op::I32Load) || !e.writeVarU32(2) ||
          !e.writeVarU32(uint32_t(js::BaseShape::offsetOfClasp())))
        return false;
      if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(uintptr_t(clasp))) ||
          !e.writeOp(Op::I32Ne))
        return false;
      if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
      if (!EmitDeopt(e, be)) return false;
      if (!e.writeOp(Op::End)) return false;
      return GetOp(e, be, obj);  // passthrough (i32 ptr)
    }
    case MDefinition::Opcode::CallGetIntrinsicValue: {
      // Self-hosted intrinsic, realm-stable (created once, cached on the global).
      // Resolve at COMPILE time + bake the boxed result into the traced const pool
      // (no helper, no per-call cost -- mirrors Ion folding it to a constant).
      JSContext* cx = js::TlsContext.get();
      if (!cx) return false;
      JS::Rooted<js::GlobalObject*> global(cx, cx->global());
      JS::Rooted<js::PropertyName*> name(cx, ins->toCallGetIntrinsicValue()->name());
      JS::RootedValue val(cx);
      if (!js::GlobalObject::getIntrinsicValue(cx, global, name, &val)) {
        cx->clearPendingException();
        return false;  // unresolved -> bail (rare)
      }
      uintptr_t slot = WJInternConstant(val.get().asRawBits());
      if (!slot) return false;
      return e.writeOp(Op::I32Const) && e.writeVarS32(int32_t(slot)) &&
             e.writeOp(Op::I64Load) && e.writeVarU32(3) && e.writeVarU32(0);
    }
    case MDefinition::Opcode::GetFrameArgument: {
      // Register ABI: actual arg i is wasm param kWJArg0Param+i (boxed Value). All entry
      // paths now stage ALL actual args (fast call_indirect pushes 8; slow WJH_CALL stages
      // argc; C++ WasmJitRunCall stages argc + pads) -- so reading the param is reliable for
      // ANY i<kWJMaxArgs, INCLUDING actual args beyond the formal nargs (self-hosted
      // IteratorReduce's GetArgument(1)=initialValue, nargs=1). The OLD i>=nargs bail was a
      // workaround for WasmJitRunCall staging only nargs (now fixed). GECKO_WJ_GFANARGS
      // restores the old bail for A/B. i>=kWJMaxArgs still bails (no param slot).
      MDefinition* idx = ins->toGetFrameArgument()->index();
      // INLINED frame -> bail: staged params/arg roots belong to the OUTER
      // function's activation, not the inlined callee's call site (see
      // ArgumentsLength above). Applies to BOTH const- and dyn-index paths.
      if (ins->block()->info().script() != be.info->script()) { fprintf(stderr, "[wj-inlargs] GetFrameArgument inlined\n"); return false; }
      // Emit a GC-current read of actual arg i (boxed Value onto the stack). Priority:
      // (1) needsArgRoots -> the traced gWJCallRoots shadow slot (variadic fns); (2) the
      // formal Parameter's rooted copy local; (3) raw param (beyond-formal/unused, no GC
      // hazard for int-ish args). Reading the raw param directly is UNSAFE across a GC
      // that moves an object arg (the copy/slot is relocated, the raw param is not).
      uintptr_t argRootsBase = uintptr_t(static_cast<void*>(&gWJCallRoots[0]));
      uint32_t argRootOff = be.usesCallee ? 2 : 1;  // slot offset from envRootIdx
      auto emitArg = [&](uint32_t i) -> bool {
        if (be.needsArgRoots) {
          return GetLocal(e, be.envRootIdx) && e.writeOp(Op::I32Const) &&
                 e.writeVarS32(int32_t(argRootOff + i)) && e.writeOp(Op::I32Add) &&
                 e.writeOp(Op::I32Const) && e.writeVarS32(8) && e.writeOp(Op::I32Mul) &&
                 e.writeOp(Op::I32Const) && e.writeVarS32(int32_t(argRootsBase)) &&
                 e.writeOp(Op::I32Add) && e.writeOp(Op::I64Load) && e.writeVarU32(3) &&
                 e.writeVarU32(0);
        }
        uint32_t src = be.argCopyLocal[i] >= 0 ? uint32_t(be.argCopyLocal[i])
                                               : (kWJArg0Param + i);
        return e.writeOp(Op::LocalGet) && e.writeVarU32(src);
      };
      if (idx->isConstant() && idx->toConstant()->type() == MIRType::Int32) {
        int32_t i = idx->toConstant()->toInt32();
        if (i < 0 || uint32_t(i) >= kWJMaxArgs) return false;
        if (WJGETENV("GECKO_WJ_GFANARGS") && be.info && uint32_t(i) >= be.info->nargs())
          return false;
        return emitArg(uint32_t(i));
      }
      // Dynamic index. Plain GetFrameArgument (not GetFrameArgumentHole) guarantees
      // 0 <= idx < argc. The function-ENTRY argc>kWJMaxArgs guard (be.needsArgcGuard,
      // set for dynamic GetFrameArgument in the pre-scan -> flag-2 did-not-run for wider
      // calls) guarantees argc<=kWJMaxArgs here, so idx < kWJMaxArgs ALWAYS -> a pure
      // select-cascade over the staged params. No bounds deopt (so no reloop #16
      // exposure), no reloop bail.
      // OPT-IN (GECKO_WJ_DYNFRAMEARG enables; default OFF -> bail to PBL). REVERTED
      // to opt-in 2026-07-03: coffeescript intermittently MISCOMPILES with this on
      // (WTB output-validation fails ~1/3 of runs -> no score; NODYNFRAMEARG was 3/3
      // clean). Same GC-staleness/warmup Heisenbug class as the earley trap this
      // feature already had -- the 43-workload browser sweep didn't include coffee.
      // Gating is a revert-to-known-good: those dynamic-arguments[i] fns bail whole
      // to PBL (CORRECT, minor perf loss on a niche path; octane unaffected). The
      // typescript/uglify frame-arg bails it "fixed" simply return to PBL. Re-enable
      // needs the residual variadic GC-staleness diagnosed (wasm value-stepper).
      // 2026-07-10: the earley/coffeescript traps that forced this opt-in were the
      // PBL TableSwitch resume-base bug (see reloop-bug-notes), fixed. Default-ON;
      // GECKO_WJ_NODYNFRAMEARG reverts. The inReloopBody bail (#16) is gone too.
      if (WJGETENV("GECKO_WJ_NODYNFRAMEARG")) return false;
      if (idx->type() != MIRType::Int32) return false;
      // Bisection: GECKO_WJ_DYNFAMAX=<n> emits the cascade for only the first n
      // compiles reaching here (others bail); GECKO_WJ_DYNFADBG logs each site.
      {
        static const char* dynfaMaxS = getenv("GECKO_WJ_DYNFAMAX");
        static int dynfaMax = dynfaMaxS ? atoi(dynfaMaxS) : -1;
        if (dynfaMax >= 0) {
          static int dynfaN = 0;
          if (dynfaN >= dynfaMax) return false;
          dynfaN++;
        }
        if (WJGETENV("GECKO_WJ_DYNFADBG")) {
          JSScript* s = be.info ? be.info->script() : nullptr;
          fprintf(stderr, "[wj-dynfa] %s:%u\n",
                  s && s->filename() ? s->filename() : "?",
                  s ? unsigned(s->lineno()) : 0);
        }
      }
      if (!GetOp(e, be, idx)) return false;                        // [idx]
      if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(be.dynArgIdx))
        return false;                                              // dynArgIdx=idx
      // acc = arg0 (each arg read GC-current via emitArg -- raw param goes stale on GC).
      if (!emitArg(0) ||
          !e.writeOp(Op::LocalSet) || !e.writeVarU32(be.dynArgAcc))
        return false;                                              // acc = arg0
      for (uint32_t i = 1; i < kWJMaxArgs; i++) {
        if (!emitArg(i)) return false;                             // [argi]
        if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(be.dynArgAcc))
          return false;                                            // [argi, acc]
        if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(be.dynArgIdx) ||
            !e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(i)) ||
            !e.writeOp(Op::I32Eq))
          return false;                                            // [argi, acc, idx==i]
        if (!e.writeOp(Op::SelectNumeric)) return false;           // [idx==i ? argi : acc]
        if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(be.dynArgAcc))
          return false;                                            // acc = select
      }
      return e.writeOp(Op::LocalGet) && e.writeVarU32(be.dynArgAcc);  // [Value]
    }
    case MDefinition::Opcode::StrictConstantCompareBoolean: {
      // `value (===|!==) <constant bool>`: a strict bit compare of the boxed
      // value against the canonical BooleanValue.
      MStrictConstantCompareBoolean* cc =
          ins->toStrictConstantCompareBoolean();
      if (!GetOp(e, be, cc->getOperand(0))) return false;  // boxed value (i64)
      uint64_t bits = JS::BooleanValue(cc->constant()).asRawBits();
      if (!e.writeOp(Op::I64Const) || !e.writeVarS64(int64_t(bits))) return false;
      JSOp jsop = cc->jsop();
      bool ne = (jsop == JSOp::Ne || jsop == JSOp::StrictNe);
      return e.writeOp(ne ? Op::I64Ne : Op::I64Eq);
    }
    case MDefinition::Opcode::BooleanToInt32: {
      // A Boolean is already an i32 0/1 in our unboxed representation, so the
      // Int32 result is the same value -- pass it through. (Absent, this forced
      // hot driver loops with a `cond|0`/`+true` idiom entirely into PBL.) An
      // Int32 input is likewise identity (both no-deopt, sound).
      MDefinition* in = ins->getOperand(0);
      if (in->type() != MIRType::Boolean && in->type() != MIRType::Int32)
        return false;
      return GetOp(e, be, in);
    }
    case MDefinition::Opcode::TruncateToInt32: {
      // JS ToInt32 (`x|0`, bitwise, asm.js): trunc toward zero, then modulo 2^32
      // as a signed int32 (WRAPPING, not clamping). Int32 passes through. For a
      // Double, i64.trunc_sat_f64_s truncates into i64 (saturating only past
      // +-2^63) and i32.wrap_i64 takes the low 32 bits = the modular ToInt32. NaN
      // -> 0. Mirrors Ion's MTruncateToInt32. Does NOT deopt (truncation IS the
      // defined semantics here; deopting was UNSOUND -- re-ran committed stores).
      MDefinition* in = ins->getOperand(0);
      if (in->type() == MIRType::Int32) return GetOp(e, be, in);
      // Emit modular ToInt32 of a double supplied by pushDbl(): `i64.trunc_sat_f64_s`+wrap
      // is the correct modular ToInt32 ONLY for |trunc(x)| < 2^63. For |x| >= 2^63 or
      // +-Infinity it SATURATES in i64 (ToInt32(1e20) -> -1 not 1661992960; ToInt32(Inf)
      // -> -1 not 0), so guard `|x| < 2^63` and route the rare big/Inf case to the correct
      // JS::ToInt32 helper (deopting is UNSOUND here -- re-runs committed typed-array
      // stores). NaN: |NaN|<2^63 is false -> helper, which also gives 0. Result: i32.
      auto emitDblToI32 = [&](auto pushDbl) -> bool {
        constexpr int noBigFix = 0;  // big-double ToInt32 fix is permanent
        if (noBigFix)
          return pushDbl() && e.writeOp(MiscOp::I64TruncSatF64S) &&
                 e.writeOp(Op::I32WrapI64);
        const uint8_t kI32bt2 = 0x7F;
        if (!pushDbl() || !e.writeOp(Op::F64Abs) || !e.writeOp(Op::F64Const) ||
            !e.writeFixedF64(9223372036854775808.0 /*2^63*/) || !e.writeOp(Op::F64Lt))
          return false;
        if (!e.writeOp(Op::If) || !e.writeFixedU8(kI32bt2)) return false;
        if (!pushDbl() || !e.writeOp(MiscOp::I64TruncSatF64S) ||
            !e.writeOp(Op::I32WrapI64))
          return false;
        if (!e.writeOp(Op::Else)) return false;
        if (!EmitStageScratch(e, be, in, 0) ||
            !EmitHelperCallResult(e, be, ins, WJH_TOINT32, 0) ||
            !EmitHelperResultAsType(e, be, MIRType::Int32))
          return false;
        return e.writeOp(Op::End);
      };
      // A boxed Value operand (`x|0` / `& 0xFF` on a boxed number -- gbemu's CPU
      // register/memory ops are pervasively this; executeIteration, the main opcode
      // loop, bailed entirely to PBL for it). Inline number fast path with NO deopt:
      //   int32 tag  -> ToInt32(int32) = the int32 (low 32 bits)
      //   double     -> modular truncate (trunc_sat_f64_s; wrap)
      //   non-number -> WJH_TOINT32 helper (full JS ToInt32; undefined/null->0, etc.)
      // The non-number case routes to a HELPER, not a deopt. NOW DEFAULT-ON: the
      // separate miscompile that previously made enabling this give ERR=Gameboy was
      // the unsound deopt-resume + mega-load blind-wrap, both since fixed (icEntry
      // alignment + EmitHelperResultAsType guardTag). Verified gbemu correct with it
      // on (113->116). GECKO_WJ_NOTRUNCVALUE reverts to bailing (whole fn -> PBL).
      if (in->type() == MIRType::Value && !WJGETENV("GECKO_WJ_NOTRUNCVALUE")) {
        int32_t inLocal = be.local(in);
        if (inLocal < 0) return false;
        const uint8_t kI32bt = 0x7F;  // wasm i32 blocktype
        auto pushTag = [&]() -> bool {
          return GetLocal(e, uint32_t(inLocal)) && e.writeOp(Op::I64Const) &&
                 e.writeVarS64(32) && e.writeOp(Op::I64ShrU) &&
                 e.writeOp(Op::I32WrapI64);
        };
        // tag == INT32 ?  (the common gbemu case: small integer registers)
        if (!pushTag() || !e.writeOp(Op::I32Const) ||
            !e.writeVarS32(int32_t(TagWord(JSVAL_TYPE_INT32))) ||
            !e.writeOp(Op::I32Eq))
          return false;
        if (!e.writeOp(Op::If) || !e.writeFixedU8(kI32bt)) return false;
        if (!GetLocal(e, uint32_t(inLocal)) || !e.writeOp(Op::I32WrapI64))
          return false;  // int32: low 32 bits
        if (!e.writeOp(Op::Else)) return false;
        // tag <= CLEAR ?  (double)
        if (!pushTag() || !e.writeOp(Op::I32Const) ||
            !e.writeVarS32(int32_t(uint32_t(JSVAL_TAG_CLEAR))) ||
            !e.writeOp(Op::I32LeU))
          return false;
        if (!e.writeOp(Op::If) || !e.writeFixedU8(kI32bt)) return false;
        if (!emitDblToI32([&]() -> bool {
              return GetLocal(e, uint32_t(inLocal)) &&
                     e.writeOp(Op::F64ReinterpretI64);
            }))
          return false;  // double: modular trunc (|x|>=2^63/Inf -> ToInt32 helper)
        if (!e.writeOp(Op::Else)) return false;
        // non-number: full ToInt32 via helper (no deopt, no resume)
        if (!EmitStageScratch(e, be, in, 0)) return false;
        if (!EmitHelperCallResult(e, be, ins, WJH_TOINT32, 0)) return false;
        if (!EmitHelperResultAsType(e, be, MIRType::Int32)) return false;
        if (!e.writeOp(Op::End)) return false;  // end double-if
        return e.writeOp(Op::End);              // end int32-if
      }
      // Float32 is held as f64 (see WJValType) -> same wrapping trunc as Double.
      if (in->type() != MIRType::Double && in->type() != MIRType::Float32)
        return false;
      return emitDblToI32([&]() -> bool { return GetOp(e, be, in); });
    }
    case MDefinition::Opcode::ToNumberInt32: {
      // ToNumberInt32 is the SPECULATIVE int conversion Warp inserts when an op
      // is typed Int32 but an operand is a Double (e.g. `x[i] += dt*s[i]` typed
      // Int32 from early all-zero iterations). Unlike TruncateToInt32, it must
      // DEOPT when the double is NOT an exact int32 -- truncating would silently
      // corrupt (dt*s[i]=0.0001 -> 0, navier addFields checksum). Mirrors Ion's
      // convertDoubleToInt32 (bail on non-representable). The deopt then trips the
      // recompile valve, which re-specializes the op as Double. Sound now (lastRp
      // gives the nearest resume point; the only store here is after this op).
      MDefinition* in = ins->getOperand(0);
      if (in->type() == MIRType::Int32) return GetOp(e, be, in);
      if (in->type() != MIRType::Double && in->type() != MIRType::Float32) {
        // Non-number input (Value/String/Boolean/...): the REAL ToInt32(ToNumber(x))
        // (truncating, NOT the speculative deopt below) via JS::ToInt32. Cold ->
        // helper. (The Double/Float32 path below is the speculative int-typing guard.)
        if (!EmitStageScratch(e, be, in, 0)) return false;
        if (!EmitHelperCallResult(e, be, ins, WJH_TONUMBERINT32, 0)) return false;
        return EmitHelperResultAsType(e, be, jit::MIRType::Int32);
      }
      // deopt unless f64.convert_i32_s(i32.trunc_sat_f64_s(d)) == d  (exact int32).
      if (!GetOp(e, be, in) || !e.writeOp(MiscOp::I32TruncSatF64S) ||
          !e.writeOp(Op::F64ConvertI32S) || !GetOp(e, be, in) ||
          !e.writeOp(Op::F64Ne))
        return false;
      if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
      if (!EmitDeopt(e, be)) return false;
      if (!e.writeOp(Op::End)) return false;
      return GetOp(e, be, in) && e.writeOp(MiscOp::I32TruncSatF64S);
    }
    case MDefinition::Opcode::GetPropertyCache: {
      MGetPropertyCache* g = ins->toGetPropertyCache();
      // Inline polymorphic IC for a constant-atom key (`obj.foo`): self-guarding
      // per-site shape->offset cache, inline slot load on hit, WJH_PROPIC on miss/
      // non-object. NO deopt, and FAST for repeated shapes (gbemu's `a.b.c` wall
      // was the bare WJH_GETPROP helper per access). GECKO_WJ_GETPROPIC gates it.
      static int getPropIC = WJGETENV("GECKO_WJ_NOGETPROPIC") ? 0 : 1;
      MDefinition* idv = g->idval();
      if (getPropIC && idv->isConstant() &&
          idv->type() == MIRType::String) {
        JSString* s = idv->toConstant()->toJSValue().toString();
        if (s->isAtom()) {
          jsid id = js::AtomToId(&s->asAtom());
          uint32_t site = WJPropSite(be);
          if (site != 0) {
            return EmitGetPropIC(e, be, ins, g->value(),
                                 id.asRawBits(), site);
          }
        }
      }
      if (!EmitStageScratch(e, be, g->value(), 0)) return false;
      if (!EmitStageScratch(e, be, g->idval(), 1)) return false;
      return EmitHelperCallResult(e, be, ins, WJH_GETPROP, 0);
    }
    case MDefinition::Opcode::GetNameCache: {
      // Global/lexical name lookup. envObj = operand 0; the name is the GETNAME/
      // GETGNAME bytecode's atom operand at the node's resume-point pc (baked as a
      // StringValue, like Ion's IonGetNameIC reading the pc). Helper does the env
      // walk (GetEnvironmentName). Result is a Value.
      if (WJGETENV("GECKO_WJ_NOGETNAME")) return false;
      MResumePoint* rp = ins->resumePoint();
      // Use the resume point's OWN script (rp->pc() indexes that script's atoms).
      // Under COLDCALL inlining rp may belong to an inlined callee, NOT
      // be.info->script() -- using the caller's script indexes the wrong atom array
      // OOB -> MOZ_CRASH at compile (earley COLDCALL crash).
      JSScript* s = rp ? rp->block()->info().script() : nullptr;
      if (!rp || !rp->pc() || !s) return false;
      js::PropertyName* name = s->getName(rp->pc());
      if (!name) return false;
      uint64_t nameBits = JS::StringValue(name).asRawBits();
      // Mirror the interpreter's GetNameOperation kludge: if the NEXT bytecode op is
      // a typeof-name op, an unresolved global must yield undefined (NOT throw). Pass
      // this so the helper uses GetNameMode::TypeOf. (`typeof foo` compiles to
      // GetName foo; Typeof; the tolerance is keyed on the following op.)
      bool isTypeof = js::IsTypeOfNameOp(JSOp(*GetNextPc(rp->pc())));
      uint32_t site = WJAllocNameSite();
      if (site == 0) {  // out of IC sites: uncached helper (rare, still correct)
        if (!EmitStageScratch(e, be, ins->getOperand(0), 0)) return false;
        if (!EmitStageConstBoxed(e, be, nameBits, 1)) return false;
        return EmitHelperCallResult(e, be, ins, WJH_GETNAME,
                                    isTypeof ? 0x80000000u : 0u);
      }
      return EmitNameIC(e, be, ins, ins->getOperand(0), nameBits, site, isTypeof);
    }
    case MDefinition::Opcode::BindNameCache: {
      // Resolve the env object holding `name`'s binding (for a following SetName).
      // gbemu deopt-stormed here (op#314, 116k deopts -> whole fns to PBL) because
      // this had no JIT case. Helper does LookupNameUnqualified (no deopt); result
      // is an Object. Name baked from the BindName/BindGName bytecode atom at the
      // node's resume-point pc (like GetNameCache).
      if (WJGETENV("GECKO_WJ_NOBINDNAME")) return false;
      MResumePoint* rp = ins->resumePoint();
      // Resume point's own script (inlined-callee-safe; see GetNameCache above).
      JSScript* s = rp ? rp->block()->info().script() : nullptr;
      if (!rp || !rp->pc() || !s) return false;
      js::PropertyName* name = s->getName(rp->pc());
      if (!name) return false;
      uint64_t nameBits = JS::StringValue(name).asRawBits();
      if (!EmitStageScratch(e, be, ins->getOperand(0), 0)) return false;  // envChain
      if (!EmitStageConstBoxed(e, be, nameBits, 1)) return false;         // name
      if (!EmitHelperCallResult(e, be, ins, WJH_BINDNAME, 0)) return false;
      return EmitHelperResultAsType(e, be, ins->type());                 // Object
    }
    case MDefinition::Opcode::ToPropertyKeyCache: {
      if (!EmitStageScratch(e, be, ins->getOperand(0), 0)) return false;
      return EmitHelperCallResult(e, be, ins, WJH_TOPROPKEY, 0);  // Value
    }
    case MDefinition::Opcode::CharCodeAt: {
      static int noCCAInline = WJGETENV("GECKO_WJ_NOCCAINLINE") ? 1 : 0;
      if (!noCCAInline && ins->getOperand(0)->type() == MIRType::String &&
          ins->getOperand(1)->type() == MIRType::Int32) {
        return EmitInlineCharCodeAt(e, be, ins, /*negOnOob=*/false);
      }
      if (!EmitStageScratch(e, be, ins->getOperand(0), 0)) return false;  // string
      if (!EmitStageScratch(e, be, ins->getOperand(1), 1)) return false;  // index
      if (!EmitHelperCallResult(e, be, ins, WJH_CHARCODEAT, 0)) return false;
      return EmitHelperResultAsType(e, be, ins->type());  // Int32
    }
    case MDefinition::Opcode::FromCodePoint: {
      // String.fromCodePoint(cp). js::StringFromCodePoint ASSERTS cp<=NonBMPMax, so
      // deopt-guard invalid code points to PBL (which throws RangeError) -- mirrors
      // Ion's bailoutCmp32(Above, cp, NonBMPMax). Unsigned `> 0x10FFFF` also catches
      // negatives. Then helper-call StringFromCodePoint for the valid range.
      MDefinition* cp = ins->getOperand(0);
      int32_t l = be.local(cp);
      if (l < 0 || WJValType(cp->type()) != uint8_t(TypeCode::I32)) return false;
      if (!GetLocal(e, uint32_t(l)) || !e.writeOp(Op::I32Const) ||
          !e.writeVarS32(int32_t(0x10FFFF)) || !e.writeOp(Op::I32GtU))
        return false;  // (unsigned)cp > NonBMPMax ?
      if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
      if (!EmitDeopt(e, be)) return false;
      if (!e.writeOp(Op::End)) return false;
      if (!EmitStageScratch(e, be, cp, 0)) return false;  // codePoint (Int32)
      if (!EmitHelperCallResult(e, be, ins, WJH_FROMCODEPOINT, 0)) return false;
      return EmitHelperResultAsType(e, be, ins->type());  // String
    }
    case MDefinition::Opcode::CodePointAt: {
      // str.codePointAt(i) -- full code point (surrogate-pair aware). Used by string
      // iteration (`for (c of str)`). VM helper (js::jit::CodePointAt), like the
      // CharCodeAt helper fallback.
      if (!EmitStageScratch(e, be, ins->getOperand(0), 0)) return false;  // string
      if (!EmitStageScratch(e, be, ins->getOperand(1), 1)) return false;  // index (Int32)
      if (!EmitHelperCallResult(e, be, ins, WJH_CODEPOINTAT, 0)) return false;
      return EmitHelperResultAsType(e, be, ins->type());  // Int32
    }
    case MDefinition::Opcode::FromCharCodeEmptyIfNegative: {
      // String.fromCharCode fused with a possibly-negative index (charAt/at
      // patterns): negative code -> the empty atom; else the FromCharCode path.
      // DEFAULT-ON 2026-07-10: the earlier terser flake was NOT this emit -- it
      // was the Parameter-copy-local rooting gap (see WJRootSet's Parameter
      // case) exposed by terser's print() newly compiling; terser 12/12 clean
      // with the rooting fix. GECKO_WJ_NOFCCEIN reverts (whole-fn PBL bail).
      if (WJGETENV("GECKO_WJ_NOFCCEIN")) return false;
      JSContext* fcx = js::TlsContext.get();
      if (!fcx || !fcx->runtime()->staticStrings) return false;
      // Empty string via a GC-TRACED interned slot (NOT a raw bake): the raw
      // emptyString pointer bake was the prime suspect for the terser flake
      // (task #64) -- if the atom is relocatable in this build, a baked ptr
      // goes stale. The interned slot is traced+relocated by WJTraceRoots.
      uintptr_t emptySlot = WJInternConstant(
          JS::StringValue(fcx->runtime()->emptyString).asRawBits());
      if (!emptySlot) return false;
      uintptr_t tableBase2 = uintptr_t(fcx->staticStrings().unitStaticTableBase());
      if (!GetOp(e, be, ins->getOperand(0)) || !e.writeOp(Op::LocalTee) ||
          !e.writeVarU32(be.propShapeLocal))
        return false;                                             // [code], propShape=code
      if (!e.writeOp(Op::I32Const) || !e.writeVarS32(0) || !e.writeOp(Op::I32LtS))
        return false;                                             // [code<0]
      if (!e.writeOp(Op::If) || !e.writeFixedU8(uint8_t(TypeCode::I32)))
        return false;
      if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(emptySlot)) ||
          !e.writeOp(Op::I64Load) || !e.writeVarU32(3) || !e.writeVarU32(0) ||
          !e.writeOp(Op::I32WrapI64))
        return false;                                             // empty atom (traced slot)
      if (!e.writeOp(Op::Else)) return false;
      // code >= 0: c16 = code & 0xFFFF; static table if < UNIT_STATIC_LIMIT.
      if (!GetLocal(e, be.propShapeLocal) || !e.writeOp(Op::I32Const) ||
          !e.writeVarS32(0xFFFF) || !e.writeOp(Op::I32And) ||
          !e.writeOp(Op::LocalTee) || !e.writeVarU32(be.propShapeLocal))
        return false;
      if (!e.writeOp(Op::I32Const) ||
          !e.writeVarS32(int32_t(js::StaticStrings::UNIT_STATIC_LIMIT)) ||
          !e.writeOp(Op::I32LtU))
        return false;
      if (!e.writeOp(Op::If) || !e.writeFixedU8(uint8_t(TypeCode::I32)))
        return false;
      if (!GetLocal(e, be.propShapeLocal) || !e.writeOp(Op::I32Const) ||
          !e.writeVarS32(4) || !e.writeOp(Op::I32Mul) ||
          !e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(tableBase2)) ||
          !e.writeOp(Op::I32Add) || !e.writeOp(Op::I32Load) ||
          !e.writeVarU32(2) || !e.writeVarU32(0))
        return false;
      if (!e.writeOp(Op::Else)) return false;
      if (!EmitStageScratch(e, be, ins->getOperand(0), 0)) return false;
      if (!EmitHelperCallResult(e, be, ins, WJH_FROMCHARCODE, 0)) return false;
      if (!EmitHelperResultAsType(e, be, ins->type())) return false;
      if (!e.writeOp(Op::End)) return false;  // inner if
      return e.writeOp(Op::End);              // outer if
    }
    case MDefinition::Opcode::FromCharCode: {
      // Inline the StaticStrings fast path (CodeUnitToString): c16 = code & 0xFFFF;
      // if c16 < UNIT_STATIC_LIMIT(256) return the PERMANENT unitStaticTable[c16] via a
      // direct table load -- no alloc, no VM helper, no GC/rooting (permanent atoms).
      // Kills the ~1.2M WJH_FROMCHARCODE C++ hops in pdfjs (string/char-heavy real code).
      // Result is a raw String ptr (i32), like the helper.
      if (true) {  // ASCII FromCharCode inline (permanent)
        JSContext* cx = js::TlsContext.get();
        if (cx && cx->runtime()->staticStrings) {
          uintptr_t tableBase =
              uintptr_t(cx->staticStrings().unitStaticTableBase());
          if (!GetOp(e, be, ins->getOperand(0))) return false;         // [code i32]
          if (!e.writeOp(Op::I32Const) || !e.writeVarS32(0xFFFF) ||
              !e.writeOp(Op::I32And) || !e.writeOp(Op::LocalTee) ||
              !e.writeVarU32(be.propShapeLocal))
            return false;                                              // [c16], propShapeLocal=c16
          if (!e.writeOp(Op::I32Const) ||
              !e.writeVarS32(int32_t(js::StaticStrings::UNIT_STATIC_LIMIT)) ||
              !e.writeOp(Op::I32LtU))
            return false;                                              // [c16<256]
          if (!e.writeOp(Op::If) || !e.writeFixedU8(uint8_t(TypeCode::I32)))
            return false;                                              // -> i32 (String ptr)
          // unitStaticTable[c16]: *(tableBase + c16*4)  (wasm32: JSAtom* is 4 bytes)
          if (!GetLocal(e, be.propShapeLocal) || !e.writeOp(Op::I32Const) ||
              !e.writeVarS32(4) || !e.writeOp(Op::I32Mul) ||
              !e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(tableBase)) ||
              !e.writeOp(Op::I32Add) || !e.writeOp(Op::I32Load) ||
              !e.writeVarU32(2) || !e.writeVarU32(0))
            return false;                                              // [String ptr]
          if (!e.writeOp(Op::Else)) return false;
          // c16 >= 256: allocate via the helper (it re-truncates the original code).
          if (!EmitStageScratch(e, be, ins->getOperand(0), 0)) return false;
          if (!EmitHelperCallResult(e, be, ins, WJH_FROMCHARCODE, 0)) return false;
          if (!EmitHelperResultAsType(e, be, ins->type())) return false;
          return e.writeOp(Op::End);
        }
      }
      if (!EmitStageScratch(e, be, ins->getOperand(0), 0)) return false;  // code
      if (!EmitHelperCallResult(e, be, ins, WJH_FROMCHARCODE, 0)) return false;
      return EmitHelperResultAsType(e, be, ins->type());  // String
    }
    case MDefinition::Opcode::ToString: {
      MDefinition* in = ins->getOperand(0);
      // Statically a string -> ToString is identity (Warp usually folds this, but be safe).
      if (in->type() == MIRType::String) return GetOp(e, be, in);
      // Value input: inline the already-a-string fast path (no WJH_TOSTRING hop). uBlock's
      // parser does ~600K ToStrings/iter, many on values that are already strings.
      static int noInline = WJGETENV("GECKO_WJ_NOTOSTRINLINE") ? 1 : 0;
      if (in->type() == MIRType::Value && !noInline) {
        if (!GetOp(e, be, in) || !e.writeOp(Op::LocalTee) ||
            !e.writeVarU32(be.unboxScratch))
          return false;
        // (value >> 32) == STRING tag ?
        if (!e.writeOp(Op::I64Const) || !e.writeVarS64(32) || !e.writeOp(Op::I64ShrU) ||
            !e.writeOp(Op::I32WrapI64) || !e.writeOp(Op::I32Const) ||
            !e.writeVarS32(int32_t(TagWord(JSVAL_TYPE_STRING))) || !e.writeOp(Op::I32Eq))
          return false;
        if (!e.writeOp(Op::If) || !e.writeFixedU8(0x7F)) return false;  // -> i32
        if (!GetLocal(e, be.unboxScratch) || !e.writeOp(Op::I32WrapI64))
          return false;  // unbox the string pointer (identity ToString)
        if (!e.writeOp(Op::Else)) return false;
        // non-string: stage the value + call the helper (full ToString).
        if (!GetLocal(e, be.scratchBase) || !GetLocal(e, be.unboxScratch) ||
            !e.writeOp(Op::I64Store) || !e.writeVarU32(3) || !e.writeVarU32(0))
          return false;
        if (!EmitHelperCallResult(e, be, ins, WJH_TOSTRING, 0)) return false;
        if (!EmitHelperResultAsType(e, be, ins->type())) return false;
        return e.writeOp(Op::End);
      }
      if (!EmitStageScratch(e, be, in, 0)) return false;
      if (!EmitHelperCallResult(e, be, ins, WJH_TOSTRING, 0)) return false;
      return EmitHelperResultAsType(e, be, ins->type());  // String
    }
    case MDefinition::Opcode::LinearizeString:
    case MDefinition::Opcode::LinearizeForCharAccess: {
      // Flatten a (possibly rope) string to a linear one for char access. The index
      // operand of LinearizeForCharAccess is a hint; a full ensureLinear is valid for
      // any index, so both map to WJH_LINEARIZE(string). Unblocks uBlock's
      // parseNetPattern/validateNet tokenizers (which scan chars).
      if (!EmitStageScratch(e, be, ins->getOperand(0), 0)) return false;  // string
      if (!EmitHelperCallResult(e, be, ins, WJH_LINEARIZE, 0)) return false;
      return EmitHelperResultAsType(e, be, ins->type());  // String
    }
    case MDefinition::Opcode::StringIncludes:
    case MDefinition::Opcode::StringIndexOf:
    case MDefinition::Opcode::StringLastIndexOf:
    case MDefinition::Opcode::StringStartsWith:
    case MDefinition::Opcode::StringEndsWith: {
      // (string, searchString) -> Bool/Int32 via WJH_STROP. uBlock parser tokenizers.
      uint32_t sub = ins->op() == MDefinition::Opcode::StringIncludes      ? 0
                     : ins->op() == MDefinition::Opcode::StringIndexOf     ? 1
                     : ins->op() == MDefinition::Opcode::StringLastIndexOf ? 2
                     : ins->op() == MDefinition::Opcode::StringStartsWith  ? 3
                                                                           : 4;
      if (!EmitStageScratch(e, be, ins->getOperand(0), 0)) return false;
      if (!EmitStageScratch(e, be, ins->getOperand(1), 1)) return false;
      if (!EmitHelperCallResult(e, be, ins, WJH_STROP, sub)) return false;
      return EmitHelperResultAsType(e, be, ins->type());  // Boolean / Int32
    }
    case MDefinition::Opcode::Substr: {
      // (string, begin, length) -> String via WJH_STROP sub=5.
      if (!EmitStageScratch(e, be, ins->getOperand(0), 0)) return false;
      if (!EmitStageScratch(e, be, ins->getOperand(1), 1)) return false;
      if (!EmitStageScratch(e, be, ins->getOperand(2), 2)) return false;
      if (!EmitHelperCallResult(e, be, ins, WJH_STROP, 5)) return false;
      return EmitHelperResultAsType(e, be, ins->type());  // String
    }
    case MDefinition::Opcode::StringConvertCase: {
      // toLowerCase/toUpperCase -> String via WJH_STROP sub=6/7.
      uint32_t sub =
          ins->toStringConvertCase()->stringCase() == jit::StringCase::Lower ? 6 : 7;
      if (!EmitStageScratch(e, be, ins->getOperand(0), 0)) return false;
      if (!EmitHelperCallResult(e, be, ins, WJH_STROP, sub)) return false;
      return EmitHelperResultAsType(e, be, ins->type());  // String
    }
    case MDefinition::Opcode::CharCodeAtOrNegative: {
      // Like CharCodeAt but -1 for out-of-bounds (no throw). WJH_CHARCODEAT site=1.
      static int noCCAInline = WJGETENV("GECKO_WJ_NOCCAINLINE") ? 1 : 0;
      if (!noCCAInline && ins->getOperand(0)->type() == MIRType::String &&
          ins->getOperand(1)->type() == MIRType::Int32) {
        return EmitInlineCharCodeAt(e, be, ins, /*negOnOob=*/true);
      }
      if (!EmitStageScratch(e, be, ins->getOperand(0), 0)) return false;  // string
      if (!EmitStageScratch(e, be, ins->getOperand(1), 1)) return false;  // index
      if (!EmitHelperCallResult(e, be, ins, WJH_CHARCODEAT, 1)) return false;
      return EmitHelperResultAsType(e, be, ins->type());  // Int32
    }
    case MDefinition::Opcode::TypeOf: {
      // -> JSType enum (Int32). MTypeOfName maps it to a string separately.
      if (!EmitStageScratch(e, be, ins->getOperand(0), 0)) return false;
      if (!EmitHelperCallResult(e, be, ins, WJH_TYPEOF, 0)) return false;
      return EmitHelperResultAsType(e, be, ins->type());  // Int32
    }
    case MDefinition::Opcode::IsArray: {
      // Array.isArray (handles proxies via the helper). -> Boolean.
      if (!EmitStageScratch(e, be, ins->getOperand(0), 0)) return false;
      if (!EmitHelperCallResult(e, be, ins, WJH_ISARRAY, 0)) return false;
      return EmitHelperResultAsType(e, be, ins->type());  // Boolean
    }
    case MDefinition::Opcode::IsTypedArray: {
      // -> Boolean. Ion's IsTypedArrayClass(clasp): clasp in the contiguous
      // pointer range [firstTypedArrayClass, lastTypedArrayClass] (unsigned).
      // Unblocks self-hosted ArrayIteratorNext (the for-of `.next`) which guards
      // typed-array iterands. possiblyWrapped (proxy) -> bail (may throw, rare).
      const bool wrapped = ins->toIsTypedArray()->isPossiblyWrapped();
      MDefinition* val = ins->getOperand(0);
      const uint32_t firstTA =
          uint32_t(uintptr_t(std::begin(js::TypedArrayObject::fixedLengthClasses)));
      const uint32_t lastTA = uint32_t(uintptr_t(
          std::prev(std::end(js::TypedArrayObject::resizableClasses))));
      uint32_t shapeOff = uint32_t(offsetof(JS::shadow::Object, shape));
      uint32_t flagsOff = uint32_t(offsetof(JSClass, flags));
      // obj ptr on stack -> load clasp -> stash in callScratch.
      auto claspToScratch = [&]() -> bool {
        return e.writeOp(Op::I32Load) && e.writeVarU32(2) &&
               e.writeVarU32(shapeOff) && e.writeOp(Op::I32Load) &&
               e.writeVarU32(2) &&
               e.writeVarU32(uint32_t(js::Shape::offsetOfBaseShape())) &&
               e.writeOp(Op::I32Load) && e.writeVarU32(2) &&
               e.writeVarU32(uint32_t(js::BaseShape::offsetOfClasp())) &&
               e.writeOp(Op::LocalSet) && e.writeVarU32(be.callScratch);
      };
      // push (clasp >=u first) & (clasp <=u last) from callScratch.
      auto inRange = [&]() -> bool {
        return e.writeOp(Op::LocalGet) && e.writeVarU32(be.callScratch) &&
               e.writeOp(Op::I32Const) && e.writeVarS32(int32_t(firstTA)) &&
               e.writeOp(Op::I32GeU) && e.writeOp(Op::LocalGet) &&
               e.writeVarU32(be.callScratch) && e.writeOp(Op::I32Const) &&
               e.writeVarS32(int32_t(lastTA)) && e.writeOp(Op::I32LeU) &&
               e.writeOp(Op::I32And);
      };
      // Assumes obj ptr on stack. Pushes the Boolean result. For wrapped: a
      // non-TA proxy deopts to PBL (IsPossiblyWrappedTypedArray unwraps it);
      // uBlock has no wrappers so this never storms.
      auto emitFromObjPtr = [&]() -> bool {
        if (!claspToScratch() || !inRange()) return false;
        if (!wrapped) return true;
        if (!e.writeOp(Op::If) || !e.writeFixedU8(0x7F)) return false;
        if (!e.writeOp(Op::I32Const) || !e.writeVarS32(1)) return false;
        if (!e.writeOp(Op::Else)) return false;
        if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(be.callScratch) ||
            !e.writeOp(Op::I32Load) || !e.writeVarU32(2) ||
            !e.writeVarU32(flagsOff) || !e.writeOp(Op::I32Const) ||
            !e.writeVarS32(int32_t(JSCLASS_IS_PROXY)) || !e.writeOp(Op::I32And))
          return false;
        if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
        bool saved = be.inDispatchBody;
        be.inDispatchBody = false;  // inline deopt (nested blocks)
        bool ok = EmitDeopt(e, be);
        be.inDispatchBody = saved;
        if (!ok || !e.writeOp(Op::End)) return false;
        if (!e.writeOp(Op::I32Const) || !e.writeVarS32(0)) return false;
        return e.writeOp(Op::End);
      };
      if (val->type() == MIRType::Object) {
        return GetOp(e, be, val) && emitFromObjPtr();
      }
      if (val->type() == MIRType::Value) {
        // tag==OBJECT ? emitFromObjPtr(unbox) : 0
        if (!GetOp(e, be, val) || !e.writeOp(Op::I64Const) ||
            !e.writeVarS64(32) || !e.writeOp(Op::I64ShrU) ||
            !e.writeOp(Op::I32WrapI64) || !e.writeOp(Op::I32Const) ||
            !e.writeVarS32(int32_t(TagWord(JSVAL_TYPE_OBJECT))) ||
            !e.writeOp(Op::I32Eq))
          return false;
        if (!e.writeOp(Op::If) || !e.writeFixedU8(0x7F)) return false;
        if (!GetOp(e, be, val) || !e.writeOp(Op::I32WrapI64) || !emitFromObjPtr())
          return false;
        if (!e.writeOp(Op::Else) || !e.writeOp(Op::I32Const) ||
            !e.writeVarS32(0))
          return false;
        return e.writeOp(Op::End);
      }
      return false;
    }
    case MDefinition::Opcode::RegExp: {
      // Regex literal: clone the baked source RegExpObject. The source is a compile-
      // time argument -> bake it (boxed) into scratch[0] via the const pool, clone.
      JSObject* src = ins->toRegExp()->source();
      if (!src) return false;
      if (!EmitStageConstBoxed(e, be, JS::ObjectValue(*src).asRawBits(), 0))
        return false;
      if (!EmitHelperCallResult(e, be, ins, WJH_REGEXPCLONE, 0)) return false;
      return EmitHelperResultAsType(e, be, ins->type());  // Object
    }
    case MDefinition::Opcode::TypeOfName: {
      // JSType int -> typeof-name string (permanent atom). MTypeOf companion.
      if (!EmitStageScratch(e, be, ins->getOperand(0), 0)) return false;
      if (!EmitHelperCallResult(e, be, ins, WJH_TYPEOFNAME, 0)) return false;
      return EmitHelperResultAsType(e, be, ins->type());  // String
    }
    case MDefinition::Opcode::NegativeToNaN: {
      // input<0 ? NaN : Double(input), boxed as a double Value. (CharCodeAtOrNegative
      // companion: -1 -> NaN for String.prototype.charCodeAt out-of-bounds.)
      MDefinition* in = ins->getOperand(0);
      if (!GetOp(e, be, in) || !e.writeOp(Op::I32Const) || !e.writeVarS32(0) ||
          !e.writeOp(Op::I32LtS))
        return false;
      if (!e.writeOp(Op::If) || !e.writeFixedU8(0x7C)) return false;  // -> f64
      if (!e.writeOp(Op::I64Const) ||
          !e.writeVarS64(int64_t(0x7FF8000000000000ULL)) ||
          !e.writeOp(Op::F64ReinterpretI64))
        return false;  // canonical NaN
      if (!e.writeOp(Op::Else)) return false;
      if (!GetOp(e, be, in) || !e.writeOp(Op::F64ConvertI32S)) return false;
      if (!e.writeOp(Op::End)) return false;
      return EmitBoxFromStack(e, MIRType::Double);  // -> boxed Value (i64)
    }
    case MDefinition::Opcode::BinaryCache: {
      // The JSOp comes from the node's RESUME POINT pc (matching Ion's
      // visitBinaryValueCache -- NOT trackedSite, which points at a JumpTarget
      // here). Only the 12 arith ops reach a BinaryValueCache (comparisons use
      // MCompare).
      MResumePoint* rp = ins->resumePoint();
      if (!rp || !rp->pc()) return false;
      JSOp jsop = JSOp(*rp->pc());
      MIRType ty = ins->type();
      int32_t l0 = be.local(ins->getOperand(0)), l1 = be.local(ins->getOperand(1));

      // (A) SPECULATIVE INLINE f64 ARITH (default-on). A cold/unspecialized arith
      // IC yields an untyped MBinaryCache; the IC never warms in this PBL build
      // (CacheIR stubs can't attach -> Warp stays blind), so navier's hot double
      // loops were stuck on it. Instead of a C++ helper hop, unbox both operands
      // as NUMBERS (int32 -> convert, double -> reinterpret, else DEOPT to PBL --
      // correct for string concat / objects) and do the f64 op inline. This is
      // typed arith WITHOUT needing the IC to warm. +,-,*,/ map directly to f64
      // ops (JS number semantics); %/** and bitops fall through to (B).
      Op fop = Op::F64Add;
      bool f64op = true;
      switch (jsop) {
        case JSOp::Add: fop = Op::F64Add; break;
        case JSOp::Sub: fop = Op::F64Sub; break;
        case JSOp::Mul: fop = Op::F64Mul; break;
        case JSOp::Div: fop = Op::F64Div; break;
        default: f64op = false; break;
      }
      // ty==Double: operands are proven numbers -> inline f64 (the fallible unbox's
      // deopt never fires).
      if (f64op && l0 >= 0 && l1 >= 0 && ty == MIRType::Double) {
        if (!EmitUnboxLocal(e, be, uint32_t(l0), MIRType::Double, /*fallible=*/true))
          return false;
        if (!EmitUnboxLocal(e, be, uint32_t(l1), MIRType::Double, /*fallible=*/true))
          return false;
        return e.writeOp(fop);  // f64 result matches the f64 local
      }
      // ty==Value Add/Sub/Mul/Div: the operands COULD be non-numbers (splay's string
      // concat `'..' + tag + '..'`). The old code unboxed them as Double and DEOPTED
      // on strings every time -> deopt-resume churn/corruption. Instead branch at
      // runtime with NO deopt: if BOTH operands are numbers, inline f64 (keeps
      // navier's Value-typed numeric arith fast); else call WJH_BINARYARITH
      // (AddValues/... -- correct for strings + mixed types). The result is a boxed
      // Value (i64) either way.
      if (f64op && l0 >= 0 && l1 >= 0 && ty == MIRType::Value) {
        auto pushTag = [&](int32_t l) -> bool {
          return GetLocal(e, uint32_t(l)) && e.writeOp(Op::I64Const) &&
                 e.writeVarS64(32) && e.writeOp(Op::I64ShrU) &&
                 e.writeOp(Op::I32WrapI64);
        };
        auto pushIsNum = [&](int32_t l) -> bool {  // (tag u<= CLEAR) | (tag == INT32)
          return pushTag(l) && e.writeOp(Op::I32Const) &&
                 e.writeVarS32(int32_t(uint32_t(JSVAL_TAG_CLEAR))) &&
                 e.writeOp(Op::I32LeU) && pushTag(l) && e.writeOp(Op::I32Const) &&
                 e.writeVarS32(int32_t(TagWord(JSVAL_TYPE_INT32))) &&
                 e.writeOp(Op::I32Eq) && e.writeOp(Op::I32Or);
        };
        auto pushF64 = [&](int32_t l) -> bool {  // int32 -> convert; else reinterpret
          if (!pushTag(l) || !e.writeOp(Op::I32Const) ||
              !e.writeVarS32(int32_t(TagWord(JSVAL_TYPE_INT32))) ||
              !e.writeOp(Op::I32Eq) || !e.writeOp(Op::If) || !e.writeFixedU8(0x7C))
            return false;
          if (!GetLocal(e, uint32_t(l)) || !e.writeOp(Op::I32WrapI64) ||
              !e.writeOp(Op::F64ConvertI32S) || !e.writeOp(Op::Else))
            return false;
          if (!GetLocal(e, uint32_t(l)) || !e.writeOp(Op::F64ReinterpretI64) ||
              !e.writeOp(Op::End))
            return false;
          return true;
        };
        if (!pushIsNum(l0) || !pushIsNum(l1) || !e.writeOp(Op::I32And)) return false;
        if (!e.writeOp(Op::If) || !e.writeFixedU8(uint8_t(TypeCode::I64)))
          return false;
        // both numbers: inline f64, box to a double Value.
        if (!pushF64(l0) || !pushF64(l1) || !e.writeOp(fop)) return false;
        if (!EmitBoxFromStack(e, MIRType::Double)) return false;
        if (!e.writeOp(Op::Else)) return false;
        // non-numbers. INLINE ROPE for string+string Add (splay's
        // `'String for key '+tag+'...'` = 2.28M WJH_BINARYARITH->ConcatStrings hops,
        // each a C++ nursery string alloc). When both operands are strings, both
        // non-empty, and wholeLen > fat-inline latin1 max (24) -- so ConcatStrings
        // would ROPE not flatten, regardless of encoding -- bump-alloc a JSRope inline
        // (flags/length/left/right; nursery-fresh so NO post-barrier) instead of the
        // C++ helper. Everything else (non-strings / short / overflow / nursery-full)
        // falls to WJH_BINARYARITH (correct for all). GECKO_WJ_NOINLINECONCAT reverts.
        static int inlineConcat = WJGETENV("GECKO_WJ_NOINLINECONCAT") ? 0 : 1;
        if (!e.writeOp(Op::Block) || !e.writeFixedU8(uint8_t(TypeCode::I64)))
          return false;
        if (inlineConcat && jsop == JSOp::Add && gWJStringHeaderWord &&
            gWJNurseryPosAddr && l0 >= 0 && l1 >= 0) {
          uint32_t hdrSize = uint32_t(js::Nursery::nurseryCellHeaderSize());
          uint32_t totalSize =
              uint32_t(js::gc::Arena::thingSize(js::gc::AllocKind::STRING)) +
              hdrSize;
          int32_t posAddr = int32_t(gWJNurseryPosAddr);
          int32_t endOff = js::Nursery::offsetOfCurrentEndFromPosition();
          uint32_t flagsOff = uint32_t(JSString::offsetOfFlags());
          uint32_t lenOff = uint32_t(JSString::offsetOfLength());
          // JSRope::offsetOfLeft/Right are private; use the layout-compatible
          // public JS::shadow::String (d.s.u2.left == the nonInlineChars union slot,
          // d.s.u3.right == the externalCallbacks slot).
          uint32_t leftOff =
              uint32_t(offsetof(JS::shadow::String, nonInlineCharsLatin1));
          uint32_t rightOff =
              uint32_t(offsetof(JS::shadow::String, externalCallbacks));
          int32_t ropeFlagsL1 =
              int32_t(js::StringFlags::ropeFlags(js::CharEncoding::Latin1));
          int32_t ropeFlagsTB =
              int32_t(js::StringFlags::ropeFlags(js::CharEncoding::TwoByte));
          int32_t latin1Bit = int32_t(js::StringFlags::LATIN1_CHARS_BIT);
          int32_t fatMaxL1 = int32_t(JSFatInlineString::MAX_LENGTH_LATIN1);
          int32_t maxLen = int32_t(uint32_t(JSString::MAX_LENGTH));
          int32_t strTag = int32_t(TagWord(JSVAL_TYPE_STRING));
          int32_t strHdr = int32_t(uint32_t(gWJStringHeaderWord));
          uint32_t lp = be.propObjLocal, rp = be.propShapeLocal,
                   wl = be.propTaggedLocal, ap = be.allocPosLocal,
                   ro = be.allocObjLocal;
          auto I32C = [&](int32_t v) {
            return e.writeOp(Op::I32Const) && e.writeVarS32(v);
          };
          auto ld = [&](uint32_t base, uint32_t off) {
            return GetLocal(e, base) && e.writeOp(Op::I32Load) &&
                   e.writeVarU32(2) && e.writeVarU32(off);
          };
          auto st = [&](uint32_t off) {
            return e.writeOp(Op::I32Store) && e.writeVarU32(2) &&
                   e.writeVarU32(off);
          };
          // both strings?
          if (!pushTag(l0) || !I32C(strTag) || !e.writeOp(Op::I32Eq)) return false;
          if (!pushTag(l1) || !I32C(strTag) || !e.writeOp(Op::I32Eq)) return false;
          if (!e.writeOp(Op::I32And)) return false;
          if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
          // lp = wrap(l0); rp = wrap(l1)
          if (!GetLocal(e, uint32_t(l0)) || !e.writeOp(Op::I32WrapI64) ||
              !e.writeOp(Op::LocalSet) || !e.writeVarU32(lp))
            return false;
          if (!GetLocal(e, uint32_t(l1)) || !e.writeOp(Op::I32WrapI64) ||
              !e.writeOp(Op::LocalSet) || !e.writeVarU32(rp))
            return false;
          // wl = leftLen + rightLen
          if (!ld(lp, lenOff) || !ld(rp, lenOff) || !e.writeOp(Op::I32Add) ||
              !e.writeOp(Op::LocalSet) || !e.writeVarU32(wl))
            return false;
          // eligible = leftLen>0 & rightLen>0 & wl>fatMaxL1 & wl<=maxLen
          if (!ld(lp, lenOff) || !I32C(0) || !e.writeOp(Op::I32Ne)) return false;
          if (!ld(rp, lenOff) || !I32C(0) || !e.writeOp(Op::I32Ne) ||
              !e.writeOp(Op::I32And))
            return false;
          if (!GetLocal(e, wl) || !I32C(fatMaxL1) || !e.writeOp(Op::I32GtU) ||
              !e.writeOp(Op::I32And))
            return false;
          if (!GetLocal(e, wl) || !I32C(maxLen) || !e.writeOp(Op::I32LeU) ||
              !e.writeOp(Op::I32And))
            return false;
          if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
          // ap = *posAddr
          if (!I32C(posAddr) || !e.writeOp(Op::I32Load) || !e.writeVarU32(2) ||
              !e.writeVarU32(0) || !e.writeOp(Op::LocalSet) || !e.writeVarU32(ap))
            return false;
          // room? *(posAddr+endOff) >= ap + totalSize
          if (!I32C(posAddr) || !e.writeOp(Op::I32Load) || !e.writeVarU32(2) ||
              !e.writeVarU32(uint32_t(endOff)))
            return false;
          if (!GetLocal(e, ap) || !I32C(int32_t(totalSize)) ||
              !e.writeOp(Op::I32Add) || !e.writeOp(Op::I32GeU))
            return false;
          if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
          // commit *posAddr = ap + totalSize
          if (!I32C(posAddr) || !GetLocal(e, ap) || !I32C(int32_t(totalSize)) ||
              !e.writeOp(Op::I32Add) || !st(0))
            return false;
          // header at ap
          if (!GetLocal(e, ap) || !I32C(strHdr) || !st(0)) return false;
          // ro = ap + hdrSize
          if (!GetLocal(e, ap) || !I32C(int32_t(hdrSize)) ||
              !e.writeOp(Op::I32Add) || !e.writeOp(Op::LocalSet) ||
              !e.writeVarU32(ro))
            return false;
          // [ro+flagsOff] = (bothLatin1 ? ropeFlagsL1 : ropeFlagsTB)
          if (!GetLocal(e, ro)) return false;
          if (!ld(lp, flagsOff) || !I32C(latin1Bit) || !e.writeOp(Op::I32And) ||
              !ld(rp, flagsOff) || !I32C(latin1Bit) || !e.writeOp(Op::I32And) ||
              !e.writeOp(Op::I32And))
            return false;
          if (!e.writeOp(Op::If) || !e.writeFixedU8(uint8_t(TypeCode::I32)) ||
              !I32C(ropeFlagsL1) || !e.writeOp(Op::Else) || !I32C(ropeFlagsTB) ||
              !e.writeOp(Op::End))
            return false;
          if (!st(flagsOff)) return false;
          // [ro+lenOff] = wl
          if (!GetLocal(e, ro) || !GetLocal(e, wl) || !st(lenOff)) return false;
          // [ro+leftOff] = lp ; [ro+rightOff] = rp
          if (!GetLocal(e, ro) || !GetLocal(e, lp) || !st(leftOff)) return false;
          if (!GetLocal(e, ro) || !GetLocal(e, rp) || !st(rightOff)) return false;
          // result = box(ro as String); br out of the i64 block (depth 3)
          if (!GetLocal(e, ro) || !EmitBoxFromStack(e, MIRType::String))
            return false;
          if (!e.writeOp(Op::Br) || !e.writeVarU32(3)) return false;
          if (!e.writeOp(Op::End)) return false;  // end room-if
          if (!e.writeOp(Op::End)) return false;  // end eligible-if
          if (!e.writeOp(Op::End)) return false;  // end bothStrings-if
        }
        // helper fallback (leaves boxed i64 -> the i64 block result)
        if (!EmitStageScratch(e, be, ins->getOperand(0), 0)) return false;
        if (!EmitStageScratch(e, be, ins->getOperand(1), 1)) return false;
        if (!EmitHelperCallResult(e, be, ins, WJH_BINARYARITH, uint32_t(jsop)))
          return false;
        if (!e.writeOp(Op::End)) return false;  // end i64 block
        return e.writeOp(Op::End);              // end bothNumbers if
      }

      // (A2) INLINE i32 ARITH for an Int32-typed result. Bitwise ops directly;
      // Add/Sub/Mul in the i64 domain with an overflow deopt (a result that
      // doesn't fit int32 becomes a double, violating the Int32 type). Operands
      // unboxed as int32 (deopt if not). Mod/Pow/Ursh fall to (B).
      if (ty == MIRType::Int32 && l0 >= 0 && l1 >= 0) {
        Op bop = Op::I32Or;
        bool bit = true;
        switch (jsop) {
          case JSOp::BitOr: bop = Op::I32Or; break;
          case JSOp::BitAnd: bop = Op::I32And; break;
          case JSOp::BitXor: bop = Op::I32Xor; break;
          case JSOp::Lsh: bop = Op::I32Shl; break;
          case JSOp::Rsh: bop = Op::I32ShrS; break;
          default: bit = false; break;
        }
        if (bit) {
          if (!EmitUnboxLocal(e, be, uint32_t(l0), MIRType::Int32, true)) return false;
          if (!EmitUnboxLocal(e, be, uint32_t(l1), MIRType::Int32, true)) return false;
          return e.writeOp(bop);
        }
        if (jsop == JSOp::Add || jsop == JSOp::Sub || jsop == JSOp::Mul) {
          Op iop = jsop == JSOp::Add   ? Op::I64Add
                   : jsop == JSOp::Sub ? Op::I64Sub
                                       : Op::I64Mul;
          if (!EmitUnboxLocal(e, be, uint32_t(l0), MIRType::Int32, true)) return false;
          if (!e.writeOp(Op::I64ExtendI32S)) return false;
          if (!EmitUnboxLocal(e, be, uint32_t(l1), MIRType::Int32, true)) return false;
          if (!e.writeOp(Op::I64ExtendI32S)) return false;
          if (!e.writeOp(iop)) return false;  // exact i64 result
          if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(be.unboxScratch))
            return false;
          // deopt unless result == sign_extend(wrap32(result)) (i.e. fits int32).
          if (!GetLocal(e, be.unboxScratch) || !e.writeOp(Op::I32WrapI64) ||
              !e.writeOp(Op::I64ExtendI32S) || !GetLocal(e, be.unboxScratch) ||
              !e.writeOp(Op::I64Ne))
            return false;
          if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
          if (!EmitDeopt(e, be)) return false;
          if (!e.writeOp(Op::End)) return false;
          return GetLocal(e, be.unboxScratch) && e.writeOp(Op::I32WrapI64);
        }
      }

      // (A3) COMPARISON (Boolean-typed MBinaryCache): a cold/untyped ==,!=,<,<=,>,
      // >=,===,!== whose operands our inline Compare couldn't type. Route to the
      // WJH_COMPARE VM helper (correct for any operand types). It's a C++ hop
      // (~PBL speed for THIS op) but keeps the whole FUNCTION compiled in JIT rather
      // than bailing it to PBL (deltablue's constraint comparisons did the latter
      // -> whole functions ran in PBL -> slow). Default-on: a compile-coverage win.
      if (ins->type() == MIRType::Boolean) {
        Op cfop = Op::F64Eq;
        switch (jsop) {
          case JSOp::Eq: case JSOp::StrictEq: cfop = Op::F64Eq; break;
          case JSOp::Ne: case JSOp::StrictNe: cfop = Op::F64Ne; break;
          case JSOp::Lt: cfop = Op::F64Lt; break;
          case JSOp::Le: cfop = Op::F64Le; break;
          case JSOp::Gt: cfop = Op::F64Gt; break;
          case JSOp::Ge: cfop = Op::F64Ge; break;
          default: return false;
        }
        // (A3a) SPECULATIVE INLINE NUMERIC COMPARE (default-on) -- the exact analog
        // of the (A) speculative-f64-arith path. A cold/untyped Boolean MBinaryCache
        // compare hits WJH_COMPARE every time (gbemu 2.34M, earley 8M C++ hops). Branch
        // at runtime with NO deopt: if BOTH operands are numbers, convert to f64 and do
        // the comparison inline (correct for all 8 ops when both are numbers -- <,<=,>,>=
        // are numeric and ==/=== reduce to numeric equality, 1===1.0 true); else call
        // WJH_COMPARE (correct for strings/objects/mixed). GECKO_WJ_NOSPECCMP reverts.
        if (l0 >= 0 && l1 >= 0 && !WJGETENV("GECKO_WJ_NOSPECCMP")) {
          auto pushTag = [&](int32_t l) -> bool {
            return GetLocal(e, uint32_t(l)) && e.writeOp(Op::I64Const) &&
                   e.writeVarS64(32) && e.writeOp(Op::I64ShrU) &&
                   e.writeOp(Op::I32WrapI64);
          };
          auto pushIsNum = [&](int32_t l) -> bool {  // (tag u<= CLEAR) | (tag == INT32)
            return pushTag(l) && e.writeOp(Op::I32Const) &&
                   e.writeVarS32(int32_t(uint32_t(JSVAL_TAG_CLEAR))) &&
                   e.writeOp(Op::I32LeU) && pushTag(l) && e.writeOp(Op::I32Const) &&
                   e.writeVarS32(int32_t(TagWord(JSVAL_TYPE_INT32))) &&
                   e.writeOp(Op::I32Eq) && e.writeOp(Op::I32Or);
          };
          auto pushF64 = [&](int32_t l) -> bool {  // int32 -> convert; else reinterpret
            if (!pushTag(l) || !e.writeOp(Op::I32Const) ||
                !e.writeVarS32(int32_t(TagWord(JSVAL_TYPE_INT32))) ||
                !e.writeOp(Op::I32Eq) || !e.writeOp(Op::If) || !e.writeFixedU8(0x7C))
              return false;
            if (!GetLocal(e, uint32_t(l)) || !e.writeOp(Op::I32WrapI64) ||
                !e.writeOp(Op::F64ConvertI32S) || !e.writeOp(Op::Else))
              return false;
            if (!GetLocal(e, uint32_t(l)) || !e.writeOp(Op::F64ReinterpretI64) ||
                !e.writeOp(Op::End))
              return false;
            return true;
          };
          if (!pushIsNum(l0) || !pushIsNum(l1) || !e.writeOp(Op::I32And)) return false;
          if (!e.writeOp(Op::If) || !e.writeFixedU8(uint8_t(TypeCode::I32)))
            return false;
          // both numbers: inline f64 compare (i32 0/1 result).
          if (!pushF64(l0) || !pushF64(l1) || !e.writeOp(cfop)) return false;
          if (!e.writeOp(Op::Else)) return false;
          // non-numbers. For ===/!== inline IDENTITY: same tag &&
          // tag in {Object,Symbol,Bool,Null,Undefined} -> the boxed bits are equal
          // iff strict-equal (reference identity / canonical singletons); String/BigInt
          // are value-equality -> helper; different tags -> strict-false (the only
          // cross-type === is int32/double, already handled by the numeric branch
          // above). Covers earley's millions of cons/symbol `eq?` identity compares
          // (8.9M WJH_COMPARE) with no C++ hop. Loose ==/relational keep the helper.
          // GECKO_WJ_NOSPECEQ reverts.
          auto cmpHelper = [&]() -> bool {
            return EmitStageScratch(e, be, ins->getOperand(0), 0) &&
                   EmitStageScratch(e, be, ins->getOperand(1), 1) &&
                   EmitHelperCallResult(e, be, ins, WJH_COMPARE, uint32_t(jsop)) &&
                   EmitHelperResultAsType(e, be, MIRType::Boolean);
          };
          auto I32C = [&](int32_t v) {
            return e.writeOp(Op::I32Const) && e.writeVarS32(v);
          };
          bool strictEq = (jsop == JSOp::StrictEq || jsop == JSOp::StrictNe);
          if (strictEq && !WJGETENV("GECKO_WJ_NOSPECEQ")) {
            // tag0 == tag1 ?
            if (!pushTag(l0) || !pushTag(l1) || !e.writeOp(Op::I32Eq)) return false;
            if (!e.writeOp(Op::If) || !e.writeFixedU8(uint8_t(TypeCode::I32)))
              return false;
            // value-equality tag (String/BigInt) -> helper; else bits-identity.
            // (Inlining string content-eq here was measured a NET LOSS: neutral for
            // ubo -- its compares are mostly same-length-different-ptr, still needing
            // the helper -- and ~4% WORSE for earley, whose hot `eq?` is symbol/object
            // identity that the bits path already handles; an upfront tag==STRING
            // branch just taxed it. Don't re-add without a cheaper discriminator.)
            if (!pushTag(l0) || !I32C(int32_t(TagWord(JSVAL_TYPE_STRING))) ||
                !e.writeOp(Op::I32Eq) || !pushTag(l0) ||
                !I32C(int32_t(TagWord(JSVAL_TYPE_BIGINT))) || !e.writeOp(Op::I32Eq) ||
                !e.writeOp(Op::I32Or))
              return false;
            if (!e.writeOp(Op::If) || !e.writeFixedU8(uint8_t(TypeCode::I32)))
              return false;
            if (!cmpHelper()) return false;  // String/BigInt value-eq
            if (!e.writeOp(Op::Else)) return false;
            // identity: full boxed-bits equal/not-equal.
            if (!GetLocal(e, uint32_t(l0)) || !GetLocal(e, uint32_t(l1)) ||
                !e.writeOp(jsop == JSOp::StrictEq ? Op::I64Eq : Op::I64Ne))
              return false;
            if (!e.writeOp(Op::End)) return false;  // end value-eq-if
            if (!e.writeOp(Op::Else)) return false;
            // different tags -> strict-equal is false (Eq->0, Ne->1).
            if (!I32C(jsop == JSOp::StrictNe ? 1 : 0)) return false;
            if (!e.writeOp(Op::End)) return false;  // end tagEq-if
            return e.writeOp(Op::End);              // end both-numbers-if
          }
          if (!cmpHelper()) return false;  // loose ==/relational on non-numbers
          return e.writeOp(Op::End);
        }
        if (!EmitStageScratch(e, be, ins->getOperand(0), 0)) return false;
        if (!EmitStageScratch(e, be, ins->getOperand(1), 1)) return false;
        if (!EmitHelperCallResult(e, be, ins, WJH_COMPARE, uint32_t(jsop)))
          return false;
        return EmitHelperResultAsType(e, be, MIRType::Boolean);
      }

      // (B) Generic VM-helper fallback (WJH_BINARYARITH = the *Values VM ops): %, **,
      // bitops, or non-number Add/Sub/Mul/Div not handled inline above. DEFAULT-ON
      // now (was ARITHCACHE-gated): a C++ hop but keeps the FUNCTION in JIT instead
      // of bailing to PBL (crypto bignum %/bitops, raytrace Math.pow). jsop from
      // rp->pc() is correct (the old crash was the trackedSite-pc bug).
      //
      switch (jsop) {
        case JSOp::Add: case JSOp::Sub: case JSOp::Mul: case JSOp::Div:
        case JSOp::Mod: case JSOp::Pow: case JSOp::BitOr: case JSOp::BitAnd:
        case JSOp::BitXor: case JSOp::Lsh: case JSOp::Rsh: case JSOp::Ursh:
          break;
        default: return false;
      }
      if (!EmitStageScratch(e, be, ins->getOperand(0), 0)) return false;
      if (!EmitStageScratch(e, be, ins->getOperand(1), 1)) return false;
      if (!EmitHelperCallResult(e, be, ins, WJH_BINARYARITH, uint32_t(jsop)))
        return false;
      return EmitHelperResultAsType(e, be, ins->type());
    }
    case MDefinition::Opcode::UnaryCache: {
      MResumePoint* rp = ins->resumePoint();
      if (!rp || !rp->pc()) return false;
      JSOp jsop = JSOp(*rp->pc());
      MIRType ty = ins->type();
      int32_t l0 = be.local(ins->getOperand(0));

      // (A) SPECULATIVE INLINE f64 (default-on; see BinaryCache). Neg/Pos/ToNumeric
      // of a Value/Double: unbox the operand as a NUMBER (deopt on non-number) and
      // do the f64 op inline -- typed arith without the (never-warming) IC. Neg ->
      // f64.neg; Pos/ToNumeric on a number is just the unboxed value.
      if ((jsop == JSOp::Neg || jsop == JSOp::Pos || jsop == JSOp::ToNumeric) &&
          l0 >= 0 && (ty == MIRType::Value || ty == MIRType::Double)) {
        if (!EmitUnboxLocal(e, be, uint32_t(l0), MIRType::Double, /*fallible=*/true))
          return false;
        if (jsop == JSOp::Neg && !e.writeOp(Op::F64Neg)) return false;
        if (ty == MIRType::Double) return true;
        return EmitBoxFromStack(e, MIRType::Double);
      }

      // Value-typed Inc/Dec (navier's `i++/j++` loop counters Warp didn't prove
      // Int32). Runtime branch, NO deopt: number -> inline f64 +/-1 (box as a double
      // Value -- semantically a number); non-number -> WJH_UNARYARITH helper. Keeps
      // the hot loop counter inline instead of a per-iteration C++ hop or a bail.
      if ((jsop == JSOp::Inc || jsop == JSOp::Dec) && l0 >= 0 &&
          ty == MIRType::Value) {
        auto pushTag = [&]() -> bool {
          return GetLocal(e, uint32_t(l0)) && e.writeOp(Op::I64Const) &&
                 e.writeVarS64(32) && e.writeOp(Op::I64ShrU) &&
                 e.writeOp(Op::I32WrapI64);
        };
        if (!pushTag() || !e.writeOp(Op::I32Const) ||
            !e.writeVarS32(int32_t(uint32_t(JSVAL_TAG_CLEAR))) ||
            !e.writeOp(Op::I32LeU) || !pushTag() || !e.writeOp(Op::I32Const) ||
            !e.writeVarS32(int32_t(TagWord(JSVAL_TYPE_INT32))) ||
            !e.writeOp(Op::I32Eq) || !e.writeOp(Op::I32Or))
          return false;  // isNumber
        if (!e.writeOp(Op::If) || !e.writeFixedU8(uint8_t(TypeCode::I64)))
          return false;
        // number: int32 -> convert, else reinterpret; +/- 1.0; box double.
        if (!pushTag() || !e.writeOp(Op::I32Const) ||
            !e.writeVarS32(int32_t(TagWord(JSVAL_TYPE_INT32))) ||
            !e.writeOp(Op::I32Eq) || !e.writeOp(Op::If) || !e.writeFixedU8(0x7C))
          return false;
        if (!GetLocal(e, uint32_t(l0)) || !e.writeOp(Op::I32WrapI64) ||
            !e.writeOp(Op::F64ConvertI32S) || !e.writeOp(Op::Else) ||
            !GetLocal(e, uint32_t(l0)) || !e.writeOp(Op::F64ReinterpretI64) ||
            !e.writeOp(Op::End))
          return false;
        if (!e.writeOp(Op::F64Const) || !e.writeFixedF64(1.0) ||
            !e.writeOp(jsop == JSOp::Inc ? Op::F64Add : Op::F64Sub))
          return false;
        if (!EmitBoxFromStack(e, MIRType::Double)) return false;
        if (!e.writeOp(Op::Else)) return false;
        if (!EmitStageScratch(e, be, ins->getOperand(0), 0)) return false;
        if (!EmitHelperCallResult(e, be, ins, WJH_UNARYARITH, uint32_t(jsop)))
          return false;
        return e.writeOp(Op::End);
      }

      // (A2) INLINE i32 for an Int32-typed result (cold loop counters i++/i--,
      // ~x). Operand unboxed as int32 (deopt if not). Inc/Dec overflow-checked;
      // Neg deopts on 0 (-> -0 double) and INT32_MIN (overflow); BitNot is exact.
      if (ty == MIRType::Int32 && l0 >= 0) {
        if (jsop == JSOp::BitNot) {
          if (!EmitUnboxLocal(e, be, uint32_t(l0), MIRType::Int32, true)) return false;
          return e.writeOp(Op::I32Const) && e.writeVarS32(-1) &&
                 e.writeOp(Op::I32Xor);
        }
        if (jsop == JSOp::Inc || jsop == JSOp::Dec) {
          if (!EmitUnboxLocal(e, be, uint32_t(l0), MIRType::Int32, true)) return false;
          if (!e.writeOp(Op::I64ExtendI32S) || !e.writeOp(Op::I64Const) ||
              !e.writeVarS64(jsop == JSOp::Inc ? 1 : -1) || !e.writeOp(Op::I64Add))
            return false;
          if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(be.unboxScratch))
            return false;
          if (!GetLocal(e, be.unboxScratch) || !e.writeOp(Op::I32WrapI64) ||
              !e.writeOp(Op::I64ExtendI32S) || !GetLocal(e, be.unboxScratch) ||
              !e.writeOp(Op::I64Ne))
            return false;
          if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
          if (!EmitDeopt(e, be)) return false;
          if (!e.writeOp(Op::End)) return false;
          return GetLocal(e, be.unboxScratch) && e.writeOp(Op::I32WrapI64);
        }
        if (jsop == JSOp::Neg) {
          // deopt if operand == 0 (Neg(0) = -0, a double) or INT32_MIN (overflow).
          if (!EmitUnboxLocal(e, be, uint32_t(l0), MIRType::Int32, true)) return false;
          if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(be.unboxScratch))
            return false;
          // (op == 0) | (op == INT32_MIN)  -> deopt
          if (!GetLocal(e, be.unboxScratch) || !e.writeOp(Op::I32Eqz))
            return false;
          if (!GetLocal(e, be.unboxScratch) || !e.writeOp(Op::I32Const) ||
              !e.writeVarS32(INT32_MIN) || !e.writeOp(Op::I32Eq) ||
              !e.writeOp(Op::I32Or))
            return false;
          if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
          if (!EmitDeopt(e, be)) return false;
          if (!e.writeOp(Op::End)) return false;
          // 0 - op
          return e.writeOp(Op::I32Const) && e.writeVarS32(0) &&
                 GetLocal(e, be.unboxScratch) && e.writeOp(Op::I32Sub);
        }
      }

      // (B) Generic VM-helper fallback (WJH_UNARYARITH = DoUnaryArithFallback).
      // DEFAULT-ON now: an uncovered unary (e.g. a Value-typed Inc/Dec navier emits)
      // must COMPILE, not bail the whole function to PBL. jsop now comes from
      // rp->pc() (correct) -- the old ARITHCACHE crash was the trackedSite-pc bug
      // giving a wrong jsop.
      switch (jsop) {
        case JSOp::BitNot: case JSOp::Pos: case JSOp::Neg:
        case JSOp::Inc: case JSOp::Dec: case JSOp::ToNumeric:
          break;
        default: return false;
      }
      if (!EmitStageScratch(e, be, ins->getOperand(0), 0)) return false;
      if (!EmitHelperCallResult(e, be, ins, WJH_UNARYARITH, uint32_t(jsop)))
        return false;
      return EmitHelperResultAsType(e, be, ins->type());
    }
    case MDefinition::Opcode::ArrayPush: {
      MArrayPush* a = ins->toArrayPush();
      // INLINE dense-append fast path (Ion parity): arr.push(v) appends at index ==
      // length. For a packed dense array length == initializedLength, so when
      // (initializedLength < capacity) && (length == initializedLength) the push is a
      // pure GC-FREE append into spare capacity -- do it inline (store value, bump
      // initializedLength + length, inline post-write barrier) and return the new
      // length, with NO C++ helper hop / GC safepoint. push is ubiquitous in real code
      // (building result arrays in parsers/decoders/map-filter); the helper hop was a
      // direct Ion-parity gap. Same proven pattern as StoreElementHole's SEHAPPEND.
      // Non-append cases (length != initializedLength: trailing-hole arrays; or
      // initializedLength >= capacity: needs realloc) fall to the WJH_ARRAYPUSH helper
      // (correct, grows in C++). GECKO_WJ_NOARRAYPUSHINLINE reverts to the pure helper.
      int32_t objL = be.local(a->object());
      static int noInline = WJGETENV("GECKO_WJ_NOARRAYPUSHINLINE") ? 1 : 0;
      if (!noInline && objL >= 0) {
        const uint32_t offIL = uint32_t(js::ObjectElements::offsetOfInitializedLength());
        const uint32_t offCap = uint32_t(js::ObjectElements::offsetOfCapacity());
        const uint32_t offLen = uint32_t(js::ObjectElements::offsetOfLength());
        // elements = obj->elements_  (reloaded on each use, like MElements)
        auto elems = [&]() -> bool {
          return GetLocal(e, uint32_t(objL)) && e.writeOp(Op::I32Load) &&
                 e.writeVarU32(2) &&
                 e.writeVarU32(uint32_t(js::NativeObject::offsetOfElements()));
        };
        auto hdr = [&](uint32_t off) -> bool {  // load an i32 ObjectElements header field
          // header offsets (initializedLength/capacity/length) are NEGATIVE relative to
          // the elements pointer, so add them to the address (memarg offset is unsigned).
          return elems() && e.writeOp(Op::I32Const) && e.writeVarS32(int32_t(off)) &&
                 e.writeOp(Op::I32Add) && e.writeOp(Op::I32Load) && e.writeVarU32(2) &&
                 e.writeVarU32(0);
        };
        // cond = (initializedLength <u capacity) & (length == initializedLength)
        if (!hdr(offIL) || !hdr(offCap) || !e.writeOp(Op::I32LtU)) return false;
        if (!hdr(offLen) || !hdr(offIL) || !e.writeOp(Op::I32Eq)) return false;
        if (!e.writeOp(Op::I32And)) return false;
        if (!e.writeOp(Op::If) || !e.writeFixedU8(uint8_t(TypeCode::I32))) return false;
        // append at index == initializedLength: elements[IL*8] = boxed value
        if (!elems() || !hdr(offIL) || !e.writeOp(Op::I32Const) ||
            !e.writeVarS32(int32_t(sizeof(JS::Value))) || !e.writeOp(Op::I32Mul) ||
            !e.writeOp(Op::I32Add))
          return false;
        if (!EmitSpillValue(e, be, a->value())) return false;
        if (!e.writeOp(Op::I64Store) || !e.writeVarU32(3) || !e.writeVarU32(0))
          return false;
        // initializedLength = IL + 1
        if (!elems() || !e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(offIL)) ||
            !e.writeOp(Op::I32Add) || !hdr(offIL) || !e.writeOp(Op::I32Const) ||
            !e.writeVarS32(1) || !e.writeOp(Op::I32Add) || !e.writeOp(Op::I32Store) ||
            !e.writeVarU32(2) || !e.writeVarU32(0))
          return false;
        // length = IL + 1 (length == IL held on the fast path; IL field now IL+1)
        if (!elems() || !e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(offLen)) ||
            !e.writeOp(Op::I32Add) || !hdr(offIL) || !e.writeOp(Op::I32Store) ||
            !e.writeVarU32(2) || !e.writeVarU32(0))
          return false;
        // GC-value append: inline post-write barrier (dedup if Warp emitted one).
        if (WJTypeMaybeGC(a->value()->type()) && !WJHasPostBarrierNode(a->object()) &&
            !EmitForcePostBarrier(e, be, a->object()))
          return false;
        // result (new length) = IL field (now IL+1)
        if (!hdr(offIL)) return false;
        if (!e.writeOp(Op::Else)) return false;
        if (!EmitStageScratch(e, be, a->object(), 0)) return false;
        if (!EmitStageScratch(e, be, a->value(), 1)) return false;
        if (!EmitHelperCallResult(e, be, ins, WJH_ARRAYPUSH, 0) ||
            !e.writeOp(Op::I32WrapI64))
          return false;
        return e.writeOp(Op::End);
      }
      if (!EmitStageScratch(e, be, a->object(), 0)) return false;
      if (!EmitStageScratch(e, be, a->value(), 1)) return false;
      // result is the new length (Int32): unbox the helper's boxed Int32 result.
      return EmitHelperCallResult(e, be, ins, WJH_ARRAYPUSH, 0) &&
             e.writeOp(Op::I32WrapI64);
    }
    case MDefinition::Opcode::ArrayPopShift: {
      MArrayPopShift* a = ins->toArrayPopShift();
      if (!EmitStageScratch(e, be, a->object(), 0)) return false;
      // result is the removed element (Value); site = Pop(0)/Shift(1).
      return EmitHelperCallResult(e, be, ins, WJH_ARRAYPOPSHIFT,
                                  a->mode() == MArrayPopShift::Shift ? 1 : 0);
    }
    case MDefinition::Opcode::MegamorphicLoadSlot: {
      // obj.name where name is a compile-time PropertyKey. Inline the per-site
      // property IC (own data property -> direct slot load, no C++ hop) when the
      // receiver is an object and a site is available; else fall back to the
      // generic helper get.
      MMegamorphicLoadSlot* m = ins->toMegamorphicLoadSlot();
      static int noPropIC = WJGETENV("GECKO_WJ_NOPROPIC") ? 1 : 0;
      if (!noPropIC && m->object()->type() == MIRType::Object) {
        uint32_t site = WJPropSite(be);
        if (site != 0) {
          // knownMega=true: Warp chose MMegamorphicLoadSlot => this site IS
          // megamorphic, so the inline js::MegamorphicCache probe pays off here
          // (unlike mono/poly GetPropertyCache sites -- see EmitPropIC useMega).
          return EmitPropIC(e, be, ins, m->object(), m->name().asRawBits(), site,
                            nullptr, 0, false, /*knownMega=*/true);
        }
      }
      if (!EmitStageScratch(e, be, m->object(), 0)) return false;
      uint64_t nameBits = js::IdToValue(m->name()).asRawBits();
      if (!EmitStageConstBoxed(e, be, nameBits, 1)) return false;
      return EmitHelperCallResult(e, be, ins, WJH_GETPROP, 0);
    }
    case MDefinition::Opcode::MegamorphicLoadSlotByValue: {
      // obj[key] megamorphic. GetElementOperation handles by-value lookups and
      // accessors correctly (a strict superset of the Ion megamorphic fast path),
      // so routing to WJH_GETPROP is sound. op0 = object, op1 = idVal (Value).
      if (WJGETENV("GECKO_WJ_NO_MLSBV")) return false;
      MMegamorphicLoadSlotByValue* mbvM = ins->toMegamorphicLoadSlotByValue();
      auto mbvMiss = [&]() -> bool {
        return EmitStageScratch(e, be, mbvM->object(), 0) &&
               EmitStageScratch(e, be, ins->getOperand(1), 1) &&
               EmitHelperCallResult(e, be, ins, WJH_GETPROP, 0);
      };
      // BY-VALUE MEGA PROBE (GECKO_WJ_MEGABYVAL, opt-in, READ-ONLY): obj[atomKey]
      // hits the per-runtime js::MegamorphicCache inline (shape+key hash -> own-data
      // slot, no C++ hop) -- the real-SPA dynamic-key GETPROP ceiling (discord
      // GETPROP=284K). Falls to WJH_GETPROP on non-Object receiver / non-string or
      // non-atom key / cache miss. Mirrors the const-key emitMegaOrMiss but with a
      // RUNTIME key+hash (idVal must be a String atom; hash read from the atom with
      // an isFatInline branch). No allFull-ways gate: Warp emits this op only when mega.
      // DEFAULT-ON (2026-07-11): validated correct (byval diff JIT==PBL, acorn/diverse)
      // + engaging (obj[k] GETPROP 5.8M->2.4M) + octane-NEUTRAL (best-of-8 earley 1.03/
      // crypto 0.985 = noise; full suite ~1.0). Read-only = safe. GECKO_WJ_NOMEGABYVAL reverts.
      static int mbv = WJGETENV("GECKO_WJ_NOMEGABYVAL") ? 0 : 1;
      JSContext* mbvCx = mbv ? js::TlsContext.get() : nullptr;
      if (mbvCx && mbvM->object()->type() == MIRType::Object) {
        using MC = js::MegamorphicCache;
        using ME = js::MegamorphicCacheEntry;
        uintptr_t cacheAddr =
            uintptr_t(&mbvCx->runtime()->caches().megamorphicCache);
        const uint32_t flagsOff = uint32_t(JSString::offsetOfFlags());
        const int32_t ATOMB = int32_t(js::StringFlags::ATOM_BIT);
        const int32_t FATM = int32_t(js::StringFlags::FAT_INLINE_MASK);
        const uint32_t fatHashOff = uint32_t(js::FatInlineAtom::offsetOfHash());
        const uint32_t normHashOff = uint32_t(js::NormalAtom::offsetOfHash());
        const uint32_t shapeOff = uint32_t(offsetof(JS::shadow::Object, shape));
        const uint32_t slotsOff = uint32_t(js::NativeObject::offsetOfSlots());
        const uint8_t I64T = uint8_t(TypeCode::I64);
        // obj -> propObjLocal ; shape -> propShapeLocal
        if (!GetOp(e, be, mbvM->object()) || !e.writeOp(Op::LocalSet) ||
            !e.writeVarU32(be.propObjLocal) || !GetLocal(e, be.propObjLocal) ||
            !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(shapeOff) ||
            !e.writeOp(Op::LocalSet) || !e.writeVarU32(be.propShapeLocal))
          return false;
        // idVal tag == JSVAL_TAG_STRING ?
        if (!GetOp(e, be, ins->getOperand(1)) || !e.writeOp(Op::I64Const) ||
            !e.writeVarS64(32) || !e.writeOp(Op::I64ShrU) || !e.writeOp(Op::I32WrapI64) ||
            !e.writeOp(Op::I32Const) ||
            !e.writeVarS32(int32_t(uint32_t(JSVAL_TAG_STRING))) || !e.writeOp(Op::I32Eq))
          return false;
        if (!e.writeOp(Op::If) || !e.writeFixedU8(I64T)) return false;  // is-string
        // strptr -> propKeyLocal
        if (!GetOp(e, be, ins->getOperand(1)) || !e.writeOp(Op::I32WrapI64) ||
            !e.writeOp(Op::LocalSet) || !e.writeVarU32(be.propBVKeyLocal))
          return false;
        // atom? flags & ATOM_BIT
        if (!GetLocal(e, be.propBVKeyLocal) || !e.writeOp(Op::I32Load) ||
            !e.writeVarU32(2) || !e.writeVarU32(flagsOff) || !e.writeOp(Op::I32Const) ||
            !e.writeVarS32(ATOMB) || !e.writeOp(Op::I32And))
          return false;
        if (!e.writeOp(Op::If) || !e.writeFixedU8(I64T)) return false;  // is-atom
        // entry ptr -> propTaggedLocal: ((shape>>S1 ^ shape>>S2)+keyHash) & (N-1) *size + base
        if (!GetLocal(e, be.propShapeLocal) || !e.writeOp(Op::I32Const) ||
            !e.writeVarS32(MC::ShapeHashShift1) || !e.writeOp(Op::I32ShrU) ||
            !GetLocal(e, be.propShapeLocal) || !e.writeOp(Op::I32Const) ||
            !e.writeVarS32(MC::ShapeHashShift2) || !e.writeOp(Op::I32ShrU) ||
            !e.writeOp(Op::I32Xor))
          return false;
        // + keyHash: I32Load at propKeyLocal + (isFatInline ? fatHashOff : normHashOff)
        if (!GetLocal(e, be.propBVKeyLocal) || !e.writeOp(Op::I32Const) ||
            !e.writeVarS32(int32_t(fatHashOff)) || !e.writeOp(Op::I32Const) ||
            !e.writeVarS32(int32_t(normHashOff)) || !GetLocal(e, be.propBVKeyLocal) ||
            !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(flagsOff) ||
            !e.writeOp(Op::I32Const) || !e.writeVarS32(FATM) || !e.writeOp(Op::I32And) ||
            !e.writeOp(Op::I32Const) || !e.writeVarS32(FATM) || !e.writeOp(Op::I32Eq) ||
            !e.writeOp(Op::SelectNumeric) || !e.writeOp(Op::I32Add) ||
            !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0) ||
            !e.writeOp(Op::I32Add))
          return false;
        if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(MC::NumEntries - 1)) ||
            !e.writeOp(Op::I32And) || !e.writeOp(Op::I32Const) ||
            !e.writeVarS32(int32_t(sizeof(MC::Entry))) || !e.writeOp(Op::I32Mul) ||
            !e.writeOp(Op::I32Const) ||
            !e.writeVarS32(int32_t(cacheAddr + MC::offsetOfEntries())) ||
            !e.writeOp(Op::I32Add) || !e.writeOp(Op::LocalSet) ||
            !e.writeVarU32(be.propTaggedLocal))
          return false;
        // cond = e.shape==shape & e.key==keyptr & e.gen==gen & e.hopsAndKind==0
        if (!GetLocal(e, be.propTaggedLocal) || !e.writeOp(Op::I32Load) ||
            !e.writeVarU32(2) || !e.writeVarU32(uint32_t(ME::offsetOfShape())) ||
            !GetLocal(e, be.propShapeLocal) || !e.writeOp(Op::I32Eq))
          return false;
        if (!GetLocal(e, be.propTaggedLocal) || !e.writeOp(Op::I32Load) ||
            !e.writeVarU32(2) || !e.writeVarU32(uint32_t(ME::offsetOfKey())) ||
            !GetLocal(e, be.propBVKeyLocal) || !e.writeOp(Op::I32Eq) || !e.writeOp(Op::I32And))
          return false;
        if (!GetLocal(e, be.propTaggedLocal) || !e.writeOp(Op::I32Load16U) ||
            !e.writeVarU32(1) || !e.writeVarU32(uint32_t(ME::offsetOfGeneration())) ||
            !e.writeOp(Op::I32Const) ||
            !e.writeVarS32(int32_t(cacheAddr + MC::offsetOfGeneration())) ||
            !e.writeOp(Op::I32Load16U) || !e.writeVarU32(1) || !e.writeVarU32(0) ||
            !e.writeOp(Op::I32Eq) || !e.writeOp(Op::I32And))
          return false;
        if (!GetLocal(e, be.propTaggedLocal) || !e.writeOp(Op::I32Load8U) ||
            !e.writeVarU32(0) || !e.writeVarU32(uint32_t(ME::offsetOfHopsAndKind())) ||
            !e.writeOp(Op::I32Eqz) || !e.writeOp(Op::I32And))
          return false;
        if (!e.writeOp(Op::If) || !e.writeFixedU8(I64T)) return false;  // hit
        // slotOffset -> propShapeLocal ; base=(tag&1)?obj:slots ; +(tag>>1) ; I64Load
        if (!GetLocal(e, be.propTaggedLocal) || !e.writeOp(Op::I32Load) ||
            !e.writeVarU32(2) || !e.writeVarU32(uint32_t(ME::offsetOfSlotOffset())) ||
            !e.writeOp(Op::LocalSet) || !e.writeVarU32(be.propShapeLocal))
          return false;
        if (!GetLocal(e, be.propObjLocal) || !GetLocal(e, be.propObjLocal) ||
            !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(slotsOff) ||
            !GetLocal(e, be.propShapeLocal) || !e.writeOp(Op::I32Const) ||
            !e.writeVarS32(1) || !e.writeOp(Op::I32And) || !e.writeOp(Op::SelectNumeric))
          return false;
        if (!GetLocal(e, be.propShapeLocal) || !e.writeOp(Op::I32Const) ||
            !e.writeVarS32(1) || !e.writeOp(Op::I32ShrU) || !e.writeOp(Op::I32Add) ||
            !e.writeOp(Op::I64Load) || !e.writeVarU32(3) || !e.writeVarU32(0))
          return false;
        if (!e.writeOp(Op::Else)) return false;  // cache miss
        if (!mbvMiss()) return false;
        if (!e.writeOp(Op::End)) return false;   // close hit-if
        if (!e.writeOp(Op::Else)) return false;  // non-atom
        if (!mbvMiss()) return false;
        if (!e.writeOp(Op::End)) return false;   // close atom-if
        if (!e.writeOp(Op::Else)) return false;  // non-string
        if (!mbvMiss()) return false;
        if (!e.writeOp(Op::End)) return false;   // close string-if
        return true;
      }
      if (!EmitStageScratch(e, be, ins->getOperand(0), 0)) return false;
      if (!EmitStageScratch(e, be, ins->getOperand(1), 1)) return false;
      return EmitHelperCallResult(e, be, ins, WJH_GETPROP, 0);
    }
    case MDefinition::Opcode::MegamorphicHasProp: {
      // `key in obj` (site=0) / own-property check (site=1). op0 = object, op1 = idVal.
      MMegamorphicHasProp* m = ins->toMegamorphicHasProp();
      if (WJGETENV("GECKO_WJ_NO_MHP")) return false;
      if (!EmitStageScratch(e, be, ins->getOperand(0), 0)) return false;
      if (!EmitStageScratch(e, be, ins->getOperand(1), 1)) return false;
      if (!EmitHelperCallResult(e, be, ins, WJH_HASPROP, m->hasOwn() ? 1 : 0))
        return false;
      return EmitHelperResultAsType(e, be, ins->type());  // Boolean
    }
    case MDefinition::Opcode::IsCallable: {
      // operand is ValueOrObject; EmitStageScratch boxes an Object operand. The helper
      // (v.isObject() && v.toObject().isCallable()) is correct (validated JIT==PBL incl
      // primitives). DEFAULT-ON 2026-06-30: the LATENT ubo miscompile it used to expose
      // (hntrieContainer undefined) was the self-hosted ObjectOrReflectDefineProperty
      // force-mega recompile bug -- FIXED by the self-hosted->PBL-on-storm valve guard
      // (WasmJitRuntime SHPBL). Validated: ubo OK + octane/realapp clean. GECKO_WJ_NOISCALL reverts.
      if (WJGETENV("GECKO_WJ_NOISCALL")) return false;
      if (!EmitStageScratch(e, be, ins->getOperand(0), 0)) return false;
      if (!EmitHelperCallResult(e, be, ins, WJH_ISCALLABLE, 0)) return false;
      return EmitHelperResultAsType(e, be, ins->type());  // Boolean
    }
    case MDefinition::Opcode::NumberParseInt: {
      // parseInt(string, radix). op0 = string, op1 = radix (Int32). Both staged boxed.
      if (WJGETENV("GECKO_WJ_NO_PARSEINT")) return false;
      if (!EmitStageScratch(e, be, ins->getOperand(0), 0)) return false;
      if (!EmitStageScratch(e, be, ins->getOperand(1), 1)) return false;
      return EmitHelperCallResult(e, be, ins, WJH_PARSEINT, 0);  // result Value
    }
    case MDefinition::Opcode::ObjectToIterator: {
      if (WJGETENV("GECKO_WJ_NO_OBJITER")) return false;
      if (!EmitStageScratch(e, be, ins->getOperand(0), 0)) return false;
      if (!EmitHelperCallResult(e, be, ins, WJH_OBJTOITER, 0)) return false;
      return EmitHelperResultAsType(e, be, ins->type());  // Object
    }
    case MDefinition::Opcode::GuardStringToIndex: {
      // Compute the canonical array index of the string; deopt if it is not one
      // (negative). Mirrors visitGuardStringToIndex (GetIndexFromString + bailout).
      if (WJGETENV("GECKO_WJ_NO_STRIDX")) return false;
      int32_t l = be.local(ins);
      if (l < 0) return false;
      if (!EmitStageScratch(e, be, ins->getOperand(0), 0)) return false;
      if (!EmitHelperCallResult(e, be, ins, WJH_STRTOINDEX, 0)) return false;
      if (!EmitHelperResultAsType(e, be, ins->type())) return false;  // Int32
      if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(uint32_t(l))) return false;
      if (!GetLocal(e, uint32_t(l)) || !e.writeOp(Op::I32Const) ||
          !e.writeVarS32(0) || !e.writeOp(Op::I32LtS))
        return false;
      if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
      if (!EmitDeopt(e, be)) return false;
      if (!e.writeOp(Op::End)) return false;
      return GetLocal(e, uint32_t(l));
    }
    case MDefinition::Opcode::InstanceOfCache: {
      // `obj instanceof ctor`: ctor is operand 1. Result is a boolean (i32);
      // unbox the boxed BooleanValue the helper returns.
      if (!EmitStageScratch(e, be, ins->getOperand(0), 0)) return false;
      if (!EmitStageScratch(e, be, ins->getOperand(1), 1)) return false;
      if (!EmitHelperCallResult(e, be, ins, WJH_INSTANCEOF, 0)) return false;
      return e.writeOp(Op::I32WrapI64);
    }
    case MDefinition::Opcode::InstanceOf: {
      // `obj instanceof Foo` with resolved proto: operand 0 = obj, operand 1 =
      // Foo.prototype. Helper does IsPrototypeOf (correct). Default-ON.
      if (WJGETENV("GECKO_WJ_NOINSTANCEOFJIT")) return false;
      // INLINE FAST PATH: a NON-OBJECT receiver is never an instance -> false, with
      // NO C++ helper hop (earley does 19.5M `x instanceof sc_Pair` and many `x` are
      // non-object atoms -- symbols/numbers/strings). Only when obj is Value-typed
      // (could be non-object). Object receivers still go to the helper.
      int32_t ol = be.local(ins->getOperand(0));
      // Object-receiver case: 1-HOP PROTO INLINE -- if obj's immediate proto ==
      // Foo.prototype, obj IS an instance (true) with NO helper hop (earley's 23.5M
      // `x instanceof sc_Pair` are exactly this 1-hop case). Else fall to the helper
      // (full IsPrototypeOf chain-walk -- correct for multi-hop/lazy/proxy). A lazy
      // proto reads as magic 0x1 != the proto ptr -> helper, so this is sound.
      //
      constexpr int noInstofInline = 0;  // inline instanceof is permanent
      uint32_t shapeOff = uint32_t(offsetof(JS::shadow::Object, shape));
      uint32_t baseOff = uint32_t(js::Shape::offsetOfBaseShape());
      uint32_t protoOff = uint32_t(js::BaseShape::offsetOfProto());
      auto pushObjPtr = [&]() -> bool {
        if (ins->getOperand(0)->type() == MIRType::Object) return GetOp(e, be, ins->getOperand(0));
        return GetLocal(e, uint32_t(ol)) && e.writeOp(Op::I32WrapI64);  // confirmed-object low32
      };
      auto emitHelperInstof = [&]() -> bool {  // -> i32 (0/1) on stack
        return EmitStageScratch(e, be, ins->getOperand(0), 0) &&
               EmitStageScratch(e, be, ins->getOperand(1), 1) &&
               EmitHelperCallResult(e, be, ins, WJH_INSTANCEOFPROTO, 0) &&
               e.writeOp(Op::I32WrapI64);
      };
      auto emitProtoLoad = [&](bool fromObj) -> bool {  // load ->shape->base->proto
        if (fromObj) { if (!pushObjPtr()) return false; }
        else if (!GetLocal(e, be.instofProto)) return false;
        return e.writeOp(Op::I32Load) && e.writeVarU32(2) && e.writeVarU32(shapeOff) &&
               e.writeOp(Op::I32Load) && e.writeVarU32(2) && e.writeVarU32(baseOff) &&
               e.writeOp(Op::I32Load) && e.writeVarU32(2) && e.writeVarU32(protoOff);
      };
      static int multihop = WJGETENV("GECKO_WJ_NOINSTOFMULTIHOP") ? 0 : 1;
      auto emitObjInstof = [&]() -> bool {
        int32_t pl = be.local(ins->getOperand(1));
        if (noInstofInline || pl < 0) return emitHelperInstof();  // pure helper
        if (multihop) {
          // BOUNDED MULTI-HOP proto walk (terser's deep AST hierarchy: `x instanceof
          // AST_Node` is many hops). instofProto = obj's proto; loop: ==Foo.proto->true;
          // null(0)->false; lazy(1) or >kMaxHops->helper; else deref next proto. The
          // p>=2 guard (null/lazy checked first) makes the deref sound. DEFAULT-ON
          // (GECKO_WJ_NOINSTOFMULTIHOP reverts to 1-hop+helper): Pareto vs 1-hop --
          // replaces the C++ helper hop with <=8 inline proto derefs, exits after 1
          // iteration on a 1-hop hit (earley 23.5M 1-hop instanceof no-regress),
          // helper only beyond kMaxHops or on lazy/proxy proto. Same soundness as the
          // 1-hop inline, iterated. terser 1.20->1.25 (toward PBL 1.27).
          const int32_t kMaxHops = 8;
          if (!emitProtoLoad(true) || !e.writeOp(Op::LocalSet) ||
              !e.writeVarU32(be.instofProto))
            return false;
          if (!e.writeOp(Op::I32Const) || !e.writeVarS32(0) ||
              !e.writeOp(Op::LocalSet) || !e.writeVarU32(be.instofCount))
            return false;
          if (!e.writeOp(Op::Block) || !e.writeFixedU8(0x40)) return false;  // $exit
          if (!e.writeOp(Op::Loop) || !e.writeFixedU8(0x40)) return false;   // $L
          auto brk = [&]() -> bool {  // value on stack -> instofResult; br to $exit
            return e.writeOp(Op::LocalSet) && e.writeVarU32(be.instofResult) &&
                   e.writeOp(Op::Br) && e.writeVarU32(2);
          };
          // instofProto == Foo.proto -> 1
          if (!GetLocal(e, be.instofProto) || !GetLocal(e, uint32_t(pl)) ||
              !e.writeOp(Op::I32Eq) || !e.writeOp(Op::If) || !e.writeFixedU8(0x40))
            return false;
          if (!e.writeOp(Op::I32Const) || !e.writeVarS32(1) || !brk() ||
              !e.writeOp(Op::End))
            return false;
          // instofProto == 0 (null) -> 0
          if (!GetLocal(e, be.instofProto) || !e.writeOp(Op::I32Eqz) ||
              !e.writeOp(Op::If) || !e.writeFixedU8(0x40))
            return false;
          if (!e.writeOp(Op::I32Const) || !e.writeVarS32(0) || !brk() ||
              !e.writeOp(Op::End))
            return false;
          // instofProto == 1 (lazy proto) OR count >= kMaxHops -> helper
          if (!GetLocal(e, be.instofProto) || !e.writeOp(Op::I32Const) ||
              !e.writeVarS32(1) || !e.writeOp(Op::I32Eq) ||
              !GetLocal(e, be.instofCount) || !e.writeOp(Op::I32Const) ||
              !e.writeVarS32(kMaxHops) || !e.writeOp(Op::I32GeU) ||
              !e.writeOp(Op::I32Or) || !e.writeOp(Op::If) || !e.writeFixedU8(0x40))
            return false;
          if (!emitHelperInstof() || !brk() || !e.writeOp(Op::End)) return false;
          // walk: instofProto = instofProto->shape->base->proto ; count++ ; continue
          if (!emitProtoLoad(false) || !e.writeOp(Op::LocalSet) ||
              !e.writeVarU32(be.instofProto))
            return false;
          if (!GetLocal(e, be.instofCount) || !e.writeOp(Op::I32Const) ||
              !e.writeVarS32(1) || !e.writeOp(Op::I32Add) ||
              !e.writeOp(Op::LocalSet) || !e.writeVarU32(be.instofCount))
            return false;
          if (!e.writeOp(Op::Br) || !e.writeVarU32(0)) return false;  // continue $L
          if (!e.writeOp(Op::End)) return false;  // end loop
          if (!e.writeOp(Op::End)) return false;  // end block $exit
          return GetLocal(e, be.instofResult);
        }
        // 1-HOP: proto1 = obj->shape->base->proto ; if (proto1 == Foo.prototype) -> 1, else helper
        if (!emitProtoLoad(true) || !GetLocal(e, uint32_t(pl)) || !e.writeOp(Op::I32Eq))
          return false;
        if (!e.writeOp(Op::If) || !e.writeFixedU8(uint8_t(TypeCode::I32))) return false;
        if (!e.writeOp(Op::I32Const) || !e.writeVarS32(1)) return false;  // 1-hop -> true
        if (!e.writeOp(Op::Else)) return false;
        if (!emitHelperInstof()) return false;
        return e.writeOp(Op::End);
      };
      bool fast = ol >= 0 && ins->getOperand(0)->type() == MIRType::Value &&
                  true;
      if (fast) {
        // tag(obj) == Object ?
        if (!GetLocal(e, uint32_t(ol)) || !e.writeOp(Op::I64Const) ||
            !e.writeVarS64(32) || !e.writeOp(Op::I64ShrU) ||
            !e.writeOp(Op::I32WrapI64) || !e.writeOp(Op::I32Const) ||
            !e.writeVarS32(int32_t(TagWord(JSVAL_TYPE_OBJECT))) ||
            !e.writeOp(Op::I32Eq))
          return false;
        if (!e.writeOp(Op::If) || !e.writeFixedU8(uint8_t(TypeCode::I32)))
          return false;
        if (!emitObjInstof()) return false;  // object -> 1-hop inline or helper
        if (!e.writeOp(Op::Else)) return false;
        if (!e.writeOp(Op::I32Const) || !e.writeVarS32(0)) return false;  // non-object -> false
        return e.writeOp(Op::End);
      }
      return emitObjInstof();
    }
    case MDefinition::Opcode::Lambda: {
      // Clone a closure over the current env. Helper matches Ion's OOL fallback
      // (LambdaOptimizedFallback). The env to capture is the environmentChain()
      // operand, staged via GetOp -- which now resolves an INLINED frame's
      // FunctionEnvironment correctly (EmitInlinedFEEnv, the inlined-FE fix), so
      // inlined lambdas capture the right scope. (Was bailed because pre-FE-fix the
      // inlined env was the outer fn's -> wrong scope.) GECKO_WJ_NOLAMBDA disables;
      // GECKO_WJ_NOINLINELAMBDA reverts inlined lambdas to the PBL bail.
      if (WJGETENV("GECKO_WJ_NOLAMBDA")) return false;
      bool lamInlined = ins->block()->info().script() != be.info->script();
      if (lamInlined && WJGETENV("GECKO_WJ_NOINLINELAMBDA")) return false;
      MLambda* l = ins->toLambda();
      if (!EmitStageScratch(e, be, l->environmentChain(), 0)) return false;
      if (!EmitStageScratch(e, be, l->getOperand(1), 1)) return false;  // template fn
      if (!EmitHelperCallResult(e, be, ins, WJH_LAMBDA, 0)) return false;
      return e.writeOp(Op::I32WrapI64);  // boxed Object -> ptr
    }
    case MDefinition::Opcode::TypeOfIs: {
      // (typeof operand jsop "typename"). The single-tag types + number reduce to a
      // NaN-box TAG test on the value -> inline it (earley does 2.2M of these as a
      // helper hop). Object/Function need the callable/class check -> WHJ_TYPEOFIS
      // helper (TypeOfValue). GECKO_WJ_NOTYPEOFINLINE reverts to the pure helper.
      if (WJGETENV("GECKO_WJ_NOTYPEOFIS")) return false;
      MTypeOfIs* t = ins->toTypeOfIs();
      bool invert = (t->jsop() == JSOp::Ne || t->jsop() == JSOp::StrictNe);
      static int noTypeofInline = WJGETENV("GECKO_WJ_NOTYPEOFINLINE") ? 1 : 0;
      JSValueType vt = JSVAL_TYPE_OBJECT;
      bool isNum = false, inlineable = true;
      switch (t->jstype()) {
        case JSTYPE_STRING: vt = JSVAL_TYPE_STRING; break;
        case JSTYPE_BOOLEAN: vt = JSVAL_TYPE_BOOLEAN; break;
        case JSTYPE_UNDEFINED: vt = JSVAL_TYPE_UNDEFINED; break;
        case JSTYPE_SYMBOL: vt = JSVAL_TYPE_SYMBOL; break;
        case JSTYPE_BIGINT: vt = JSVAL_TYPE_BIGINT; break;
        case JSTYPE_NUMBER: isNum = true; break;
        default: inlineable = false; break;  // OBJECT, FUNCTION -> callable check
      }
      if (!noTypeofInline && inlineable) {
        if (!GetOp(e, be, t->input())) return false;
        if (t->input()->type() != MIRType::Value &&
            !EmitBoxFromStack(e, t->input()->type()))
          return false;
        if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(be.unboxScratch)) return false;
        auto pushTag = [&]() -> bool {  // high-32 tag word of the boxed value
          return GetLocal(e, be.unboxScratch) && e.writeOp(Op::I64Const) &&
                 e.writeVarS64(32) && e.writeOp(Op::I64ShrU) && e.writeOp(Op::I32WrapI64);
        };
        if (isNum) {  // (tag u<= CLEAR) | (tag == INT32)
          if (!pushTag() || !e.writeOp(Op::I32Const) ||
              !e.writeVarS32(int32_t(uint32_t(JSVAL_TAG_CLEAR))) || !e.writeOp(Op::I32LeU) ||
              !pushTag() || !e.writeOp(Op::I32Const) ||
              !e.writeVarS32(int32_t(TagWord(JSVAL_TYPE_INT32))) || !e.writeOp(Op::I32Eq) ||
              !e.writeOp(Op::I32Or))
            return false;
        } else if (!pushTag() || !e.writeOp(Op::I32Const) ||
                   !e.writeVarS32(int32_t(TagWord(vt))) || !e.writeOp(Op::I32Eq)) {
          return false;
        }
        if (invert && !e.writeOp(Op::I32Eqz)) return false;  // != : negate
        return true;
      }
      if (!EmitStageScratch(e, be, t->input(), 0)) return false;
      uint32_t packed = (uint32_t(t->jstype()) << 1) | (invert ? 1u : 0u);
      if (!EmitHelperCallResult(e, be, ins, WJH_TYPEOFIS, packed)) return false;
      return e.writeOp(Op::I32WrapI64);  // boxed Boolean -> i32
    }
    case MDefinition::Opcode::NewCallObject: {
      // Allocate a CallObject (closure scope) via the traced shared shape; the
      // enclosing-env/callee slots are filled by subsequent StoreFixedSlot ops
      // (matches Ion's visitNewCallObject -> CallObject::createWithShape). Lets
      // earley/Boyer closure-creating functions fully compile (they create a
      // scope object then Lambda over it). GECKO_WJ_NONEWCALLOBJ disables.
      if (WJGETENV("GECKO_WJ_NONEWCALLOBJ")) return false;
      MNewCallObject* n = ins->toNewCallObject();
      js::CallObject* tmpl = n->templateObject();
      if (!tmpl) return false;
      uintptr_t shapeAddr = WJInternShape(uintptr_t(tmpl->sharedShape()));
      if (!shapeAddr) return false;
      auto stG = [&](void* g, int32_t v) -> bool {
        return e.writeOp(Op::I32Const) && e.writeVarS32(int32_t(uintptr_t(g))) &&
               e.writeOp(Op::I32Const) && e.writeVarS32(v) &&
               e.writeOp(Op::I32Store) && e.writeVarU32(2) && e.writeVarU32(0);
      };
      if (!stG(&gWJNewShapeSlot, int32_t(shapeAddr)) ||
          !stG(&gWJNewHeap, int32_t(uint32_t(n->initialHeap()))))
        return false;
      if (!EmitHelperCallResult(e, be, ins, WJH_NEWCALLOBJ, 0)) return false;
      return e.writeOp(Op::I32WrapI64);  // boxed Object -> ptr
    }
    case MDefinition::Opcode::CreateArgumentsObject: {
      // Materialize `arguments` (block-0 op). Stage all kWJMaxArgs actuals (wasm
      // params kWJArg0Param+i; caller pads beyond argc), the runtime callee
      // (gWJCallRoots[envRootIdx+1]), the env (gWJCallRoots[envRootIdx]) and the
      // argc snapshot, then WJH_NEWARGUMENTS -> ArgumentsObject::createForWasmJit.
      // Only the outer running frame (inlined frames' args aren't staged here).
      // DEFAULT-ON again (GECKO_WJ_NOARGSOBJ disables). The >kWJMaxArgs-actuals
      // correctness bug (object sized to argc but only 8 actuals staged -> elements
      // >=8 read scratch metadata as garbage; browser NaN) is now handled by the
      // function-ENTRY argc>kWJMaxArgs guard (see the prologue `needsArgcGuard` flag-2
      // return): the body only ever runs with actualArgc<=kWJMaxArgs, so this staging
      // (8 params -> scratch[0..7], metadata at 8/9/10) is correct with no collision.
      // Wider calls return flag-2 "did not run" -> PBL (reliable, no deopt valve).
      if (WJGETENV("GECKO_WJ_NOARGSOBJ")) return false;
      if (ins->block()->info().script() != be.info->script()) return false;
      if (!be.usesArgc || !be.usesCallee || !be.useEnvRoot) return false;
      if (be.info && be.info->nargs() > kWJMaxArgs) return false;
      // MAPPED-ARGS ALIASING (bug-917585): for a non-strict simple-param fn with a
      // closed-over (aliased) formal, arguments[i] FORWARDS to the CallObject slot so
      // a post-creation formal mutation is observed. Our args-object is a SNAPSHOT of
      // the actuals at creation -> it can't track that forwarding -> wrong result.
      // Bail exactly that case to PBL (correct). Fns using `arguments` with no
      // aliased formal (the common case: uglify etc.) keep the JIT snapshot.
      if (be.info && be.info->script() &&
          be.info->script()->anyFormalIsForwarded())
        return false;
      uintptr_t rootsBase = uintptr_t(static_cast<void*>(&gWJCallRoots[0]));
      auto emitEnvRootLoad = [&](uint32_t byteOff) -> bool {
        // push boxed gWJCallRoots[envRootIdx] at byteOff (0=env, 8=callee)
        return e.writeOp(Op::LocalGet) && e.writeVarU32(be.envRootIdx) &&
               e.writeOp(Op::I32Const) && e.writeVarS32(8) && e.writeOp(Op::I32Mul) &&
               e.writeOp(Op::I32Const) && e.writeVarS32(int32_t(rootsBase)) &&
               e.writeOp(Op::I32Add) && e.writeOp(Op::I64Load) && e.writeVarU32(3) &&
               e.writeVarU32(byteOff);
      };
      for (uint32_t i = 0; i < kWJMaxArgs; i++) {  // scratch[i] = actual arg i
        if (!GetLocal(e, be.scratchBase)) return false;
        if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(kWJArg0Param + i)) return false;
        if (!e.writeOp(Op::I64Store) || !e.writeVarU32(3) || !e.writeVarU32(i * 8))
          return false;
      }
      if (!GetLocal(e, be.scratchBase) || !emitEnvRootLoad(8) ||  // scratch[8]=callee
          !e.writeOp(Op::I64Store) || !e.writeVarU32(3) || !e.writeVarU32(8 * 8))
        return false;
      if (!GetLocal(e, be.scratchBase) || !emitEnvRootLoad(0) ||  // scratch[9]=env
          !e.writeOp(Op::I64Store) || !e.writeVarU32(3) || !e.writeVarU32(9 * 8))
        return false;
      if (!GetLocal(e, be.scratchBase) ||                          // scratch[10]=argc (raw)
          !GetLocal(e, be.actualArgcLocal) || !e.writeOp(Op::I64ExtendI32U) ||
          !e.writeOp(Op::I64Store) || !e.writeVarU32(3) || !e.writeVarU32(10 * 8))
        return false;
      if (!EmitHelperCallResult(e, be, ins, WJH_NEWARGUMENTS, 0)) return false;
      return EmitHelperResultAsType(e, be, jit::MIRType::Object);
    }
    case MDefinition::Opcode::NewNamedLambdaObject: {
      // Named-lambda scope for a named function expression. Its callee is the
      // running function -> stage the runtime callee (gWJCallRoots[envRootIdx+1])
      // to scratch[0] and call NamedLambdaObject::createWithoutEnclosing (enclosing
      // set by a later store, mirroring Ion).
      if (ins->block()->info().script() != be.info->script()) return false;
      if (!be.usesCallee || !be.useEnvRoot) return false;
      uintptr_t rootsBase = uintptr_t(static_cast<void*>(&gWJCallRoots[0]));
      if (!GetLocal(e, be.scratchBase)) return false;
      // boxed gWJCallRoots[envRootIdx] at byteOff 8 = the runtime callee
      if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(be.envRootIdx) ||
          !e.writeOp(Op::I32Const) || !e.writeVarS32(8) || !e.writeOp(Op::I32Mul) ||
          !e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(rootsBase)) ||
          !e.writeOp(Op::I32Add) || !e.writeOp(Op::I64Load) || !e.writeVarU32(3) ||
          !e.writeVarU32(8))
        return false;
      if (!e.writeOp(Op::I64Store) || !e.writeVarU32(3) || !e.writeVarU32(0))
        return false;
      if (!EmitHelperCallResult(e, be, ins, WJH_NEWNAMEDLAMBDA, 0)) return false;
      return EmitHelperResultAsType(e, be, jit::MIRType::Object);
    }
    case MDefinition::Opcode::GetArgumentsObjectArg: {
      // arguments[argno] (const index) from the materialized object: load the
      // ArgumentsData ptr (PrivateValue in DATA_SLOT), then the boxed Value at
      // args[argno]. Mirrors Ion's visitGetArgumentsObjectArg.
      MGetArgumentsObjectArg* g = ins->toGetArgumentsObjectArg();
      if (!GetOp(e, be, g->argsObject())) return false;             // [obj]
      if (!e.writeOp(Op::I64Load) || !e.writeVarU32(3) ||
          !e.writeVarU32(uint32_t(js::ArgumentsObject::getDataSlotOffset())))
        return false;                                               // [data_private i64]
      if (!e.writeOp(Op::I32WrapI64)) return false;                 // [data ptr]
      uint32_t off = uint32_t(js::ArgumentsData::offsetOfArgs()) +
                     uint32_t(g->argno()) * 8;
      if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(off)) ||
          !e.writeOp(Op::I32Add))
        return false;                                               // [argAddr]
      return e.writeOp(Op::I64Load) && e.writeVarU32(3) && e.writeVarU32(0);  // [Value]
    }
    case MDefinition::Opcode::ArgumentsObjectLength: {
      // arguments.length from the materialized object: INITIAL_LENGTH_SLOT holds
      // (length << PACKED_BITS_COUNT) | flags; deopt if LENGTH_OVERRIDDEN_BIT set
      // (Ion bails), else length = word >> PACKED_BITS_COUNT. Mirrors Ion's
      // loadArgumentsObjectLength.
      MArgumentsObjectLength* a = ins->toArgumentsObjectLength();
      if (!GetOp(e, be, a->argsObject())) return false;             // [obj]
      if (!e.writeOp(Op::I64Load) || !e.writeVarU32(3) ||
          !e.writeVarU32(uint32_t(js::ArgumentsObject::getInitialLengthSlotOffset())))
        return false;                                               // [word_boxed i64]
      if (!e.writeOp(Op::I32WrapI64)) return false;                 // [word i32]
      if (!e.writeOp(Op::LocalTee) || !e.writeVarU32(be.propShapeLocal))
        return false;                                               // [word], propShapeLocal=word
      if (!e.writeOp(Op::I32Const) ||
          !e.writeVarS32(int32_t(js::ArgumentsObject::LENGTH_OVERRIDDEN_BIT)) ||
          !e.writeOp(Op::I32And))
        return false;                                               // [overridden?]
      if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
      if (!EmitDeopt(e, be)) return false;
      if (!e.writeOp(Op::End)) return false;
      if (!GetLocal(e, be.propShapeLocal)) return false;            // [word]
      return e.writeOp(Op::I32Const) &&
             e.writeVarS32(int32_t(js::ArgumentsObject::PACKED_BITS_COUNT)) &&
             e.writeOp(Op::I32ShrU);                                // [length]
    }
    case MDefinition::Opcode::LoadArgumentsObjectArg: {
      // arguments[index] (dynamic) from the materialized object. Mirrors Ion's
      // loadArgumentsObjectElement: deopt if ELEMENT_OVERRIDDEN_BIT set, bounds
      // check (index>=length), then load args[index]; deopt if it's a MAGIC slot
      // (a mapped formal forwarded to the CallObject -- PBL reads it correctly).
      MLoadArgumentsObjectArg* l = ins->toLoadArgumentsObjectArg();
      if (!GetOp(e, be, l->argsObject())) return false;             // [obj]
      if (!e.writeOp(Op::I64Load) || !e.writeVarU32(3) ||
          !e.writeVarU32(uint32_t(js::ArgumentsObject::getInitialLengthSlotOffset())))
        return false;                                               // [word i64]
      if (!e.writeOp(Op::I32WrapI64)) return false;                 // [word]
      if (!e.writeOp(Op::LocalTee) || !e.writeVarU32(be.propShapeLocal))
        return false;                                               // [word], propShape=word
      if (!e.writeOp(Op::I32Const) ||
          !e.writeVarS32(int32_t(js::ArgumentsObject::ELEMENT_OVERRIDDEN_BIT)) ||
          !e.writeOp(Op::I32And))
        return false;                                               // [overridden?]
      if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
      if (!EmitDeopt(e, be)) return false;
      if (!e.writeOp(Op::End)) return false;
      // propShapeLocal = length = word >> PACKED_BITS_COUNT
      if (!GetLocal(e, be.propShapeLocal) || !e.writeOp(Op::I32Const) ||
          !e.writeVarS32(int32_t(js::ArgumentsObject::PACKED_BITS_COUNT)) ||
          !e.writeOp(Op::I32ShrU) || !e.writeOp(Op::LocalSet) ||
          !e.writeVarU32(be.propShapeLocal))
        return false;
      // bounds: (unsigned) index >= length -> deopt
      if (!GetOp(e, be, l->index())) return false;                  // [index]
      if (!GetLocal(e, be.propShapeLocal) || !e.writeOp(Op::I32GeU))
        return false;                                               // [oob?]
      if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
      if (!EmitDeopt(e, be)) return false;
      if (!e.writeOp(Op::End)) return false;
      // argValue = *(data + offsetOfArgs() + index*8)
      if (!GetOp(e, be, l->argsObject())) return false;             // [obj]
      if (!e.writeOp(Op::I64Load) || !e.writeVarU32(3) ||
          !e.writeVarU32(uint32_t(js::ArgumentsObject::getDataSlotOffset())))
        return false;                                               // [data_priv i64]
      if (!e.writeOp(Op::I32WrapI64)) return false;                 // [data]
      if (!GetOp(e, be, l->index()) || !e.writeOp(Op::I32Const) ||
          !e.writeVarS32(8) || !e.writeOp(Op::I32Mul) || !e.writeOp(Op::I32Add))
        return false;                                               // [data + index*8]
      if (!e.writeOp(Op::I32Const) ||
          !e.writeVarS32(int32_t(js::ArgumentsData::offsetOfArgs())) ||
          !e.writeOp(Op::I32Add))
        return false;                                               // [argAddr]
      if (!e.writeOp(Op::I64Load) || !e.writeVarU32(3) || !e.writeVarU32(0))
        return false;                                               // [argValue i64]
      if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(be.unboxScratch))
        return false;                                               // unboxScratch=argValue
      // magic (forwarded-to-CallObject) slot -> deopt
      if (!GetLocal(e, be.unboxScratch) || !e.writeOp(Op::I64Const) ||
          !e.writeVarS64(32) || !e.writeOp(Op::I64ShrU) || !e.writeOp(Op::I32WrapI64) ||
          !e.writeOp(Op::I32Const) ||
          !e.writeVarS32(int32_t(TagWord(JSVAL_TYPE_MAGIC))) || !e.writeOp(Op::I32Eq))
        return false;                                               // [isMagic?]
      if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
      if (!EmitDeopt(e, be)) return false;
      if (!e.writeOp(Op::End)) return false;
      return GetLocal(e, be.unboxScratch);                          // [argValue] (Value)
    }
    case MDefinition::Opcode::InArray: {
      // `index in array` dense fast path -> Boolean. operands: elements(Elements ptr),
      // index(Int32), initLength(Int32). Mirrors Ion visitInArray: in-bounds AND the
      // element is not a HOLE (JS_ELEMENTS_HOLE magic) -> true. Out-of-bounds -> false,
      // EXCEPT a NEGATIVE index deopts when needsNegativeIntCheck (Ion bails; `-1 in a`
      // is a real property lookup PBL handles).
      auto* a = ins->toInArray();
      // (unsigned) index >= initLength ?  (also true for a negative index)
      if (!GetOp(e, be, a->index()) || !GetOp(e, be, a->initLength()) ||
          !e.writeOp(Op::I32GeU))
        return false;
      if (!e.writeOp(Op::If) || !e.writeFixedU8(0x7F)) return false;  // result i32
      // out of bounds: deopt on a genuine negative index (needsNegativeIntCheck), else false
      if (a->needsNegativeIntCheck()) {
        if (!GetOp(e, be, a->index()) || !e.writeOp(Op::I32Const) ||
            !e.writeVarS32(0) || !e.writeOp(Op::I32LtS))
          return false;
        if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
        if (!EmitDeopt(e, be)) return false;
        if (!e.writeOp(Op::End)) return false;
      }
      if (!e.writeOp(Op::I32Const) || !e.writeVarS32(0)) return false;  // false
      if (!e.writeOp(Op::Else)) return false;
      // in bounds: value = *(elements + index*8); present = (tag != MAGIC)
      if (!GetOp(e, be, a->elements()) || !GetOp(e, be, a->index()) ||
          !e.writeOp(Op::I32Const) || !e.writeVarS32(8) || !e.writeOp(Op::I32Mul) ||
          !e.writeOp(Op::I32Add) || !e.writeOp(Op::I64Load) || !e.writeVarU32(3) ||
          !e.writeVarU32(0))
        return false;  // [value i64]
      if (!e.writeOp(Op::I64Const) || !e.writeVarS64(32) || !e.writeOp(Op::I64ShrU) ||
          !e.writeOp(Op::I32WrapI64) || !e.writeOp(Op::I32Const) ||
          !e.writeVarS32(int32_t(TagWord(JSVAL_TYPE_MAGIC))) || !e.writeOp(Op::I32Ne))
        return false;  // tag != MAGIC -> present(1) ; hole -> 0
      return e.writeOp(Op::End);
    }
    case MDefinition::Opcode::CheckThis: {
      // `this` TDZ guard in a derived-class constructor: throw
      // ThrowUninitializedThis if thisValue is MagicValue(JS_UNINITIALIZED_LEXICAL)
      // (used before super()). The only magic `this` can hold here is that, so
      // deopt-to-PBL on ANY magic tag (PBL's CheckThis throws). Passthrough else.
      // GECKO_WJ_NOCHECKTHIS reverts.
      if (WJGETENV("GECKO_WJ_NOCHECKTHIS")) return false;
      MDefinition* tv = ins->toCheckThis()->thisValue();
      if (!GetOp(e, be, tv)) return false;                  // thisValue (i64)
      if (!e.writeOp(Op::I64Const) || !e.writeVarS64(32) ||
          !e.writeOp(Op::I64ShrU) || !e.writeOp(Op::I32WrapI64))
        return false;                                       // tag = value>>32
      if (!e.writeOp(Op::I32Const) ||
          !e.writeVarS32(int32_t(uint32_t(JSVAL_TAG_MAGIC))) || !e.writeOp(Op::I32Eq))
        return false;                                       // tag == MAGIC ?
      if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
      if (!EmitDeopt(e, be)) return false;
      if (!e.writeOp(Op::End)) return false;
      return GetOp(e, be, tv);                              // passthrough (Value)
    }
    case MDefinition::Opcode::GuardIsNativeObject: {
      // Passthrough the object; deopt if it is NOT a native object. Mirrors Ion's
      // branchIfNonNativeObj: load obj->shape, test shape->immutableFlags &
      // isNativeBit; (flags & bit)==0 -> non-native -> deopt to PBL. Non-effectful,
      // alias_set:none (a native object never becomes non-native). Common on real
      // code before megamorphic element/slot access on maybe-non-native receivers.
      // GECKO_WJ_NOGUARDNATIVE reverts (bail whole fn to PBL).
      if (WJGETENV("GECKO_WJ_NOGUARDNATIVE")) return false;
      MDefinition* obj = ins->toGuardIsNativeObject()->object();
      uint32_t shapeOff = uint32_t(offsetof(JS::shadow::Object, shape));
      uint32_t flagsOff = uint32_t(js::Shape::offsetOfImmutableFlags());
      if (!GetOp(e, be, obj)) return false;                 // obj (current)
      if (!e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(shapeOff))
        return false;                                       // obj->shape_
      if (!e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(flagsOff))
        return false;                                       // shape->immutableFlags
      if (!e.writeOp(Op::I32Const) ||
          !e.writeVarS32(int32_t(js::Shape::isNativeBit())))
        return false;
      if (!e.writeOp(Op::I32And)) return false;
      if (!e.writeOp(Op::I32Eqz)) return false;             // (flags & isNativeBit)==0 ?
      if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;  // non-native -> deopt
      if (!EmitDeopt(e, be)) return false;
      if (!e.writeOp(Op::End)) return false;
      return GetOp(e, be, obj);                             // passthrough
    }
    case MDefinition::Opcode::ObjectClassToString: {
      // Object.prototype.toString fast path -> "[object X]" String. The helper
      // returns a 0-ptr sentinel when the object may have an @@toStringTag (Ion
      // bailout-on-null) -> deopt to PBL.
      MObjectClassToString* o = ins->toObjectClassToString();
      if (!EmitStageScratch(e, be, o->object(), 0)) return false;
      if (!EmitHelperCallResult(e, be, ins, WJH_OBJCLASSTOSTRING, 0)) return false;
      if (!EmitHelperResultAsType(e, be, jit::MIRType::String)) return false;  // [str ptr]
      if (!e.writeOp(Op::LocalTee) || !e.writeVarU32(be.propShapeLocal))
        return false;                                               // [str], propShape=str
      if (!e.writeOp(Op::I32Eqz)) return false;                     // [null?]
      if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
      if (!EmitDeopt(e, be)) return false;
      if (!e.writeOp(Op::End)) return false;
      return GetLocal(e, be.propShapeLocal);                        // [str]
    }
    case MDefinition::Opcode::CreateThis: {
      MCreateThis* c = ins->toCreateThis();
      // If this CreateThis is the `this` of a constructing call, pre-stage that
      // call's object/value args into scratch[0..argc) while their locals are
      // still fresh -- WJH_CREATETHIS allocates and a GC would otherwise stale
      // them (raytrace getRay/rayTrace `new Vector(...)` with object args ->
      // "rendered incorrectly"). callee/newTarget go into the TRACED 62/63 slots
      // (NOT scratch 0/1, which now hold pre-staged args); WJH_CREATETHIS reads
      // 62/63 and the construct call reuses them.
      if (MCall* cc = WJConstructConsumer(ins)) {
        uint32_t ccargc = cc->numActualArgs();
        if (ccargc > kWJMaxArgs) return false;
        for (uint32_t i = 0; i < ccargc; i++) {
          MDefinition* a = cc->getArg(1 + i);
          if (a->type() == MIRType::Object || a->type() == MIRType::Value) {
            if (!EmitStageOperandLocal(e, be, a, i)) return false;
          } else if (!EmitStageScratch(e, be, a, i)) {
            return false;
          }
        }
      }
      if (!EmitStageOperandLocal(e, be, c->getOperand(0), kWJCalleeSlot))
        return false;  // callee -> 62
      if (!EmitStageOperandLocal(e, be, c->getOperand(1), kWJNewTargetSlot))
        return false;  // newTarget -> 63
      // FUSE CreateThis into the construct call: instead of a separate
      // WJH_CREATETHIS helper hop (CreateThisFromIon) per `new`, emit the
      // JS_IS_CONSTRUCTING sentinel and let the single WJH_CONSTRUCT call allocate
      // `this`. Halves the construct helper-boundary crossings (raytrace did 2.45M
      // CreateThis calls/run). Sentinel `this` is also the spec-faithful
      // uninitialized-this state, so a deopt-resume captures it correctly.
      // GECKO_WJ_NOFUSECTOR reverts to the separate helper.
      static int fuseCtor = WJGETENV("GECKO_WJ_NOFUSECTOR") ? 0 : 1;
      if (fuseCtor) {
        return e.writeOp(Op::I64Const) &&
               e.writeVarS64(int64_t(
                   JS::MagicValue(JS_IS_CONSTRUCTING).asRawBits()));
      }
      return EmitHelperCallResult(e, be, ins, WJH_CREATETHIS, 0);
    }
    case MDefinition::Opcode::NewLexicalEnvironmentObject: {
      if (WJGETENV("GECKO_WJ_NONEWLEXENV")) return false;
      // Block-scope env alloc. The template object is a compile-time constant
      // BlockLexicalEnvironmentObject; bake its LexicalScope* and call the helper
      // (BlockLexicalEnvironmentObject::createWithoutEnclosing). uBlock's parser has
      // many block-scoped (let/const) functions -- this was the dominant ubo bail.
      MDefinition* tmpl = ins->toNewLexicalEnvironmentObject()->templateObj();
      if (!tmpl->isConstant() || tmpl->toConstant()->type() != MIRType::Object)
        return false;
      JSObject* tobj = &tmpl->toConstant()->toObject();
      js::Scope* scope = &tobj->as<js::BlockLexicalEnvironmentObject>().scope();
      if (!e.writeOp(Op::I32Const) ||
          !e.writeVarS32(int32_t(uintptr_t(&js::wasm::gWJLexScope))) ||
          !e.writeOp(Op::I32Const) ||
          !e.writeVarS32(int32_t(uintptr_t(static_cast<void*>(scope)))) ||
          !e.writeOp(Op::I32Store) || !e.writeVarU32(2) || !e.writeVarU32(0))
        return false;
      if (!EmitHelperCallResult(e, be, ins, WJH_NEWLEXENV, 0)) return false;
      return EmitHelperResultAsType(e, be, ins->type());  // Object
    }
    case MDefinition::Opcode::NewVarEnvironmentObject: {
      // Var-scope env alloc. Template is a compile-time constant VarEnvironmentObject;
      // bake its VarScope* and call VarEnvironmentObject::createWithoutEnclosing.
      // Mirrors NewLexicalEnvironmentObject above.
      MDefinition* tmpl = ins->toNewVarEnvironmentObject()->templateObj();
      if (!tmpl->isConstant() || tmpl->toConstant()->type() != MIRType::Object)
        return false;
      JSObject* tobj = &tmpl->toConstant()->toObject();
      js::Scope& s = tobj->as<js::VarEnvironmentObject>().scope();
      if (!s.is<js::VarScope>()) return false;  // EvalScope var-env -> bail (helper wants VarScope)
      js::Scope* scope = &s;
      if (!e.writeOp(Op::I32Const) ||
          !e.writeVarS32(int32_t(uintptr_t(&js::wasm::gWJVarScope))) ||
          !e.writeOp(Op::I32Const) ||
          !e.writeVarS32(int32_t(uintptr_t(static_cast<void*>(scope)))) ||
          !e.writeOp(Op::I32Store) || !e.writeVarU32(2) || !e.writeVarU32(0))
        return false;
      if (!EmitHelperCallResult(e, be, ins, WJH_NEWVARENV, 0)) return false;
      return EmitHelperResultAsType(e, be, ins->type());  // Object
    }
    case MDefinition::Opcode::NewArrayDynamicLength: {
      // new Array(n): ArrayConstructorOneArg(template, n) -- the exact Ion VM fallback
      // (throws RangeError on bad length == the op's negative-length guard).
      // OPT-IN (GECKO_WJ_NADL). Flipped default-on 2026-06-30 then REVERTED same day:
      // cdjs is fine, but `realapp marked` (recursive markdown parser) crashes
      // INTERMITTENTLY with NADL on (2/10 runs) -- vs 1/10 with it off, so NADL appears
      // to worsen marked's pre-existing intermittent failure. Not "confirmed safe", and
      // not worth it for 17 cold bails. Use GECKO_WJ_NADL + GECKO_WJ_STACKGUARD for the
      // safe combo. (Separately: marked has a PRE-EXISTING ~10% flake even NADL-off --
      // investigate that before re-attempting the flip.)
      // 2026-07-02 FLIPPED default-on: the marked flake was the stale-string
      // corruption (string-rooting keystone) that NADL's extra alloc/GC amplified.
      // Post-fix marked is 57/57 clean NADL-on.
      MNewArrayDynamicLength* m = ins->toNewArrayDynamicLength();
      if (!EmitStageScratch(e, be, m->length(), 0)) return false;
      uint64_t tBits = JS::ObjectValue(*m->templateObject()).asRawBits();
      if (!EmitStageConstBoxed(e, be, tBits, 1)) return false;  // template (GC-traced)
      if (!EmitHelperCallResult(e, be, ins, WJH_NEWARRDYN, 0)) return false;
      return EmitHelperResultAsType(e, be, ins->type());  // Object
    }
    case MDefinition::Opcode::NewObject:
    case MDefinition::Opcode::NewPlainObject:
    case MDefinition::Opcode::NewArrayObject:
    case MDefinition::Opcode::NewArray: {
      // If this allocation is the inline `this` of a constructing call, the
      // call's callee/newTarget wasm locals will be STALE by the time the call
      // emits (this alloc can GC and wasm locals aren't GC-traced). Pre-stage
      // them into the traced 62/63 scratch slots NOW, while their locals are
      // still fresh (no GC has happened yet) -- the traced slots are then
      // relocated by any GC this alloc triggers, so the construct reads a
      // GC-current callee. This is the raytrace `new Vector` construct crash fix.
      if (MCall* cc = WJConstructConsumer(ins)) {
        uint32_t ccargc = cc->numActualArgs();
        if (ccargc > kWJMaxArgs) return false;
        // Pre-stage ALL construct operands while their locals are still fresh:
        // object/value args (and the callee/newTarget) read their cached local
        // directly (NOT rematerialized -- a remat would re-read a now-dead
        // source). Number args don't move, so stage them the normal way.
        for (uint32_t i = 0; i < ccargc; i++) {
          MDefinition* a = cc->getArg(1 + i);
          if (a->type() == MIRType::Object || a->type() == MIRType::Value) {
            if (!EmitStageOperandLocal(e, be, a, i)) return false;
          } else if (!EmitStageScratch(e, be, a, i)) {
            return false;
          }
        }
        if (!EmitStageOperandLocal(e, be, cc->getCallee(), kWJCalleeSlot))
          return false;
        if (!EmitStageOperandLocal(e, be, cc->getArg(cc->numStackArgs() - 1),
                                   kWJNewTargetSlot))
          return false;
      }
      // Object/array allocation -> wjhelp(WJH_NEW*) which calls the same VM
      // fallback Ion uses (NewPlainObjectOptimizedFallback / NewArrayObject-
      // OptimizedFallback / NewArrayOperation). Keeps the allocating function in
      // JIT (was bailing whole to PBL: splay GeneratePayloadTree, navier reset,
      // raytrace Vectors, earley conses). Stage the alloc params into the (non-GC)
      // globals, call, unwrap the boxed Object result to a pointer. NB the alloc
      // itself is a C++ hop; inline nursery bump-alloc (Ion's createPlainGCObject)
      // is the follow-up for alloc-bound benches (splay). The helper rematerializes
      // the shape from the traced pool (GC-current).
      // DEFAULT-OFF (GECKO_WJ_NEWALLOC to enable): measured a REGRESSION on the
      // alloc-bound benches it targets -- splay 1.03x->0.87x (helper C++ hop +
      // fresh-array StoreElement deopts make the JIT'd GeneratePayloadTree SLOWER
      // than clean PBL), and raytrace allocates a Vector per vector-op (same).
      // Bailing those functions whole to PBL (return false) is faster than helper-
      // alloc. The REAL fix is INLINE nursery bump-allocation (Ion's
      // createPlainGCObject: load nursery position, bump, bounds-check vs
      // currentEnd, store NurseryCellHeader+shape+slots+elements, fill undefined;
      // this WJH_NEW* helper becomes the OOL nursery-full fallback). Kept gated.
      // DEFAULT-ON now (was gated): bailing the whole function to PBL is worse
      // than the helper hop per the "compile everything" directive, and inline
      // bump-alloc below handles the hot NewPlainObject case. GECKO_WJ_NONEWALLOC
      // reverts to bailing.
      static int doNewAlloc = WJGETENV("GECKO_WJ_NONEWALLOC") ? 0 : 1;
      if (!doNewAlloc) return false;
      auto stG = [&](void* g, int32_t v) -> bool {
        return e.writeOp(Op::I32Const) && e.writeVarS32(int32_t(uintptr_t(g))) &&
               e.writeOp(Op::I32Const) && e.writeVarS32(v) &&
               e.writeOp(Op::I32Store) && e.writeVarU32(2) && e.writeVarU32(0);
      };
      int kind;
      // NewObject in ObjectLiteral mode carries NO compile-time MNewObject template,
      // but for JSOp::NewObject the SharedShape IS baked into the script
      // (script->getShape(pc) -- exactly what NewObjectOperation uses). Grab it at
      // compile time and feed the inline bump-alloc below: splay GeneratePayloadTree's
      // {array,left,right} literal = 2.1M WJH_NEWOBJECT helper hops -> inline nursery
      // bumps (each helper hop was also a GC safepoint forcing a full root spill).
      // NewInit/Object literal forms have no baked shape -> stay on WJH_NEWOBJECT.
      js::SharedShape* litShape = nullptr;
      if (ins->op() == MDefinition::Opcode::NewObject &&
          !ins->toNewObject()->templateObject()) {
        // Use the RESUME POINT pc, not trackedSite() -- the latter points at a
        // JumpTarget here (NewObjectOperation would read a garbage gcthing ->
        // wrong-shaped object -> OOB StoreFixedSlot -> heap corruption). The rp pc
        // is the JSOp::NewObject bytecode (matches Ion's visitNewObject site).
        MResumePoint* rp = ins->resumePoint();
        if (!rp || !rp->pc()) return false;
        JSScript* sc = rp->block()->info().script();
        if (!sc) return false;
        jsbytecode* pc = rp->pc();
        if (JSOp(*pc) == JSOp::NewObject) {
          litShape = sc->getShape(pc);  // SharedShape baked in the script data
        }
        if (!litShape) {
          if (JSOp(*pc) != JSOp::NewInit && JSOp(*pc) != JSOp::Object) {
            if (WJGETENV("GECKO_WJ_NEWOBJDBG"))
              fprintf(stderr, "[wj-newobj] %s:%u UNEXPECTED op@rp=%s -> bail\n",
                      sc->filename() ? sc->filename() : "?", unsigned(sc->lineno()),
                      js::CodeName(JSOp(*pc)));
            return false;  // unexpected pc -> bail rather than risk corruption
          }
          uint32_t pcOff = uint32_t(pc - sc->code());
          if (!stG(&js::wasm::gWJNewObjScript,
                   int32_t(uintptr_t(static_cast<void*>(sc)))) ||
              !stG(&js::wasm::gWJNewObjPcOff, int32_t(pcOff)))
            return false;
          if (!EmitHelperCallResult(e, be, ins, WJH_NEWOBJECT, 0)) return false;
          return e.writeOp(Op::I32WrapI64);
        }
      }
      // NewObject (object literal `{a:.., b:..}` -- splay GeneratePayloadTree) shares
      // the NewPlainObject inline bump-alloc, deriving the shape/layout from its
      // template object. Both allocate an object with a fixed shape, fixed slots
      // init'd to undefined (the bytecode's InitProp/StoreFixedSlot fills the real
      // values). Falls to the WJH_NEWPLAIN helper (any shared shape) on nursery-full.
      bool isPlainLike = ins->op() == MDefinition::Opcode::NewPlainObject ||
                         ins->op() == MDefinition::Opcode::NewObject;
      if (isPlainLike) {
        const js::Shape* pShape = nullptr;
        uint32_t pNfixed = 0, pNdynamic = 0;
        js::gc::AllocKind pAllocKind = js::gc::AllocKind::OBJECT0;
        gc::Heap pHeap = gc::Heap::Default;
        if (ins->op() == MDefinition::Opcode::NewPlainObject) {
          MNewPlainObject* n = ins->toNewPlainObject();
          pShape = n->shape();
          pNfixed = n->numFixedSlots();
          pNdynamic = n->numDynamicSlots();
          pAllocKind = n->allocKind();
          pHeap = n->initialHeap();
        } else if (litShape) {
          // Object literal (JSOp::NewObject), shape from the bytecode. Mirror
          // PlainObject::createWithShape EXACTLY: allocKind =
          // GetGCObjectKind(numFixedSlots), nursery heap. The shape's nfixed is
          // allocKind-aligned (literal shapes are created that way), so the inline
          // undef-fill covers all physical fixed slots. ndynamic>0 (slotSpan beyond
          // fixed) routes to the helper via the guard below.
          pShape = litShape;
          pNfixed = litShape->numFixedSlots();
          pAllocKind = js::gc::GetGCObjectKind(pNfixed);
          uint32_t span = litShape->slotSpan();
          pNdynamic = span > pNfixed ? span - pNfixed : 0;
          pHeap = ins->toNewObject()->initialHeap();
        } else {
          JSObject* t = ins->toNewObject()->templateObject();
          if (!t || !t->is<js::NativeObject>()) return false;  // VM-call mode: bail
          js::NativeObject* nt = &t->as<js::NativeObject>();
          pShape = nt->shape();
          pNfixed = nt->numFixedSlots();
          pNdynamic = nt->numDynamicSlots();
          pAllocKind = nt->asTenured().getAllocKind();
          pHeap = ins->toNewObject()->initialHeap();
        }
        uintptr_t shapeAddr = WJInternShape(uintptr_t(pShape));
        if (!shapeAddr) return false;
        if (WJGETENV("GECKO_WJ_HEAPDBG"))
          fprintf(stderr, "[wj-heap] %s:%u op=%s heap=%s nfixed=%u ndyn=%u\n",
                  ins->block()->info().script()->filename(),
                  unsigned(ins->block()->info().script()->lineno()),
                  WJOpName(ins->op()),
                  pHeap == gc::Heap::Tenured ? "TENURED" : "default",
                  pNfixed, pNdynamic);
        // INLINE nursery bump-allocation (Ion createPlainGCObject), with the
        // WJH_NEWPLAIN helper as the nursery-full OOL fallback. Makes alloc-bound
        // benches (raytrace Vectors, splay payload objects) fast instead of a C++
        // hop. Gated GECKO_WJ_INLINEALLOC. Only ndynamic==0 (else helper). All
        // layout values (header size, currentEnd offset, thingSize) come from the
        // real runtime functions so wasm32 alignment is correct by construction.
        // pHeap==Tenured (Warp pretenuring) MUST skip the nursery bump -> use the
        // helper, which allocates in the requested heap (else we'd ignore the hint
        // and nursery-allocate long-lived objects -> needless promotion+barriers).
        static int inlineAlloc = WJGETENV("GECKO_WJ_NOINLINEALLOC") ? 0 : 1;
        if (inlineAlloc && pNdynamic == 0 && gWJNurseryPosAddr &&
            gWJObjHeaderWord && pHeap != gc::Heap::Tenured) {
          uint32_t headerSize = uint32_t(js::Nursery::nurseryCellHeaderSize());
          int32_t endOff = js::Nursery::offsetOfCurrentEndFromPosition();
          uint32_t totalSize =
              uint32_t(js::gc::Arena::thingSize(pAllocKind)) + headerSize;
          uint32_t nfixed = pNfixed;
          uint32_t shapeOff = uint32_t(offsetof(JS::shadow::Object, shape));
          uint32_t slotsOff = uint32_t(js::NativeObject::offsetOfSlots());
          uint32_t elemsOff = uint32_t(js::NativeObject::offsetOfElements());
          int32_t emptySlots = int32_t(uintptr_t(js::emptyObjectSlots));
          int32_t emptyElems = int32_t(uintptr_t(js::emptyObjectElements));
          int32_t posAddr = int32_t(gWJNurseryPosAddr);
          int64_t undefBits = int64_t(JS::UndefinedValue().asRawBits());
          auto I32C = [&](int32_t v) {
            return e.writeOp(Op::I32Const) && e.writeVarS32(v);
          };
          auto store32 = [&](uint32_t off) {
            return e.writeOp(Op::I32Store) && e.writeVarU32(2) &&
                   e.writeVarU32(off);
          };
          // allocPos = *posAddr
          if (!I32C(posAddr) || !e.writeOp(Op::I32Load) || !e.writeVarU32(2) ||
              !e.writeVarU32(0) || !e.writeOp(Op::LocalSet) ||
              !e.writeVarU32(be.allocPosLocal))
            return false;
          // (currentEnd = load posAddr+endOff) < (newpos = allocPos + totalSize) ?
          if (!I32C(posAddr) || !e.writeOp(Op::I32Load) || !e.writeVarU32(2) ||
              !e.writeVarU32(uint32_t(endOff)))
            return false;
          if (!GetLocal(e, be.allocPosLocal) || !I32C(int32_t(totalSize)) ||
              !e.writeOp(Op::I32Add) || !e.writeOp(Op::I32LtU))
            return false;
          if (!e.writeOp(Op::If) || !e.writeFixedU8(uint8_t(TypeCode::I32)))
            return false;
          // OOL (nursery full): helper alloc, unwrap boxed Object -> ptr.
          if (!stG(&gWJNewShapeSlot, int32_t(shapeAddr)) ||
              !stG(&gWJNewAux, int32_t(uint32_t(pAllocKind))) ||
              !stG(&gWJNewHeap, int32_t(uint32_t(pHeap))) ||
              !EmitHelperCallResult(e, be, ins, WJH_NEWPLAIN, 0) ||
              !e.writeOp(Op::I32WrapI64))
            return false;
          if (!e.writeOp(Op::Else)) return false;
          // INLINE: commit position, write header, shape, slots, elements, slots.
          if (!I32C(posAddr) || !GetLocal(e, be.allocPosLocal) ||
              !I32C(int32_t(totalSize)) || !e.writeOp(Op::I32Add) || !store32(0))
            return false;  // *posAddr = newpos
          if (!GetLocal(e, be.allocPosLocal) || !I32C(int32_t(gWJObjHeaderWord)) ||
              !store32(0))
            return false;  // header at allocPos
          if (!GetLocal(e, be.allocPosLocal) || !I32C(int32_t(headerSize)) ||
              !e.writeOp(Op::I32Add) || !e.writeOp(Op::LocalSet) ||
              !e.writeVarU32(be.allocObjLocal))
            return false;  // obj = allocPos + headerSize
          if (!GetLocal(e, be.allocObjLocal) || !I32C(int32_t(shapeAddr)) ||
              !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0) ||
              !store32(shapeOff))
            return false;  // obj->shape = *(shape pool slot) (relocated)
          if (!GetLocal(e, be.allocObjLocal) || !I32C(emptySlots) ||
              !store32(slotsOff))
            return false;
          if (!GetLocal(e, be.allocObjLocal) || !I32C(emptyElems) ||
              !store32(elemsOff))
            return false;
          for (uint32_t f = 0; f < nfixed; f++) {
            uint32_t off = uint32_t(js::NativeObject::getFixedSlotOffset(f));
            if (!GetLocal(e, be.allocObjLocal) || !e.writeOp(Op::I64Const) ||
                !e.writeVarS64(undefBits) || !e.writeOp(Op::I64Store) ||
                !e.writeVarU32(3) || !e.writeVarU32(off))
              return false;
          }
          if (!GetLocal(e, be.allocObjLocal) || !e.writeOp(Op::End)) return false;
          return true;  // obj pointer (i32) on stack
        }
        if (!stG(&gWJNewShapeSlot, int32_t(shapeAddr)) ||
            !stG(&gWJNewAux, int32_t(uint32_t(pAllocKind))) ||
            !stG(&gWJNewHeap, int32_t(uint32_t(pHeap))))
          return false;
        kind = WJH_NEWPLAIN;
      } else if (ins->op() == MDefinition::Opcode::NewArrayObject) {
        MNewArrayObject* n = ins->toNewArrayObject();
        if (!stG(&gWJNewAux, int32_t(n->length())) ||
            !stG(&gWJNewHeap, int32_t(uint32_t(n->initialHeap()))))
          return false;
        kind = WJH_NEWARROBJ;
      } else {
        MNewArray* n = ins->toNewArray();
        // INLINE dense-array bump-alloc (mirrors masm.createArrayWithFixedElements):
        // needs a compile-time template (shape/capacity/allocKind) with FIXED
        // (inline) elements + no dynamic slots. initializedLength=0 (the subsequent
        // InitElemArray ops fill the values and bump it), so NO element-value init is
        // needed here -- simpler than the plain-object path. splay's leaf `[0..9]` =
        // 1.07M WJH_NEWARR helper hops (each a GC safepoint) -> inline nursery bumps.
        // Falls to WJH_NEWARR on no-template / dynamic elements / nursery-full /
        // pretenured. GECKO_WJ_NOINLINEARR reverts.
        static int inlineArr = WJGETENV("GECKO_WJ_NOINLINEARR") ? 0 : 1;
        // Derive (shape, allocKind, capacity) for an inline dense array. The
        // template is ALWAYS null in this PBL build (no warmed IC), so -- exactly
        // like the NewObject-literal `litShape` path -- synthesize from the realm's
        // canonical dense-array shape (GlobalObject::getArrayShapeWithDefaultProto,
        // nfixed=0, the same shape NewArrayOperation->NewArray uses) and compute
        // allocKind=GuessArrayGCKind(length), inline capacity=GetGCKindSlots-2.
        // Only when the elements fit INLINE (capacity>=length); larger arrays
        // (dynamic elements) fall to the WJH_NEWARR helper.
        JSObject* t = n->templateObject();
        js::ArrayObject* ta =
            (t && t->is<js::ArrayObject>()) ? &t->as<js::ArrayObject>() : nullptr;
        const js::Shape* arrShape = nullptr;
        js::gc::AllocKind ak = js::gc::AllocKind::OBJECT0;
        uint32_t arrLen = n->length();
        uint32_t arrCap = 0;
        if (ta && ta->numDynamicSlots() == 0 && ta->hasFixedElements()) {
          arrShape = ta->shape();
          ak = ta->asTenured().getAllocKind();
          arrCap = ta->getDenseCapacity();
        } else if (!ta) {
          JSContext* cx = js::TlsContext.get();
          if (cx) {
            js::SharedShape* s =
                js::GlobalObject::getArrayShapeWithDefaultProto(cx);
            if (s) {
              ak = js::GuessArrayGCKind(arrLen);
              uint32_t cap = uint32_t(js::gc::GetGCKindSlots(ak)) -
                             uint32_t(js::ObjectElements::VALUES_PER_HEADER);
              if (cap >= arrLen) {  // fits inline (else dynamic elems -> helper)
                arrShape = s;
                arrCap = cap;
              }
            }
          }
        }
        if (inlineArr && arrShape && n->initialHeap() != gc::Heap::Tenured &&
            gWJNurseryPosAddr && gWJObjHeaderWord) {
          uintptr_t shapeAddr = WJInternShape(uintptr_t(arrShape));
          if (!shapeAddr) return false;
          uint32_t headerSize = uint32_t(js::Nursery::nurseryCellHeaderSize());
          uint32_t totalSize =
              uint32_t(js::gc::Arena::thingSize(ak)) + headerSize;
          int32_t endOff = js::Nursery::offsetOfCurrentEndFromPosition();
          uint32_t shapeOff = uint32_t(offsetof(JS::shadow::Object, shape));
          uint32_t slotsOff = uint32_t(js::NativeObject::offsetOfSlots());
          uint32_t elemsOff = uint32_t(js::NativeObject::offsetOfElements());
          uint32_t fixedElemsOff =
              uint32_t(js::NativeObject::offsetOfFixedElements());
          uint32_t flagsOff =
              fixedElemsOff + uint32_t(js::ObjectElements::offsetOfFlags());
          uint32_t initLenOff =
              fixedElemsOff +
              uint32_t(js::ObjectElements::offsetOfInitializedLength());
          uint32_t capOff =
              fixedElemsOff + uint32_t(js::ObjectElements::offsetOfCapacity());
          uint32_t lenOff =
              fixedElemsOff + uint32_t(js::ObjectElements::offsetOfLength());
          int32_t emptySlots = int32_t(uintptr_t(js::emptyObjectSlots));
          int32_t posAddr = int32_t(gWJNurseryPosAddr);
          auto I32C = [&](int32_t v) {
            return e.writeOp(Op::I32Const) && e.writeVarS32(v);
          };
          auto store32 = [&](uint32_t off) {
            return e.writeOp(Op::I32Store) && e.writeVarU32(2) &&
                   e.writeVarU32(off);
          };
          // allocPos = *posAddr
          if (!I32C(posAddr) || !e.writeOp(Op::I32Load) || !e.writeVarU32(2) ||
              !e.writeVarU32(0) || !e.writeOp(Op::LocalSet) ||
              !e.writeVarU32(be.allocPosLocal))
            return false;
          // (currentEnd) < (allocPos + totalSize) ?
          if (!I32C(posAddr) || !e.writeOp(Op::I32Load) || !e.writeVarU32(2) ||
              !e.writeVarU32(uint32_t(endOff)))
            return false;
          if (!GetLocal(e, be.allocPosLocal) || !I32C(int32_t(totalSize)) ||
              !e.writeOp(Op::I32Add) || !e.writeOp(Op::I32LtU))
            return false;
          if (!e.writeOp(Op::If) || !e.writeFixedU8(uint8_t(TypeCode::I32)))
            return false;
          // OOL (nursery full): WJH_NEWARR helper, unwrap boxed Object -> ptr.
          if (!stG(&gWJNewAux, int32_t(arrLen)) ||
              !EmitHelperCallResult(e, be, ins, WJH_NEWARR, 0) ||
              !e.writeOp(Op::I32WrapI64))
            return false;
          if (!e.writeOp(Op::Else)) return false;
          // INLINE: commit pos, header, obj = pos + headerSize.
          if (!I32C(posAddr) || !GetLocal(e, be.allocPosLocal) ||
              !I32C(int32_t(totalSize)) || !e.writeOp(Op::I32Add) || !store32(0))
            return false;
          if (!GetLocal(e, be.allocPosLocal) || !I32C(int32_t(gWJObjHeaderWord)) ||
              !store32(0))
            return false;
          if (!GetLocal(e, be.allocPosLocal) || !I32C(int32_t(headerSize)) ||
              !e.writeOp(Op::I32Add) || !e.writeOp(Op::LocalSet) ||
              !e.writeVarU32(be.allocObjLocal))
            return false;
          // obj->shape = *(shape pool slot) (GC-relocated)
          if (!GetLocal(e, be.allocObjLocal) || !I32C(int32_t(shapeAddr)) ||
              !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0) ||
              !store32(shapeOff))
            return false;
          if (!GetLocal(e, be.allocObjLocal) || !I32C(emptySlots) ||
              !store32(slotsOff))
            return false;
          // obj->elements_ = obj + offsetOfFixedElements (inline elements)
          if (!GetLocal(e, be.allocObjLocal) || !GetLocal(e, be.allocObjLocal) ||
              !I32C(int32_t(fixedElemsOff)) || !e.writeOp(Op::I32Add) ||
              !store32(elemsOff))
            return false;
          // ObjectElements header: flags=FIXED, initializedLength=0, capacity, length
          if (!GetLocal(e, be.allocObjLocal) ||
              !I32C(int32_t(uint32_t(js::ObjectElements::FIXED))) ||
              !store32(flagsOff))
            return false;
          if (!GetLocal(e, be.allocObjLocal) || !I32C(0) || !store32(initLenOff))
            return false;
          if (!GetLocal(e, be.allocObjLocal) || !I32C(int32_t(arrCap)) ||
              !store32(capOff))
            return false;
          if (!GetLocal(e, be.allocObjLocal) || !I32C(int32_t(arrLen)) ||
              !store32(lenOff))
            return false;
          if (!GetLocal(e, be.allocObjLocal) || !e.writeOp(Op::End)) return false;
          return true;  // array obj pointer (i32) on stack
        }
        if (!stG(&gWJNewAux, int32_t(n->length()))) return false;
        kind = WJH_NEWARR;
      }
      if (!EmitHelperCallResult(e, be, ins, kind, 0)) return false;
      return e.writeOp(Op::I32WrapI64);  // boxed Object -> pointer
    }
    case MDefinition::Opcode::BitAnd:
    case MDefinition::Opcode::BitOr:
    case MDefinition::Opcode::BitXor:
    case MDefinition::Opcode::Lsh:
    case MDefinition::Opcode::Rsh:
    case MDefinition::Opcode::Ursh: {
      // NOTE (2026-07-04): tried rotate-idiom recognition here — `(x<<c1)|(x>>>c2)`,
      // c1+c2==32 -> i32.rotl — CORRECT but ZERO measurable benefit (crypto-sha1
      // 1159ms rotl vs 1133ms without). V8 already pattern-matches the shl|shr_u|or
      // idiom into a native rotate at wasm->native, so explicit i32.rotl is identical.
      // sha1's near-parity ceiling is V8-tier codegen, NOT our emission. Reverted.
      // JS bitwise/shift ops operate on i32 (shift counts masked mod 32, which wasm
      // shifts already do). Operands must be Int32. Result is signed Int32 for all
      // EXCEPT Ursh (>>>): its UNSIGNED result can exceed INT32_MAX, so Ion types it
      // Double -- we then widen the uint32 with f64.convert_i32_u (exact, Ion parity).
      // (Float32/type7 never occurs for these ops.) Revert the Double-Ursh path with
      // GECKO_WJ_NOURSHDBL.
      bool dblUrsh = ins->op() == MDefinition::Opcode::Ursh &&
                     ins->type() == MIRType::Double &&
                     !WJGETENV("GECKO_WJ_NOURSHDBL");
      if (ins->type() != MIRType::Int32 && !dblUrsh) return false;
      if (ins->getOperand(0)->type() != MIRType::Int32 ||
          ins->getOperand(1)->type() != MIRType::Int32) {
        return false;
      }
      if (!GetOp(e, be, ins->getOperand(0)) || !GetOp(e, be, ins->getOperand(1))) {
        return false;
      }
      switch (ins->op()) {
        case MDefinition::Opcode::BitAnd: return e.writeOp(Op::I32And);
        case MDefinition::Opcode::BitOr: return e.writeOp(Op::I32Or);
        case MDefinition::Opcode::BitXor: return e.writeOp(Op::I32Xor);
        case MDefinition::Opcode::Lsh: return e.writeOp(Op::I32Shl);
        case MDefinition::Opcode::Rsh: return e.writeOp(Op::I32ShrS);
        default: {  // Ursh
          if (!e.writeOp(Op::I32ShrU)) return false;
          // Double-typed result -> widen the uint32 to f64 (unsigned conversion).
          if (dblUrsh) return e.writeOp(Op::F64ConvertI32U);
          // A constant shift of n where (n % 32) != 0 ALWAYS clears bit 31, so the uint32
          // result is < 2^31 = a valid non-negative Int32 -> no deopt needed (covers the
          // hot crypto `>>>8/16/24` shifts with zero added cost). Only `>>>0` (or a variable
          // shift that could be 0) can preserve bit 31.
          jit::MDefinition* amt = ins->getOperand(1);
          if (amt->isConstant()) {
            int32_t c = amt->toConstant()->toInt32();
            if ((uint32_t(c) % 32u) != 0) return true;  // bit 31 always clear -> safe
          }
          // Int32-typed Ursh: Warp speculates the uint32 result fits a NON-NEGATIVE
          // int32 (bit 31 clear). But `>>>` on a negative input yields a uint32 > INT32_MAX
          // whose bits, read as signed int32, are NEGATIVE -- and JS requires that value to
          // be a Double, not a negative Int32. Ion BAILS in that case; mirror it: deopt to
          // PBL (which produces the correct double) when the result has bit 31 set. Without
          // this, `(-1)>>>0` returned -1 instead of 4294967295 (silent hash/crypto/uint32
          // corruption -- warmed on non-negative inputs so Warp typed it Int32). Mirrors the
          // Add/Sub/Mul overflow-deopt above; reuses the i64 unboxScratch (survives EmitDeopt).
          if (!e.writeOp(Op::I64ExtendI32S) || !e.writeOp(Op::LocalSet) ||
              !e.writeVarU32(be.unboxScratch))
            return false;
          if (!GetLocal(e, be.unboxScratch) || !e.writeOp(Op::I64Const) ||
              !e.writeVarS64(0) || !e.writeOp(Op::I64LtS))
            return false;
          if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
          if (!EmitDeopt(e, be)) return false;
          if (!e.writeOp(Op::End)) return false;
          return GetLocal(e, be.unboxScratch) && e.writeOp(Op::I32WrapI64);
        }
      }
    }
    case MDefinition::Opcode::BitNot: {
      // ~x = ToInt32(x) ^ -1. Int32 operand inline; a boxed Value operand bails (rare).
      if (ins->type() != MIRType::Int32) return false;
      MDefinition* in = ins->getOperand(0);
      if (in->type() != MIRType::Int32) return false;
      return GetOp(e, be, in) && e.writeOp(Op::I32Const) && e.writeVarS32(-1) &&
             e.writeOp(Op::I32Xor);
    }
    case MDefinition::Opcode::CheckIsObj: {
      // Deopt unless the value's tag is OBJECT; PBL re-runs CheckIsObj, which throws a
      // TypeError (caught by an enclosing try via deopt-in-error). Result = unboxed obj.
      int32_t l = be.local(ins->getOperand(0));
      if (l < 0) return false;
      if (!GetLocal(e, uint32_t(l)) || !e.writeOp(Op::I64Const) || !e.writeVarS64(32) ||
          !e.writeOp(Op::I64ShrU) || !e.writeOp(Op::I32WrapI64) ||
          !e.writeOp(Op::I32Const) ||
          !e.writeVarS32(int32_t(TagWord(JSVAL_TYPE_OBJECT))) || !e.writeOp(Op::I32Ne))
        return false;
      if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
      if (!EmitDeopt(e, be)) return false;
      if (!e.writeOp(Op::End)) return false;
      return GetLocal(e, uint32_t(l)) && e.writeOp(Op::I32WrapI64);  // unbox object
    }
    case MDefinition::Opcode::Call: {
      MCall* call = ins->toCall();
      // Inline the reserved-slot self-hosted intrinsics that Ion always inlines
      // (UnsafeGet/SetReservedSlot) -- in ubo these are ~236K slow WJH_CALL/JS::Call
      // boundary crossings (Map/Set/iterator internals). Ion lowers them to a raw
      // fixed-slot load/store (CacheIR tryAttachUnsafeGetReservedSlot). Args (MCall
      // operand layout): getArg(0)=this, getArg(1)=object, getArg(2)=slot(const i32),
      // getArg(3)=value. Reserved slots are FIXED (slot < MAX_FIXED_SLOTS).
      static int noIntrinSlot = WJGETENV("GECKO_WJ_NOINTRINSICSLOT") ? 1 : 0;
      if (!noIntrinSlot && !call->isConstructing()) {
        // getSingleTarget() is null in our PBL-feedback build, but self-hosted
        // intrinsic calls have a CONSTANT callee holding the intrinsic JSFunction.
        // Read the InlinableNative off it.
        JSFunction* ifn = nullptr;
        MDefinition* ceD = call->getCallee();
        if (ceD->isConstant() && ceD->toConstant()->type() == MIRType::Object) {
          JSObject* o = ceD->toConstant()->toObjectOrNull();
          if (o && o->is<JSFunction>()) ifn = &o->as<JSFunction>();
        }
        if (ifn && ifn->isNativeFun() && ifn->hasJitInfo() &&
            ifn->jitInfo()->type() == JSJitInfo::InlinableNative) {
          {
            jit::InlinableNative nat = ifn->jitInfo()->inlinableNative;
            bool isGet =
                nat == jit::InlinableNative::IntrinsicUnsafeGetReservedSlot ||
                nat == jit::InlinableNative::IntrinsicUnsafeGetObjectFromReservedSlot ||
                nat == jit::InlinableNative::IntrinsicUnsafeGetInt32FromReservedSlot ||
                nat == jit::InlinableNative::IntrinsicUnsafeGetStringFromReservedSlot;
            bool isSet = nat == jit::InlinableNative::IntrinsicUnsafeSetReservedSlot;
            uint32_t na = call->numActualArgs();
            MDefinition* slotD =
                (isGet && na >= 2) ? call->getArg(2)
                : (isSet && na >= 3) ? call->getArg(2) : nullptr;
            if ((isGet || isSet) && slotD && slotD->isConstant() &&
                slotD->toConstant()->type() == MIRType::Int32) {
              int32_t slot = slotD->toConstant()->toInt32();
              if (slot >= 0 && uint32_t(slot) < js::NativeObject::MAX_FIXED_SLOTS) {
                uint32_t off = uint32_t(js::NativeObject::getFixedSlotOffset(slot));
                MDefinition* objD = call->getArg(1);
                // GetOp returns a raw i32 ptr for Object/String/etc.-typed args
                // (already unboxed) but a boxed i64 for Value/Int64 args -- only the
                // latter needs I32WrapI64. ArrayIteratorNext passes an Object-typed
                // `this`, so an unconditional wrap double-wrapped -> invalid wasm.
                bool objBoxed = objD->type() == MIRType::Value ||
                                objD->type() == MIRType::Int64;
                if (isGet) {
                  // unbox to ptr, load the slot (boxed Value result; MCall result
                  // type is Value, leave boxed).
                  if (!GetOp(e, be, objD) ||
                      (objBoxed && !e.writeOp(Op::I32WrapI64)) ||
                      !e.writeOp(Op::I64Load) || !e.writeVarU32(3) ||
                      !e.writeVarU32(off))
                    return false;
                  return true;
                }
                MDefinition* valD = call->getArg(3);
                // INLINE the reserved-slot store (slot write + pre/post barriers),
                // staying in wasm -- WJH_SETSLOT was ~1M/iter in ubo (Map/Set/iterator
                // internals). Mirrors the StoreFixedSlot(barrier) inline. Restrict to
                // an Object-typed `obj` (the dominant case; the barrier helpers take
                // an Object MDefinition); a Value-typed obj falls to the helper below.
                static int noInlineSet =
                    WJGETENV("GECKO_WJ_NOINLINERESERVEDSET") ? 1 : 0;
                if (!noInlineSet && objD->type() == MIRType::Object) {
                  if (!EmitGuardedValuePreBarrier(e, be, [&]() {
                        return GetOp(e, be, objD) && e.writeOp(Op::I64Load) &&
                               e.writeVarU32(3) && e.writeVarU32(off);
                      }))
                    return false;
                  if (!GetOp(e, be, objD)) return false;        // [objptr]
                  if (!EmitSpillValue(e, be, valD)) return false;  // [objptr, boxedVal]
                  if (!e.writeOp(Op::I64Store) || !e.writeVarU32(3) ||
                      !e.writeVarU32(off))
                    return false;
                  if (WJValueMightBeGCThing(valD->type()) &&
                      !EmitForcePostBarrier(e, be, objD, valD))
                    return false;
                  return e.writeOp(Op::I64Const) &&
                         e.writeVarS64(int64_t(JS::UndefinedValue().asRawBits()));
                }
                // Fallback (Value-typed obj / opt-out): gWJHelpObj/Slot/Val +
                // wjhelp(WJH_SETSLOT) does the barriered setSlot in C++.
                uintptr_t objAddr = uintptr_t(static_cast<void*>(&gWJHelpObj));
                uintptr_t slotAddr = uintptr_t(static_cast<void*>(&gWJHelpSlot));
                uintptr_t valAddr = uintptr_t(static_cast<void*>(&gWJHelpVal));
                if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(objAddr)) ||
                    !GetOp(e, be, objD) ||
                    (objBoxed && !e.writeOp(Op::I32WrapI64)) ||
                    !e.writeOp(Op::I32Store) || !e.writeVarU32(2) || !e.writeVarU32(0))
                  return false;
                if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(slotAddr)) ||
                    !e.writeOp(Op::I32Const) || !e.writeVarS32(slot) ||
                    !e.writeOp(Op::I32Store) || !e.writeVarU32(2) || !e.writeVarU32(0))
                  return false;
                if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(valAddr)) ||
                    !EmitSpillValue(e, be, valD) ||
                    !e.writeOp(Op::I64Store) || !e.writeVarU32(3) || !e.writeVarU32(0))
                  return false;
                if (!e.writeOp(Op::F64Const) || !e.writeFixedF64(double(WJH_SETSLOT)) ||
                    !e.writeOp(Op::F64Const) || !e.writeFixedF64(0.0) ||
                    !e.writeOp(Op::Call) || !e.writeVarU32(0) || !e.writeOp(Op::Drop))
                  return false;
                return e.writeOp(Op::I64Const) &&
                       e.writeVarS64(int64_t(JS::UndefinedValue().asRawBits()));
              }
            }
          }
        }
      }
      // Constructing calls: implemented (WJH_CONSTRUCT via
      // InternalConstructWithProvidedThis) and CORRECT for interp->JIT callers
      // (the construct produces valid objects; repro /tmp/ctortest2.js passes).
      // DISABLED because of an unresolved JIT->JIT issue: when a JIT'd caller
      // (e.g. blend) fast-call_indirects into a JIT'd construct-containing callee
      // (repro /tmp/ct5.js, /tmp/ctortest3.js), the FIRST construct succeeds
      // (verified: fval=function, valid result) but a subsequent throw surfaces
      // as `unreachable` -- it's a ReportValueError whose stack decompilation
      // (FrameIter -> baselineScriptAndPc -> retAddrEntryFromReturnAddress) traps
      // because this is JS_CODEGEN_NONE (no baseline frames) and FrameIter can't
      // walk our wasm-JIT frames during error reporting. NOT a GC-rooting issue
      // (NOROOT doesn't change it; the dominance-rooting fix in WJCollectRoots is
      // kept and benefits all other helpers). Re-enable after the error/frame
      // re-entrancy across nested wasm-JIT construct calls is resolved.
      if (call->isConstructing()) {
        // EHABI: the construct path's CallIndirect + flag-based exception returns aren't
        // converted yet -> bail constructing fns to PBL under EHABI (staged; functions
        // with `new` stay PBL until the construct path is EH-converted in a later step).
        if constexpr (kWJEHABI) return false;
        static int noConstruct = WJGETENV("GECKO_WJ_NOCONSTRUCT") ? 1 : 0;
        if (noConstruct) return false;
        uint32_t cargc = call->numActualArgs();
        if (cargc > kWJMaxArgs) return false;
        if (WJGETENV("GECKO_WJ_CTORDBG")) {
          MDefinition* cal = call->getCallee();
          fprintf(stderr, "[wj-ctor] calleeOp=%s const=%d type=%s cargc=%u\n",
                  WJOpName(cal->op()), cal->isConstant(),
                  StringFromMIRType(cal->type()), cargc);
        }
        // If `this` (arg0) is an allocating op, the alloc site already pre-staged
        // the args + callee + newTarget into the traced scratch slots (while
        // their locals were fresh; the alloc can GC and wasm locals aren't
        // traced, so re-staging here would spill stale/moved pointers -- the
        // raytrace construct crash / "rendered incorrectly"). Otherwise no GC
        // occurs between operand load and here, so stage everything now.
        MDefinition* thisDef = call->getArg(0);
        bool thisAllocates =
            thisDef->op() == MDefinition::Opcode::NewObject ||
            thisDef->op() == MDefinition::Opcode::NewPlainObject ||
            thisDef->op() == MDefinition::Opcode::NewArrayObject ||
            thisDef->op() == MDefinition::Opcode::NewArray;
        bool thisIsCreateThis = thisDef->op() == MDefinition::Opcode::CreateThis;
        // CTORINLINE fast path: eliminate the WJH_CONSTRUCT boundary for monomorphic
        // `new X(args)` sites (earley's 8.87M cons-cell crossings). Keyed on a per-
        // site cache (gWJCtor*[site]) filled by WJH_CONSTRUCT. On a hit: inline
        // nursery-bump alloc `this` (cached shape/size, slot-init loop) + call_indirect
        // the ctor + result-select. On miss/nursery-full: fall to the WJH_CONSTRUCT
        // slow path below (wrapped in the same Block so both produce the boxed i64).
        static int ctorInlineBE = WJGETENV("GECKO_WJ_NOCTORINLINE") ? 0 : 1;
        uint32_t inlSite = (ctorInlineBE && thisIsCreateThis && gWJObjHeaderWord &&
                            gWJNurseryPosAddr)
                               ? WJAllocCtorSite()
                               : 0;
        if (inlSite) {
          auto I32C = [&](int32_t v) {
            return e.writeOp(Op::I32Const) && e.writeVarS32(v);
          };
          auto loadG = [&](void* g) {  // i32 load of a global word
            return I32C(int32_t(uintptr_t(g))) && e.writeOp(Op::I32Load) &&
                   e.writeVarU32(2) && e.writeVarU32(0);
          };
          uint32_t headerSize = uint32_t(js::Nursery::nurseryCellHeaderSize());
          int32_t endOff = js::Nursery::offsetOfCurrentEndFromPosition();
          int32_t posAddr = int32_t(gWJNurseryPosAddr);
          uint32_t shapeOff = uint32_t(offsetof(JS::shadow::Object, shape));
          uint32_t slotsOff = uint32_t(js::NativeObject::offsetOfSlots());
          uint32_t elemsOff = uint32_t(js::NativeObject::offsetOfElements());
          int32_t emptySlots = int32_t(uintptr_t(js::emptyObjectSlots));
          int32_t emptyElems = int32_t(uintptr_t(js::emptyObjectElements));
          uint32_t fixedBase = uint32_t(js::NativeObject::getFixedSlotOffset(0));
          int64_t undefBits = int64_t(JS::UndefinedValue().asRawBits());
          int32_t envAddr = int32_t(uintptr_t(static_cast<void*>(&gWJCurrentEnv)));
          uint32_t envOff = uint32_t(JSFunction::offsetOfEnvironment());
          MDefinition* nt = call->getArg(call->numStackArgs() - 1);
          auto store32 = [&](uint32_t off) {
            return e.writeOp(Op::I32Store) && e.writeVarU32(2) && e.writeVarU32(off);
          };
          // Block (result i64): the construct's boxed value.
          if (!e.writeOp(Op::Block) || !e.writeFixedU8(uint8_t(TypeCode::I64)))
            return false;
          // Cheap empty-cache short-circuit: if this site never filled (e.g. raytrace's
          // Class.create forwarding wrappers, deliberately not cached), gWJCtorCallee
          // is 0 -> skip boxing the callee/newTarget + the whole gate (was ~29% of
          // raytrace -- boxing operands on every construct that can never inline).
          if (!loadG(&gWJCtorCallee[inlSite]) || !e.writeOp(Op::If) ||
              !e.writeFixedU8(0x40))
            return false;  // cache-If
          // callee ptr -> callScratch
          if (!EmitObjPtr(e, be, call->getCallee()) || !e.writeOp(Op::LocalSet) ||
              !e.writeVarU32(be.callScratch))
            return false;
          // gate: callee==cached && newTarget==callee && tblIdx>=0
          if (!loadG(&gWJCtorCallee[inlSite]) || !GetLocal(e, be.callScratch) ||
              !e.writeOp(Op::I32Eq))
            return false;
          if (!EmitObjPtr(e, be, nt) || !GetLocal(e, be.callScratch) ||
              !e.writeOp(Op::I32Eq) || !e.writeOp(Op::I32And))
            return false;
          if (!loadG(&gWJCtorTblIdx[inlSite]) || !I32C(0) ||
              !e.writeOp(Op::I32GeS) || !e.writeOp(Op::I32And))
            return false;
          if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;  // gate-If
          // allocPos = *posAddr
          if (!I32C(posAddr) || !e.writeOp(Op::I32Load) || !e.writeVarU32(2) ||
              !e.writeVarU32(0) || !e.writeOp(Op::LocalSet) ||
              !e.writeVarU32(be.allocPosLocal))
            return false;
          // GECKO_WJ_CTORHELPERALLOC: allocate `this` via the GC-correct
          // PlainObject::createWithShape helper instead of the manual nursery bump
          // (whose half-built `this` gets swept during the ctor's mid-call GC --
          // validator site=5). Isolates/fixes that. Room-check is always-true here.
          static int helperAlloc = WJGETENV("GECKO_WJ_CTORHELPERALLOC") ? 1 : 0;
          // room: currentEnd >= allocPos + size
          if (helperAlloc) {
            if (!I32C(1)) return false;
          } else {
            if (!I32C(posAddr) || !e.writeOp(Op::I32Load) || !e.writeVarU32(2) ||
                !e.writeVarU32(uint32_t(endOff)))
              return false;
            if (!GetLocal(e, be.allocPosLocal) || !loadG(&gWJCtorSize[inlSite]) ||
                !e.writeOp(Op::I32Add) || !e.writeOp(Op::I32GeU))
              return false;
          }
          if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;  // room-If
          if (helperAlloc) {
            // gWJNewShapeSlot = &gWJCtorShape[inlSite]; WJH_CTORALLOC; this = scratch[kWJThisSlot]
            if (!I32C(int32_t(uintptr_t(static_cast<void*>(&gWJNewShapeSlot)))) ||
                !I32C(int32_t(uintptr_t(static_cast<void*>(&gWJCtorShape[inlSite])))) ||
                !store32(0))
              return false;
            if (!e.writeOp(Op::F64Const) || !e.writeFixedF64(double(WJH_CTORALLOC)) ||
                !e.writeOp(Op::F64Const) || !e.writeFixedF64(0.0) ||
                !e.writeOp(Op::Call) || !e.writeVarU32(0) || !e.writeOp(Op::Drop))
              return false;
            if (!GetLocal(e, be.scratchBase) || !e.writeOp(Op::I64Load) ||
                !e.writeVarU32(3) || !e.writeVarU32(kWJThisOff) ||
                !e.writeOp(Op::I32WrapI64) || !e.writeOp(Op::LocalSet) ||
                !e.writeVarU32(be.allocObjLocal))
              return false;
          } else {
          // *posAddr = allocPos + size
          if (!I32C(posAddr) || !GetLocal(e, be.allocPosLocal) ||
              !loadG(&gWJCtorSize[inlSite]) || !e.writeOp(Op::I32Add) || !store32(0))
            return false;
          // header @ allocPos
          if (!GetLocal(e, be.allocPosLocal) || !I32C(int32_t(gWJObjHeaderWord)) ||
              !store32(0))
            return false;
          // obj = allocPos + headerSize
          if (!GetLocal(e, be.allocPosLocal) || !I32C(int32_t(headerSize)) ||
              !e.writeOp(Op::I32Add) || !e.writeOp(Op::LocalSet) ||
              !e.writeVarU32(be.allocObjLocal))
            return false;
          // shape, slots, elements
          if (!GetLocal(e, be.allocObjLocal) || !loadG(&gWJCtorShape[inlSite]) ||
              !store32(shapeOff))
            return false;
          if (!GetLocal(e, be.allocObjLocal) || !I32C(emptySlots) ||
              !store32(slotsOff))
            return false;
          if (!GetLocal(e, be.allocObjLocal) || !I32C(emptyElems) ||
              !store32(elemsOff))
            return false;
          // slot-init loop: for (i=0; i<nfixed; i++) obj[fixedBase+i*8] = undefined
          if (!I32C(0) || !e.writeOp(Op::LocalSet) || !e.writeVarU32(be.propShapeLocal))
            return false;
          if (!e.writeOp(Op::Loop) || !e.writeFixedU8(0x40)) return false;
          if (!GetLocal(e, be.propShapeLocal) || !loadG(&gWJCtorNfixed[inlSite]) ||
              !e.writeOp(Op::I32LtU))
            return false;
          if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
          if (!GetLocal(e, be.allocObjLocal) || !I32C(int32_t(fixedBase)) ||
              !e.writeOp(Op::I32Add) || !GetLocal(e, be.propShapeLocal) || !I32C(8) ||
              !e.writeOp(Op::I32Mul) || !e.writeOp(Op::I32Add))
            return false;
          if (!e.writeOp(Op::I64Const) || !e.writeVarS64(undefBits) ||
              !e.writeOp(Op::I64Store) || !e.writeVarU32(3) || !e.writeVarU32(0))
            return false;
          if (!GetLocal(e, be.propShapeLocal) || !I32C(1) || !e.writeOp(Op::I32Add) ||
              !e.writeOp(Op::LocalSet) || !e.writeVarU32(be.propShapeLocal))
            return false;
          if (!e.writeOp(Op::Br) || !e.writeVarU32(1)) return false;  // re-loop
          if (!e.writeOp(Op::End)) return false;                      // end inner-If
          if (!e.writeOp(Op::End)) return false;                      // end Loop
          }  // end !helperAlloc manual-bump
          // gWJCurrentEnv = callee->environment()
          if (!I32C(envAddr) || !GetLocal(e, be.callScratch) ||
              !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(envOff) ||
              !store32(0))
            return false;
          // gWJCallCallee = boxed ctor callee (runtime-callee for MCallee; see
          // usesCallee -- same set-before-transfer invariant as gWJCurrentEnv).
          {
            uintptr_t ctorCalleeAddr = uintptr_t(static_cast<void*>(&gWJCallCallee));
            if (!I32C(int32_t(ctorCalleeAddr))) return false;
            if (!GetLocal(e, be.callScratch) || !e.writeOp(Op::I64ExtendI32U) ||
                !e.writeOp(Op::I64Const) ||
                !e.writeVarS64(int64_t(uint64_t(TagWord(JSVAL_TYPE_OBJECT)) << 32)) ||
                !e.writeOp(Op::I64Or) ||
                !e.writeOp(Op::I64Store) || !e.writeVarU32(3) || !e.writeVarU32(0))
              return false;
          }
          // GC-root spill across the ctor call_indirect (it may GC; the caller's
          // live locals AND the freshly-allocated `this` must survive + reload).
          // Mirrors EmitHelperCallResult. WITHOUT this, deltablue/raytrace corrupt
          // at default nursery (NO_NURSERY masks it).
          std::vector<uint32_t> croot;
          std::vector<uint8_t> crootObj;
          WJCollectRoots(be, ins, croot, crootObj);
          if (WJGETENV("GECKO_WJ_CROOTDBG") && gWJCurScriptLine ==
              uint32_t(atoi(WJGETENV("GECKO_WJ_CROOTDBG")))) {
            fprintf(stderr, "[crootdbg] line=%u construct croot locals:", gWJCurScriptLine);
            for (uint32_t k = 0; k < croot.size(); k++)
              fprintf(stderr, " %u(obj=%d)", croot[k], crootObj[k]);
            fprintf(stderr, "\n");
          }
          // Exclude the construct's `this` operand from the spill: it currently holds
          // the JS_IS_CONSTRUCTING MAGIC sentinel (not a real object). rootAll types
          // CreateThis as Object, so spilling it would box the magic bits as an
          // ObjectValue -> a GC during the ctor traces them as a JSObject* -> crash.
          // The REAL `this` is allocObjLocal, rooted separately at slot cn below.
          int thisLocal = be.local(call->getArg(0));
          if (thisLocal >= 0) {
            for (uint32_t k = 0; k < croot.size(); k++) {
              if (croot[k] == uint32_t(thisLocal)) {
                croot.erase(croot.begin() + k);
                crootObj.erase(crootObj.begin() + k);
                break;
              }
            }
          }
          uint32_t cn = uint32_t(croot.size());
          uintptr_t crootsBase = uintptr_t(static_cast<void*>(&gWJCallRoots[0]));
          uintptr_t cspAddr = uintptr_t(static_cast<void*>(&gWJRootSP));
          auto cspAdj = [&](int32_t d) -> bool {
            return I32C(int32_t(cspAddr)) && I32C(int32_t(cspAddr)) &&
                   e.writeOp(Op::I32Load) && e.writeVarU32(2) && e.writeVarU32(0) &&
                   I32C(d) && e.writeOp(Op::I32Add) && e.writeOp(Op::I32Store) &&
                   e.writeVarU32(2) && e.writeVarU32(0);
          };
          static int noCtorRoot = WJGETENV("GECKO_WJ_CTORNOROOT") ? 1 : 0;
          if (!noCtorRoot) {
            // rootBaseLocal = gWJRootSP*8 + rootsBase
            if (!I32C(int32_t(cspAddr)) || !e.writeOp(Op::I32Load) ||
                !e.writeVarU32(2) || !e.writeVarU32(0) || !I32C(8) ||
                !e.writeOp(Op::I32Mul) || !I32C(int32_t(crootsBase)) ||
                !e.writeOp(Op::I32Add) || !e.writeOp(Op::LocalSet) ||
                !e.writeVarU32(be.rootBaseLocal))
              return false;
            for (uint32_t k = 0; k < cn; k++) {
              if (!GetLocal(e, be.rootBaseLocal) || !GetLocal(e, croot[k])) return false;
              if (crootObj[k] == 1 && !EmitBoxFromStack(e, MIRType::Object)) return false;
              if (crootObj[k] == 2 && !EmitBoxFromStack(e, MIRType::String)) return false;
              if (crootObj[k] == 3 && !EmitBoxFromStack(e, MIRType::Symbol)) return false;
              if (crootObj[k] == 4 && !EmitBoxFromStack(e, MIRType::BigInt)) return false;
              if (!e.writeOp(Op::I64Store) || !e.writeVarU32(3) || !e.writeVarU32(k * 8))
                return false;
            }
            // Spill `this` at shadow slot cn (nesting-safe: the ctor inherits gWJRootSP
            // and bumps ABOVE this slot, so a GC inside the ctor traces+relocates it).
            // gWJScratch is NOT safe here -- the ctor clobbers kWJThisSlot via its own
            // calls/constructs, so the post-ctor reload reads a stale pointer.
            if (!GetLocal(e, be.rootBaseLocal) || !GetLocal(e, be.allocObjLocal) ||
                !EmitBoxFromStack(e, MIRType::Object) || !e.writeOp(Op::I64Store) ||
                !e.writeVarU32(3) || !e.writeVarU32(cn * 8))
              return false;
            if (!cspAdj(int32_t(cn) + 1)) return false;  // cn caller roots + `this`
          }
          // call_indirect(sb, this=box(obj), args..., pad, tblidx)
          if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(0)) return false;  // sb
          if (!GetLocal(e, be.allocObjLocal) ||
              !EmitBoxFromStack(e, MIRType::Object))
            return false;  // this
          for (uint32_t i = 0; i < cargc; i++) {
            if (!EmitSpillValue(e, be, call->getArg(1 + i))) return false;
          }
          for (uint32_t i = cargc; i < kWJMaxArgs; i++) {
            if (!e.writeOp(Op::I64Const) || !e.writeVarS64(undefBits)) return false;
          }
          if (!loadG(&gWJCtorTblIdx[inlSite])) return false;  // tblidx
          if (!e.writeOp(Op::CallIndirect) || !e.writeVarU32(0) || !e.writeVarU32(0))
            return false;
          be.rematInvalidate();  // moving GC point (ctor ran user code/allocs)
          if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(be.callResultLocal))
            return false;
          if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(be.callFlagLocal))
            return false;
          // SP -= cn+1; reload caller roots + `this` (GC may have moved them).
          if (!noCtorRoot) {
            if (!cspAdj(-(int32_t(cn) + 1))) return false;
            for (uint32_t k = 0; k < cn; k++) {
              if (!GetLocal(e, be.rootBaseLocal) || !e.writeOp(Op::I64Load) ||
                  !e.writeVarU32(3) || !e.writeVarU32(k * 8))
                return false;
              if (crootObj[k] && !e.writeOp(Op::I32WrapI64)) return false;
              if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(croot[k])) return false;
            }
            // reload `this` (GC-relocated) from shadow slot cn.
            if (!GetLocal(e, be.rootBaseLocal) || !e.writeOp(Op::I64Load) ||
                !e.writeVarU32(3) || !e.writeVarU32(cn * 8) ||
                !e.writeOp(Op::I32WrapI64) || !e.writeOp(Op::LocalSet) ||
                !e.writeVarU32(be.allocObjLocal))
              return false;
          }
          // VALIDATOR (GECKO_WJ_CTORINLINEDBG): abort if the reloaded `this`/roots are
          // a freed/poisoned cell -> pinpoints the residual GC crash's bogus value.
          if (WJGETENV("GECKO_WJ_CTORINLINEDBG")) {
            uintptr_t hobj = uintptr_t(static_cast<void*>(&gWJHelpObj));
            auto chk = [&](uint32_t loc, double site) -> bool {
              return I32C(int32_t(hobj)) && GetLocal(e, loc) &&
                     e.writeOp(Op::I32Store) && e.writeVarU32(2) && e.writeVarU32(0) &&
                     e.writeOp(Op::F64Const) && e.writeFixedF64(double(WJH_CHECKCELL)) &&
                     e.writeOp(Op::F64Const) && e.writeFixedF64(site) &&
                     e.writeOp(Op::Call) && e.writeVarU32(0) && e.writeOp(Op::Drop);
            };
            if (!chk(be.allocObjLocal, 5.0)) return false;  // `this`
            for (uint32_t k = 0; k < cn; k++)
              if (crootObj[k] == 1 && !chk(croot[k], 6.0)) return false;  // object roots
          }
          // exception: flag == 1 -> return [1.0, 0]. flag == 2 (entry GGG
          // mismatch: the ctor did NOT run) falls through to the slow
          // WJH_CONSTRUCT path below instead.
          if (!GetLocal(e, be.callFlagLocal) || !e.writeOp(Op::F64Const) ||
              !e.writeFixedF64(1.0) || !e.writeOp(Op::F64Eq))
            return false;
          if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
          if (!e.writeOp(Op::F64Const) || !e.writeFixedF64(1.0) ||
              !e.writeOp(Op::I64Const) || !e.writeVarS64(0) || !e.writeOp(Op::Return))
            return false;
          if (!e.writeOp(Op::End)) return false;
          // flag != 2: complete inline (result select + Br out). flag == 2
          // falls through this void If to the slow path.
          if (!GetLocal(e, be.callFlagLocal) || !e.writeOp(Op::F64Const) ||
              !e.writeFixedF64(2.0) || !e.writeOp(Op::F64Ne))
            return false;
          if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
          // result = ctorRet.isObject() ? ctorRet : box(this)
          if (!GetLocal(e, be.callResultLocal) || !e.writeOp(Op::I64Const) ||
              !e.writeVarS64(32) || !e.writeOp(Op::I64ShrU) ||
              !e.writeOp(Op::I32WrapI64) || !e.writeOp(Op::I32Const) ||
              !e.writeVarS32(int32_t(TagWord(JSVAL_TYPE_OBJECT))) ||
              !e.writeOp(Op::I32Eq))
            return false;
          if (!e.writeOp(Op::If) || !e.writeFixedU8(uint8_t(TypeCode::I64)))
            return false;
          if (!GetLocal(e, be.callResultLocal)) return false;
          if (!e.writeOp(Op::Else)) return false;
          if (!GetLocal(e, be.allocObjLocal) ||
              !EmitBoxFromStack(e, MIRType::Object))
            return false;
          if (!e.writeOp(Op::End)) return false;        // end result-If (i64 on stack)
          if (!e.writeOp(Op::Br) || !e.writeVarU32(4)) return false;  // -> Block
          if (!e.writeOp(Op::End)) return false;        // end flag!=2-If
          if (!e.writeOp(Op::End)) return false;        // end room-If
          if (!e.writeOp(Op::End)) return false;        // end gate-If
          if (!e.writeOp(Op::End)) return false;        // end cache-If
          // fall-through (miss/nursery-full/empty-cache): the slow WJH_CONSTRUCT path
          // below runs, leaving the boxed i64 result as the Block's value.
        }
        if (thisAllocates || thisIsCreateThis) {
          // callee/newTarget were pre-staged at the allocating `this` instruction
          // (traced 62/63, GC-current). BUT RE-STAGE the object/value ARGS here from
          // their now-current locals: the pre-stage at CreateThis/NewObject captured
          // the arg locals "while fresh", which is WRONG if CreateThis was scheduled
          // BEFORE an arg-producing call (raytrace getRay `new Ray(pos, dir.normalize())`
          // -- the pre-stage spilled a STALE local from a prior invocation; under a
          // minor GC that old object is freed -> wrong render). rootAll reloaded the
          // arg locals across the CreateThis/alloc helper, so they are GC-current here.
          for (uint32_t i = 0; i < cargc; i++) {
            MDefinition* a = call->getArg(1 + i);
            if (a->type() == MIRType::Object || a->type() == MIRType::Value) {
              if (!EmitStageScratch(e, be, a, i)) return false;
            }
          }
        } else {
          // No allocation between operand load and here (e.g. MagicIsConstructing
          // `this`); locals are fresh -> stage everything normally.
          for (uint32_t i = 0; i < cargc; i++) {
            if (!EmitStageScratch(e, be, call->getArg(1 + i), i)) return false;
          }
          if (!EmitStageScratch(e, be, call->getCallee(), kWJCalleeSlot))
            return false;
          if (!EmitStageScratch(e, be, call->getArg(call->numStackArgs() - 1),
                                kWJNewTargetSlot))
            return false;
        }
        if (!EmitStageScratch(e, be, call->getArg(0), kWJThisSlot)) return false;
        uintptr_t aAddr = uintptr_t(static_cast<void*>(&gWJCallArgc));
        if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(aAddr)) ||
            !e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(cargc)) ||
            !e.writeOp(Op::I32Store) || !e.writeVarU32(2) || !e.writeVarU32(0))
          return false;
        // Slow path: WJH_CONSTRUCT(inlSite). It also FILLS the per-site cache (so the
        // inline fast path above handles this site next time). Leaves the boxed i64.
        if (!EmitHelperCallResult(e, be, ins, WJH_CONSTRUCT, inlSite)) return false;
        if (inlSite) {
          // close the Block opened by the inline fast path; both the hit (Br 2) and
          // this miss path leave the boxed i64 as the Block's result.
          if (!e.writeOp(Op::End)) return false;
        }
        return true;
      }
      uint32_t argc = call->numActualArgs();
      // WIDE calls (argc > kWJMaxArgs): the 8-reg fast ABI can't carry them, but
      // the slow WJH_CALL path stages args in gWJScratch[0..argc-1] (traced slots
      // 0..61 are free below callee@62) -- route them there instead of bailing
      // the whole fn to PBL (typescript resolveNameHelper: 9-arg calls stranded
      // 4 hot fns). GECKO_WJ_NOWIDECALL reverts.
      bool wideCall = argc > kWJMaxArgs;
      if (argc > 61 || (wideCall && WJGETENV("GECKO_WJ_NOWIDECALL"))) {
        return false;  // beyond the scratch-staged slow path too
      }
      // Box `this` + actual args into the call-arg locals (reused by both the
      // fast register-call path and the slow scratch+wjhelp path). Pad unused
      // arg slots with undefined. WIDE calls skip this (slow path stages the
      // locals' values straight into scratch below).
      if (!wideCall) {
      if (!EmitSpillValue(e, be, call->getArg(0))) return false;  // this
      if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(be.callArgBase)) return false;
      for (uint32_t i = 0; i < kWJMaxArgs; i++) {
        if (i < argc) {
          if (!EmitSpillValue(e, be, call->getArg(1 + i))) return false;
        } else if (!e.writeOp(Op::I64Const) ||
                   !e.writeVarS64(int64_t(JS::UndefinedValue().asRawBits()))) {
          return false;
        }
        if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(be.callArgBase + 1 + i))
          return false;
      }
      }  // !wideCall

      // GC-root shadow stack: spill live object/Value locals (which the GC can't
      // see in wasm locals) so a moving minor GC inside the callee updates them.
      std::vector<uint32_t> rootLocal;
      std::vector<uint8_t> rootIsObj;
      WJCollectRoots(be, ins, rootLocal, rootIsObj);
      static int noRoot = WJGETENV("GECKO_WJ_NOROOT") ? 1 : 0;
      if (noRoot) {
        rootLocal.clear();
        rootIsObj.clear();
      }
      uint32_t nRoots = uint32_t(rootLocal.size());
      uintptr_t rootsBase = uintptr_t(static_cast<void*>(&gWJCallRoots[0]));
      uintptr_t spAddr = uintptr_t(static_cast<void*>(&gWJRootSP));
      auto emitSPAdjust = [&](int32_t delta) -> bool {
        return e.writeOp(Op::I32Const) && e.writeVarS32(int32_t(spAddr)) &&
               e.writeOp(Op::I32Const) && e.writeVarS32(int32_t(spAddr)) &&
               e.writeOp(Op::I32Load) && e.writeVarU32(2) && e.writeVarU32(0) &&
               e.writeOp(Op::I32Const) && e.writeVarS32(delta) &&
               e.writeOp(Op::I32Add) && e.writeOp(Op::I32Store) &&
               e.writeVarU32(2) && e.writeVarU32(0);
      };
      // Spill base computed ONCE (= gWJRootSP*8 + rootsBase); slot k = base + k*8 via
      // the store/load immediate offset. SP is unchanged between spill and reload, so
      // rootBaseLocal serves both. Avoids reloading gWJRootSP + *8+base per root (the
      // splay GC-root-spill lever; survives the call_indirect -- wasm locals do).
      if (nRoots) {
        if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(spAddr)) ||
            !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0) ||
            !e.writeOp(Op::I32Const) || !e.writeVarS32(8) || !e.writeOp(Op::I32Mul) ||
            !e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(rootsBase)) ||
            !e.writeOp(Op::I32Add) || !e.writeOp(Op::LocalSet) ||
            !e.writeVarU32(be.rootBaseLocal))
          return false;
      }
      auto emitSlotBase = [&]() -> bool { return GetLocal(e, be.rootBaseLocal); };
      for (uint32_t k = 0; k < nRoots; k++) {
        if (!emitSlotBase()) return false;
        if (!GetLocal(e, rootLocal[k])) return false;
        if (rootIsObj[k] == 1 && !EmitBoxFromStack(e, MIRType::Object)) return false;
    if (rootIsObj[k] == 2 && !EmitBoxFromStack(e, MIRType::String)) return false;
    if (rootIsObj[k] == 3 && !EmitBoxFromStack(e, MIRType::Symbol)) return false;
    if (rootIsObj[k] == 4 && !EmitBoxFromStack(e, MIRType::BigInt)) return false;
        if (!e.writeOp(Op::I64Store) || !e.writeVarU32(3) || !e.writeVarU32(k * 8))
          return false;
      }
      if (nRoots && !emitSPAdjust(int32_t(nRoots))) return false;

      uint32_t site = WJAllocCallSite();
      if (site) gWJCallSiteLine[site] = gWJCurScriptLine;  // DEBUG: caller line
      // DEBUG (GECKO_WJ_CALLARGDBG=<line>): map call-site serials to their MIR
      // arg sources for one fn -- pairs with the VALIDATEHELPER stale-arg site
      // number to name the exact def feeding a corrupt argument.
      if (WJGETENV("GECKO_WJ_CALLARGDBG") && gWJCurScriptLine ==
          uint32_t(atoi(WJGETENV("GECKO_WJ_CALLARGDBG")))) {
        fprintf(stderr, "[wj-callarg] line=%u site=%u argc=%u", gWJCurScriptLine,
                site, argc);
        for (uint32_t a = 0; a < argc && a < 4; a++) {
          MDefinition* ad = call->getArg(1 + a);
          fprintf(stderr, " arg%u={%s#%u:%s local=%d blk=%u}", a,
                  WJOpName(ad->op()), unsigned(ad->id()),
                  StringFromMIRType(ad->type()), be.local(ad),
                  ad->block() ? unsigned(ad->block()->id()) : 0u);
        }
        fprintf(stderr, "\n");
      }
      uintptr_t calleeAddr = uintptr_t(static_cast<void*>(&gWJCallCallee));
      uintptr_t argcAddr = uintptr_t(static_cast<void*>(&gWJCallArgc));

      // callScratch = actual callee object pointer.
      if (WJGETENV("GECKO_WJ_CALLEEOP") && be.info && be.info->script() &&
          uint32_t(be.info->script()->lineno()) ==
              uint32_t(atoi(WJGETENV("GECKO_WJ_CALLEEOP")))) {
        jit::MDefinition* ce = call->getCallee();
        fprintf(stderr, "[calleeop] line=%u callee op=%s type=%s local=%d id=%u block=%u callId=%u nargs=%u\n",
                gWJCurScriptLine, WJOpName(ce->op()), StringFromMIRType(ce->type()),
                be.local(ce), unsigned(ce->id()),
                ce->block() ? unsigned(ce->block()->id()) : 0u, unsigned(ins->id()),
                call->numActualArgs());
        // walk the callee's operand chain (slots base / proto source)
        jit::MDefinition* d = ce;
        for (int depth = 0; d && depth < 6; depth++) {
          fprintf(stderr, "    [chain %d] op=%s type=%s local=%d nops=%u\n", depth,
                  WJOpName(d->op()), StringFromMIRType(d->type()), be.local(d),
                  unsigned(d->numOperands()));
          if (d->numOperands() == 0) break;
          d = d->getOperand(0);
        }
      }
      if (!EmitObjPtr(e, be, call->getCallee())) return false;
      if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(be.callScratch)) return false;

      // gWJCallArgc = actual argc: the callee snapshots it at entry for ArgumentsLength.
      // Set on ALL dispatch paths (fast call_indirect ways AND the slow WJH_CALL hop)
      // here -- after arg evaluation (any nested calls have finished overwriting the
      // global) and before transferring control. The fast register path previously
      // skipped this (it passes a fixed kWJMaxArgs padded args), so without it a callee
      // reading gWJCallArgc would see a stale count. (The slow path re-stores the same
      // value below; harmless.)
      if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(argcAddr)) ||
          !e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(argc)) ||
          !e.writeOp(Op::I32Store) || !e.writeVarU32(2) || !e.writeVarU32(0))
        return false;
      // Polymorphic inline cache. block (result f64) { for each way: if the
      // cached callee matches (and its table slot is in range) do a direct
      // wasm->wasm call_indirect and br out with the result; else fall to the
      // next way, then to the slow wjhelp hop }. Way 0 first => monomorphic
      // calls cost one compare; polymorphic dispatch (task.run) stays in wasm.
      if (!e.writeOp(Op::Block) || !e.writeFixedU8(0x40)) return false;  // void
      static int forceSlowCallEnv = WJGETENV("GECKO_WJ_FORCESLOWCALL") ? 1 : 0;
      // Inlining-containing functions route calls through the slow (clean-boundary)
      // path: the fast call_indirect corrupts on a deopt nested under an inlined
      // caller. NOINLINESLOWCALL disables this targeted guard (for A/B).
      static int noInlineSlowCall = WJGETENV("GECKO_WJ_NOINLINESLOWCALL") ? 1 : 0;
      // An inlined-frame function forces calls through the slow (clean-boundary) path:
      // the fast call_indirect under an inlined caller corrupts when an error/THROW
      // triggers FrameIter stack-decompilation through the inlined fast-call frames
      // (JS_CODEGEN_NONE can't walk them -> "unreachable"). The old "crypto +80%,
      // navier +13%" win from removing this is STALE: re-measured 2026-07-11 best-of-3
      // idle (NOINLINESLOWCALL=1 vs default) = NEUTRAL-to-NEGATIVE (crypto 1904/1925,
      // navier 11009/10873, richards 2348/2302, earley 2796/2930) -- the slow WJH_CALL
      // dispatch got as fast as the fast call_indirect (8192-way entry caches + fast
      // directIdx C-fn-ptr path). So this guard is now ~FREE; keep it (it also prevents
      // the ubo FrameIter-through-inlined-fast-call crash on a throw). NARROWINLINESLOW/
      // NOINLINESLOWCALL are not worth pursuing -- no perf win to capture. See memory
      // typescript-jit-profile-2026-07-11. NOINLINESLOWCALL removes it (unsound on throw).
      static int narrowSlow = WJGETENV("GECKO_WJ_NARROWINLINESLOW") ? 1 : 0;
      bool forceSlowCall =
          wideCall || forceSlowCallEnv ||
          (be.hasInlinedFrames && !noInlineSlowCall &&
           !(narrowSlow && !be.hasCatchTrynote));
      static uint32_t maxWays = WJGETENV("GECKO_WJ_MONOCALL") ? 1 : kWJCallWays;
      for (uint32_t w = 0; w < maxWays && !forceSlowCall; w++) {
        uintptr_t fnW =
            uintptr_t(static_cast<void*>(&gWJCallFn[site * kWJCallWays + w]));
        uintptr_t tblW =
            uintptr_t(static_cast<void*>(&gWJCallTblIdx[site * kWJCallWays + w]));
        if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(fnW)) ||
            !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0))
          return false;
        if (!GetLocal(e, be.callScratch) || !e.writeOp(Op::I32Eq)) return false;
        if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(tblW)) ||
            !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0))
          return false;
        if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(4096)) ||
            !e.writeOp(Op::I32LtU) || !e.writeOp(Op::I32And))
          return false;
        if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
        // gWJCurrentEnv = callee->environment(): the callee's prologue saves this
        // into its envLocal so its MFunctionEnvironment is correct (closures).
        {
          uintptr_t envAddr = uintptr_t(static_cast<void*>(&gWJCurrentEnv));
          if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(envAddr)))
            return false;
          if (!GetLocal(e, be.callScratch) || !e.writeOp(Op::I32Load) ||
              !e.writeVarU32(2) ||
              !e.writeVarU32(uint32_t(JSFunction::offsetOfEnvironment())))
            return false;
          if (!e.writeOp(Op::I32Store) || !e.writeVarU32(2) || !e.writeVarU32(0))
            return false;
        }
        // gWJCallCallee = boxed callee: the callee's prologue roots it for
        // MCallee (runtime-callee; see usesCallee). Same set-before-transfer
        // invariant as gWJCurrentEnv/gWJCallArgc.
        {
          if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(calleeAddr)))
            return false;
          if (!GetLocal(e, be.callScratch) || !e.writeOp(Op::I64ExtendI32U) ||
              !e.writeOp(Op::I64Const) ||
              !e.writeVarS64(int64_t(uint64_t(TagWord(JSVAL_TYPE_OBJECT)) << 32)) ||
              !e.writeOp(Op::I64Or))
            return false;
          if (!e.writeOp(Op::I64Store) || !e.writeVarU32(3) || !e.writeVarU32(0))
            return false;
        }
        // Register call: push sb (f64) [EHABI: + boxed callee] + this + args (i64) +
        // table index, then call_indirect the callee's main directly.
        if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(0)) return false;  // sb
        if constexpr (kWJEHABI) {
          // boxed callee in ABI slot 1 (callScratch holds the i32 callee ptr).
          if (!GetLocal(e, be.callScratch) || !EmitBoxFromStack(e, MIRType::Object))
            return false;
        }
        for (uint32_t a = 0; a <= kWJMaxArgs; a++) {  // this + kWJMaxArgs args
          if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(be.callArgBase + a))
            return false;
        }
        if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(tblW)) ||
            !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0))
          return false;  // table index
        if (!e.writeOp(Op::CallIndirect) || !e.writeVarU32(0) || !e.writeVarU32(0))
          return false;  // EHABI: type0 -> [i64 result] (exn via EH); old -> [flag,result]
        be.rematInvalidate();  // moving GC point (callee ran user code/allocs)
        if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(be.callResultLocal))
          return false;  // pop result (top)
        if constexpr (!kWJEHABI) {
          if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(be.callFlagLocal))
            return false;  // pop flag (old ABI)
        }
        // flag==2.0 = the callee did NOT run (entry GuardGlobalGeneration
        // mismatch): fall through to the slow path (JS::Call -> PBL) instead of
        // branching out with a bogus result. No resume machinery => no nesting.
        if (!GetLocal(e, be.callFlagLocal) || !e.writeOp(Op::F64Const) ||
            !e.writeFixedF64(2.0) || !e.writeOp(Op::F64Ne))
          return false;
        if (!e.writeOp(Op::BrIf) || !e.writeVarU32(1)) return false;  // -> block end
        if (!e.writeOp(Op::End)) return false;                      // end if (way w)
      }
      {  // SLOW: store this+args to scratch (the callee's trampoline / JS::Call
        // reads them), set callee+argc, wjhelp(WJH_CALL, site).
        for (uint32_t a = 0; a < argc; a++) {
          if (!GetLocal(e, be.scratchBase)) return false;
          if (wideCall) {
            // Wide call: box the arg def straight into scratch (no callArgBase
            // staging happened; values come from already-computed locals, and
            // nothing GC-capable runs between these stores and WJH_CALL).
            if (!EmitSpillValue(e, be, call->getArg(1 + a))) return false;
          } else if (!GetLocal(e, be.callArgBase + 1 + a)) {
            return false;
          }
          if (!e.writeOp(Op::I64Store) || !e.writeVarU32(3) ||
              !e.writeVarU32(a * 8))
            return false;
        }
        if (!GetLocal(e, be.scratchBase)) return false;
        if (wideCall) {
          if (!EmitSpillValue(e, be, call->getArg(0))) return false;
        } else if (!GetLocal(e, be.callArgBase)) {
          return false;
        }
        if (!e.writeOp(Op::I64Store) || !e.writeVarU32(3) ||
            !e.writeVarU32(kWJThisOff))
          return false;
        if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(calleeAddr))) return false;
        if (!EmitSpillValue(e, be, call->getCallee())) return false;
        if (!e.writeOp(Op::I64Store) || !e.writeVarU32(3) || !e.writeVarU32(0)) return false;
        if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(argcAddr)) ||
            !e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(argc)) ||
            !e.writeOp(Op::I32Store) || !e.writeVarU32(2) || !e.writeVarU32(0)) {
          return false;
        }
        if (!e.writeOp(Op::F64Const) || !e.writeFixedF64(double(WJH_CALL)) ||
            !e.writeOp(Op::F64Const) || !e.writeFixedF64(double(site)) ||
            !e.writeOp(Op::Call) || !e.writeVarU32(0)) {
          return false;  // wjhelp(WJH_CALL, site) -> f64 flag
        }
        if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(be.callFlagLocal))
          return false;  // flag
        if constexpr (kWJEHABI) {
          // EHABI: the WJH_CALL helper still signals via the flag (it can't natively
          // throw a wasm exn); convert it to a wasm throw here so it joins the EH path.
          // (The FAST inline call_indirect path propagates the callee's throw natively
          // and never reaches here.) Unwinds to a try-region catch_all or the trampoline.
          if (!GetLocal(e, be.callFlagLocal) || !e.writeOp(Op::F64Const) ||
              !e.writeFixedF64(0.0) || !e.writeOp(Op::F64Ne))
            return false;
          if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
          if (!EmitThrowWjexn(e)) return false;
          if (!e.writeOp(Op::End)) return false;
        }
        // wjhelp left the boxed result in gWJScratch[kWJResultOff]; load it.
        if (!GetLocal(e, be.scratchBase) || !e.writeOp(Op::I64Load) ||
            !e.writeVarU32(3) || !e.writeVarU32(kWJResultOff) ||
            !e.writeOp(Op::LocalSet) || !e.writeVarU32(be.callResultLocal))
          return false;
      }
      if (!e.writeOp(Op::End)) return false;  // end block (void)
      // Pop the GC-root frame (runs on both the normal and exception paths).
      if (nRoots && !emitSPAdjust(-int32_t(nRoots))) return false;
      // Propagate a pending exception (flag != 0 -> return [1.0, dummy]). EHABI handles
      // this inside the slow path (throw) + the fast path via native EH, so the
      // post-block flag check is old-ABI only (the fast path doesn't set the flag under
      // EHABI, so a stale flag here would spuriously throw).
      // Reload the rooted locals (GC may have moved the objects) BEFORE the flag check,
      // so an in-try deopt-in-error (EmitExceptionExit) spills GC-CURRENT locals.
      for (uint32_t k = 0; k < nRoots; k++) {
        if (!emitSlotBase()) return false;
        if (!e.writeOp(Op::I64Load) || !e.writeVarU32(3) || !e.writeVarU32(k * 8))
          return false;
        if (rootIsObj[k] && !e.writeOp(Op::I32WrapI64)) return false;
        if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(rootLocal[k])) return false;
      }
      static int noFlagCheck6 = WJGETENV("GECKO_WJ_NOFLAGCHECK") ? 1 : 0;
      if (!kWJEHABI && !noFlagCheck6) {
      if (!GetLocal(e, be.callFlagLocal) || !e.writeOp(Op::F64Const) ||
          !e.writeFixedF64(0.0) || !e.writeOp(Op::F64Ne)) {
        return false;
      }
      if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
      // in-try -> deopt-in-error (PBL runs the catch); else [1.0,0] propagate.
      if (!EmitExceptionExit(e, be)) return false;
      if (!e.writeOp(Op::End)) return false;
      }  // end !noFlagCheck6
      // The call's value is the boxed result returned in a register.
      return GetLocal(e, be.callResultLocal);
    }
    case MDefinition::Opcode::Compare: {
      MCompare* c = ins->toCompare();
      MDefinition* l = c->getOperand(0);
      MDefinition* r = c->getOperand(1);
      MCompare::CompareType ct0 = c->compareType();
      // `x == null` / `x == undefined`: a tag check on the tested Value operand
      // (the other operand is the null/undefined literal, which has no local).
      if (ct0 == MCompare::Compare_Null || ct0 == MCompare::Compare_Undefined) {
        MDefinition* tested = be.local(l) >= 0 ? l : r;
        if (be.local(tested) < 0) return false;
        if (tested->type() != MIRType::Value) {
          // A statically-typed non-null value: result is a constant.
          bool isEq = c->jsop() == JSOp::Eq || c->jsop() == JSOp::StrictEq;
          return e.writeOp(Op::I32Const) && e.writeVarS32(isEq ? 0 : 1);
        }
        if (!GetOp(e, be, tested)) return false;
        if (!e.writeOp(Op::I64Const) || !e.writeVarS64(32) ||
            !e.writeOp(Op::I64ShrU) || !e.writeOp(Op::I32WrapI64)) {
          return false;
        }
        bool strict = c->jsop() == JSOp::StrictEq || c->jsop() == JSOp::StrictNe;
        uint32_t litTag = ct0 == MCompare::Compare_Null
                              ? TagWord(JSVAL_TYPE_NULL)
                              : TagWord(JSVAL_TYPE_UNDEFINED);
        if (strict) {
          if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(litTag)) ||
              !e.writeOp(Op::I32Eq)) {
            return false;
          }
        } else {
          // loose: matches null OR undefined.
          if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(be.callScratch))
            return false;  // stash tag in a scratch i32
          if (!GetLocal(e, be.callScratch) ||
              !e.writeOp(Op::I32Const) ||
              !e.writeVarS32(int32_t(TagWord(JSVAL_TYPE_NULL))) ||
              !e.writeOp(Op::I32Eq) || !GetLocal(e, be.callScratch) ||
              !e.writeOp(Op::I32Const) ||
              !e.writeVarS32(int32_t(TagWord(JSVAL_TYPE_UNDEFINED))) ||
              !e.writeOp(Op::I32Eq) || !e.writeOp(Op::I32Or)) {
            return false;
          }
        }
        if (c->jsop() == JSOp::Ne || c->jsop() == JSOp::StrictNe) {
          if (!e.writeOp(Op::I32Eqz)) return false;  // negate
        }
        return true;
      }
      if (be.local(l) < 0 || be.local(r) < 0) {
        return WJBAIL("compare ct=%d lty=%s rty=%s (operand no local)\n",
                      int(ct0), StringFromMIRType(l->type()),
                      StringFromMIRType(r->type()));
      }
      MCompare::CompareType ct = c->compareType();
      // Float32 operands are held as f64 in our repr (WJValType maps Float32->f64),
      // and every f32 value is exactly representable as f64, so an f64 compare yields
      // the identical result -- treat Compare_Float32 exactly like Compare_Double.
      // (Closes the mandreel bail: Emscripten Math.fround math does Float32 compares.)
      bool dbl = ct == MCompare::Compare_Double || ct == MCompare::Compare_Float32;
      // Int32/Boolean/Object(+Null/Undefined ptr) all compare as i32 here.
      bool i32 = ct == MCompare::Compare_Int32 || ct == MCompare::Compare_Object ||
                 ct == MCompare::Compare_Null || ct == MCompare::Compare_Undefined;
      if (!dbl && !i32) {
        // String/Symbol/BigInt compare. Route through the WJH_COMPARE VM helper --
        // the SAME proven-sound path the untyped Boolean MBinaryCache compare uses
        // (default-on, see (A3) above): EmitStageScratch boxes each operand via
        // EmitSpillValue (which boxes a String-typed local correctly), and the helper
        // does the REAL js comparison (EqualStrings/CompareStrings -- correct for
        // interned AND dynamically-built strings, any jsop). The result is a boxed
        // boolean -> EmitHelperResultAsType(Boolean) unboxes its low 32 bits to i32,
        // matching the typed path's i32 output. This keeps the WHOLE function compiled
        // in JIT instead of bailing it to PBL -- date-format-tofte's arrayExists
        // (`array[i] == x`, inlined into the hot formatDate) bailed the entire fn here.
        // The earlier GECKO_WJ_STRCMP attempt produced a wrong pdfjs result; this one
        // reuses the boxing+helper primitives the MBinaryCache path proves sound.
        //
        // OPT-IN (GECKO_WJ_STRCMP2), default-OFF. Result is provably correct (the
        // helper does full LooselyEqual/StrictlyEqual), and this is a real 1.5x win on
        // date-format-tofte (arrayExists `array[i]==x` -> the whole hot formatDate
        // compiles instead of bailing to PBL). BUT it exposes a SEPARATE downstream
        // miscompile in pdfjs's dynamically-built-string parser (isSeparator:16935 +
        // ~6 other fns now compile their string compares -> "Malformed PDF" wrong
        // result). The compare RESULT is correct (verified: helper = real VM compare;
        // the Ion-faithful string type-guard below is also emitted) -- the regression
        // is a latent bug in some OTHER op in those now-fully-compiled fns, NOT this
        // compare. No operand-type discriminator separates the safe (date-tofte) from
        // the unsafe (pdfjs) case -- both are ct=Compare_String, String/String, jsop=Eq.
        // So default-OFF until the pdfjs downstream miscompile is found+fixed (then
        // flip default-on for the JetStream win). See memory + task.
        // OPT-IN (default-off, 2026-06-29). STRCMP2 (Compare_String via WJH_COMPARE)
        // is CORRECT itself, but enabling it lets pdfjs Lexer_getObj compile through to
        // a latent miscompile -> the pure-assertion-guard fix (GECKO_WJ_KEEPASSERTGUARD)
        // fixes pdfjs BUT that fix regresses acorn (deopt-resume corruption). Until the
        // deopt-resume bug (task #15) is fixed, STRCMP2 default-on breaks pdfjs, so it
        // stays OPT-IN. Win when unblocked: date-format-tofte 0.94x -> ~1.46x. Enable:
        // DEFAULT-ON (2026-06-29): unblocked by the valve-result fix + KEEPASSERTGUARD
        // (pdfjs getObj now correct, acorn no longer mis-parses). Win: date-format-tofte
        // 0.94x -> ~1.46x (arrayExists `array[i]==x` -> hot formatDate compiles). Revert:
        // NOTE (2026-07-04): an INLINE string-equality fast-path (ptrL==ptrR -> equal;
        // else lenL!=lenR -> not-equal; else WJH_COMPARE) was implemented + measured here
        // and REVERTED -- it gave NO reliable speedup. pdfjs's hot string compares are
        // RELATIONAL (jsop Lt/Le/Gt/Ge, lexicographic -- a length check can't shortcut
        // them), and even on equality-heavy code the win was noise (the WJH_COMPARE call
        // COUNT is high but the C++ hop is not the time bottleneck -- V8 compiles the call
        // well; per-call/boxing overhead dominates). gbemu was marginally WORSE (extra
        // branches before the inevitable helper). See [[pdfjs-laggard-fromcharcode]].
        constexpr int strCmp2 = 1;  // two-way string-compare path is permanent
        // Bisection: STRCMP2SKIP=N excludes the fn at lineno N; STRCMP2ONLY=N routes
        // ONLY that fn. Lets us env-bisect which now-fully-compiled pdfjs fn carries
        // the downstream miscompile, without a rebuild per step.
        static int strCmp2Skip =
            WJGETENV("GECKO_WJ_STRCMP2SKIP") ? atoi(WJGETENV("GECKO_WJ_STRCMP2SKIP")) : -1;
        static int strCmp2Only =
            WJGETENV("GECKO_WJ_STRCMP2ONLY") ? atoi(WJGETENV("GECKO_WJ_STRCMP2ONLY")) : -1;
        int curLine = be.info && be.info->script() ? int(be.info->script()->lineno()) : -1;
        bool skip = (strCmp2Skip >= 0 && curLine == strCmp2Skip) ||
                    (strCmp2Only >= 0 && curLine != strCmp2Only);
        // Within-fn pc bisect: STRCMP2MAXPC=N bails compares whose bytecode offset > N.
        static int strCmp2MaxPc =
            WJGETENV("GECKO_WJ_STRCMP2MAXPC") ? atoi(WJGETENV("GECKO_WJ_STRCMP2MAXPC")) : -1;
        if (strCmp2MaxPc >= 0) {
          const jit::BytecodeSite* st = ins->trackedSite();
          uint32_t off = (st && st->script() && st->pc())
                             ? uint32_t(st->pc() - st->script()->code()) : 0;
          if (int(off) > strCmp2MaxPc) skip = true;
        }
        if (strCmp2 && !skip && be.local(l) >= 0 && be.local(r) >= 0) {
          // Type-refinement guard (Ion-faithful): Compare_String/Symbol/BigInt implies
          // Ion GUARDED both operands to that type (MGuardIsString etc.), which REFINES
          // the operand's type for DOWNSTREAM code (uses assume it's a string). Routing
          // to WJH_COMPARE alone gives the correct boolean (the helper does full
          // LooselyEqual/StrictlyEqual for any types) BUT skips the refinement, so a
          // Value-typed operand that downstream treats as a guaranteed string is left
          // unrefined -> pdfjs's parser miscompiled (the GECKO_WJ_STRCMP regression).
          // Emit the tag-guard on each VALUE-typed operand (deopt to PBL if not the
          // proven type) so downstream sees a genuine string; String/Symbol/BigInt-
          // typed operands are already proven (no guard).
          if (WJGETENV("GECKO_WJ_STRCMPDBG")) {
            const jit::BytecodeSite* st = ins->trackedSite();
            uint32_t coff = (st && st->script() && st->pc())
                                ? uint32_t(st->pc() - st->script()->code()) : 0;
            uint32_t cline = (st && st->script() && st->pc())
                                 ? js::PCToLineNumber(st->script(), st->pc()) : 0;
            fprintf(stderr, "[strcmp2] fn:%u cmp@off=%u line=%u ct=%d jsop=%d lTy=%s rTy=%s rC=%d\n",
                    be.info && be.info->script() ? unsigned(be.info->script()->lineno()) : 0,
                    coff, cline, int(ct), int(c->jsop()), StringFromMIRType(l->type()),
                    StringFromMIRType(r->type()), int(r->isConstant()));
          }
          uint32_t wantTag =
              ct == MCompare::Compare_Symbol ? TagWord(JSVAL_TYPE_SYMBOL)
              : ct == MCompare::Compare_BigInt ? TagWord(JSVAL_TYPE_BIGINT)
                                               : TagWord(JSVAL_TYPE_STRING);
          static int noGuard = WJGETENV("GECKO_WJ_STRCMP2NOGUARD") ? 1 : 0;
          if (!noGuard && l->type() == MIRType::Value &&
              !EmitTagGuardLocal(e, be, uint32_t(be.local(l)), wantTag))
            return false;
          if (!noGuard && r->type() == MIRType::Value &&
              !EmitTagGuardLocal(e, be, uint32_t(be.local(r)), wantTag))
            return false;
          return EmitStageScratch(e, be, l, 0) && EmitStageScratch(e, be, r, 1) &&
                 EmitHelperCallResult(e, be, ins, WJH_COMPARE, uint32_t(c->jsop())) &&
                 EmitHelperResultAsType(e, be, MIRType::Boolean);
        }
        if (WJGETENV("GECKO_WJWARP_DUMP"))
          fprintf(stderr, "[wb-be] Compare type=%d lhsTy=%d unhandled\n",
                  int(ct), int(l->type()));
        return false;
      }
      // Operand-repr handling. Emitting i32.* on an f64-repr operand (or f64.* on an
      // i32) is INVALID wasm (V8 rejects the WHOLE module). For NUMERIC compares
      // (Int32/Double/Float32) with a MIXED operand repr (one f64, one i32), compare
      // as f64: int32->f64 is EXACT, so a mixed int32/double numeric compare yields the
      // SAME result as the f64 compare (unblocks ct=Compare_Int32 with a Double-repr
      // operand, ~16 suite bails). Both-i32 numeric -> i32 compare. Ref/identity
      // compares (Object/Null/Undefined) require i32 (pointer) operands. Only these
      // types reach here (String/Symbol/BigInt/UInt32 took the !dbl&&!i32 path above).
      bool refCmp = ct == MCompare::Compare_Object || ct == MCompare::Compare_Null ||
                    ct == MCompare::Compare_Undefined;
      const uint8_t kI32 = uint8_t(TypeCode::I32), kF64 = uint8_t(TypeCode::F64);
      uint8_t ltc = WJValType(l->type()), rtc = WJValType(r->type());
      bool useF64;
      static int noCmpF64 = WJGETENV("GECKO_WJ_NO_CMPF64") ? 1 : 0;
      if (refCmp) {
        if (ltc != kI32 || rtc != kI32) {
          js::wasm::gWJBailReason = "compare-ref-operand-repr";
          return false;
        }
        useF64 = false;
      } else {
        // numeric: each operand must be i32 or f64 repr.
        if ((ltc != kI32 && ltc != kF64) || (rtc != kI32 && rtc != kF64)) {
          js::wasm::gWJBailReason = "compare-operand-repr-mismatch";
          return false;
        }
        bool mixed = ltc != rtc;
        if (noCmpF64 && mixed) {  // revert path: old strict bail
          js::wasm::gWJBailReason = "compare-operand-repr-mismatch";
          return false;
        }
        useF64 = dbl || ltc == kF64 || rtc == kF64;
      }
      if (!GetOp(e, be, l)) return false;
      if (useF64 && ltc == kI32 && !e.writeOp(Op::F64ConvertI32S)) return false;
      if (!GetOp(e, be, r)) return false;
      if (useF64 && rtc == kI32 && !e.writeOp(Op::F64ConvertI32S)) return false;
      switch (c->jsop()) {
        case JSOp::Lt: return !refCmp && e.writeOp(useF64 ? Op::F64Lt : Op::I32LtS);
        case JSOp::Le: return !refCmp && e.writeOp(useF64 ? Op::F64Le : Op::I32LeS);
        case JSOp::Gt: return !refCmp && e.writeOp(useF64 ? Op::F64Gt : Op::I32GtS);
        case JSOp::Ge: return !refCmp && e.writeOp(useF64 ? Op::F64Ge : Op::I32GeS);
        case JSOp::Eq:
        case JSOp::StrictEq: return e.writeOp(useF64 ? Op::F64Eq : Op::I32Eq);
        case JSOp::Ne:
        case JSOp::StrictNe: return e.writeOp(useF64 ? Op::F64Ne : Op::I32Ne);
        default: return false;
      }
    }
    default:
      WJTallyBail(ins->op());
      return false;
  }
}

// Conservative whole-cell post-write barrier on `objDef`'s container: store the
// (current, rematerialized) object pointer to gWJHelpObj and call WJH_POSTBARRIER,
// which buffers it ONLY if tenured. Safe to call after any store whose value may be
// a GC pointer; over-buffers harmlessly. Used to repair barriers that Warp's
// MPostWriteBarrier insertion/elimination may have dropped under trial inlining
// (the deltablue-under-COLDCALL heap-slot staleness: NO_NURSERY fixes it, ROOTALL
// -- which roots locals, not heap slots -- does not). Returns false on emit failure.
static bool WJValueMightBeGCThing(jit::MIRType t);  // fwd
// True if `obj` already has an explicit MPostWriteBarrier node as a consumer (the
// container operand). Warp inserts MPostWriteBarrier after stores that need it (the
// Ion-faithful representation); when present, an inline store must NOT also emit its
// own post-barrier -- doing both DOUBLE-emits the (~10-op inline tenured/nursery)
// check per store (gbemu's tiny `parentObj.reg = ...` handlers showed POST,PRE,store,
// POST for one store). The inline barrier stays only as a SAFETY NET for the rare
// store with no following barrier node. Reverts via GECKO_WJ_NODEDUPPOSTBAR.
static bool WJHasPostBarrierNode(jit::MDefinition* obj) {
  for (jit::MUseIterator u = obj->usesBegin(); u != obj->usesEnd(); u++) {
    jit::MNode* c = u->consumer();
    if (!c->isDefinition()) continue;
    jit::MDefinition* d = c->toDefinition();
    if (d->op() == jit::MDefinition::Opcode::PostWriteBarrier &&
        d->numOperands() >= 1 && d->getOperand(0) == obj)
      return true;
  }
  return false;
}
static bool EmitForcePostBarrier(Encoder& e, WJBackend& be, jit::MDefinition* objDef,
                                 jit::MDefinition* valDef) {
  // Inline the isTenured() gate (Ion's branchPtrInNurseryChunk) so we STAY IN
  // WASM for the common case: a NURSERY container needs no store-buffer entry, so
  // skip the C++ WJH_POSTBARRIER helper entirely for it. chunk(obj)->storeBuffer
  // == 0  <=>  obj is tenured  <=>  barrier needed. This is exactly equivalent to
  // the helper's own `if (obj->isTenured())` guard, just moved inline -- raytrace
  // makes ~13.7M of these calls/run, ~all on nursery Vectors/Colors, so the helper
  // boundary crossing (not the GC work) was ~65% of its C++-helper traffic.
  // When valDef is given, ALSO require the value to be a nursery GC cell (the full
  // Ion condition objTenured && valNursery): a tenured/non-GC value needs no barrier.
  // This eliminates the helper call for tenured->tenured stores -- and ALL stores
  // under NO_NURSERY (every value tenured -> never buffer), which was 3.7M pointless
  // POSTBAR calls/run in richards under NO_NURSERY.
  // No nursery -> no tenured->nursery edges ever exist -> post-write barriers are
  // entirely unnecessary. Skip them (sound). Speeds up NO_NURSERY runs (richards
  // paid ~1.9M pointless WJH_POSTBARRIER calls). Compile-time: GECKO_NO_NURSERY is
  // fixed at startup.
  static int noNursery = WJGETENV("GECKO_NO_NURSERY") ? 1 : 0;
  if (noNursery) return true;
  // PROBE (unsound): GECKO_WJ_NOBARRIER no-ops ALL post-write barriers to measure the
  // absolute ceiling of eliminating barrier cost (is POSTBAR the bottleneck or not?).
  static int noBarrier = WJGETENV("GECKO_WJ_NOBARRIER") ? 1 : 0;
  if (noBarrier) return true;
  // DEFAULT-ON (opt out GECKO_WJ_NOVALGUARD). The valNursery skip's pointer-type
  // chunk-load is now ADDRESS-BOUNDS-GUARDED (below): it only loads chunk(value)->sb
  // when the load address is in wasm memory, else conservatively emits the barrier.
  // That fixed the earley trap (1/10 "out of bounds" -> 0/16). +7% earley.
  static int vgEnv0 = WJGETENV("GECKO_WJ_NOVALGUARD") ? 0 : 1;
  if (vgEnv0 && valDef && !WJValueMightBeGCThing(valDef->type())) return true;
  static int noInlineBar = WJGETENV("GECKO_WJ_NOINLINEBARRIER") ? 1 : 0;
  uintptr_t objAddr = uintptr_t(static_cast<void*>(&gWJHelpObj));
  const uint32_t chunkMaskInv = uint32_t(~js::gc::ChunkMask);
  const uint32_t sbOff = uint32_t(js::gc::ChunkStoreBufferOffset);
  // valNursery skip: Ion's full post-barrier condition is `objTenured AND the stored
  // value is a NURSERY GC cell`. A tenured or non-GC value needs no barrier -> skip
  // the WJH_POSTBARRIER helper entirely. Correct + matches Ion. CRITICAL FIX vs the
  // prior (reverted) impl: for a Value-typed operand the isGCThing test must
  // SHORT-CIRCUIT the chunk load -- a non-GC Value's low-32 payload is garbage, and
  // loading chunk(garbage)->storeBuffer either reads junk or traps (out-of-bounds);
  // the old code loaded unconditionally and AND-masked after, which is why it broke
  // raytrace/deltablue/splay. Callers pre-filter with WJValueMightBeGCThing, so a
  // provided valDef is always a GC-possible type. GECKO_WJ_NOVALGUARD reverts to the
  // objTenured-only gate for A/B.
  // OPT-IN (GECKO_WJ_VALGUARD), default OFF = proven-correct objTenured-only gate.
  // The valNursery skip STILL crashes deltablue/splay even with the short-circuit
  // GC-thing fix (a missed-barrier soundness bug not yet located -- objTenured and
  // valNursery each look correct in isolation; the bug is elusive). AND it is NOT a
  // lever for the target benches: earley stores nursery cons cells (barrier genuinely
  // needed, not skippable); splay is minor-GC-PAUSE bound not barrier-CALL bound
  // (proven: halving POSTBAR didn't move splay). So this doesn't advance earley/splay/
  // gbemu 3x. Kept opt-in (not deleted) for a future proper debug + the NO_NURSERY/
  // tenured->tenured cases. The real earley lever is inlining the 8.87M WJH_CONSTRUCT.
  // Only refine with the valNursery skip if valDef is materializable -- else fall
  // back to the (sound) objTenured-only barrier rather than bailing the whole
  // function (GetOp(valDef) returning false = the 23 earley PostWriteBarrier
  // compile-bails that caused the CALL explosion + slowdown).
  // The valNursery skip is restricted to UNAMBIGUOUS POINTER types (Object/String/
  // Symbol/BigInt), NEVER Value. A Value-typed operand requires inspecting the tag
  // bits to decide if it's a GC cell, but a non-canonical NaN double can land its
  // high32 in the GC-tag range -> misclassified -> chunk(mantissa-garbage)->sb load
  // TRAPS (the crypto OOB). For Value-typed values we fall back to the sound
  // unconditional (objTenured-only) barrier -- and earley keeps its win regardless
  // (its gain is from the Object-typed skips; VG_OBJONLY measured ~635 == full).
  // GECKO_WJ_VG_VALUE re-enables the (unsafe) Value tag-inspection branch for A/B.
  static int vgValue = WJGETENV("GECKO_WJ_VG_VALUE") ? 1 : 0;
  // OPT-IN: valNursery skip restricted to reliably-pointer MIR types (never Value).
  // Still trap-prone (earley 1/10 OOB) -- see note above; opt-in pending the value-
  // pointer staleness fix.
  // The Object-branch chunk-load below is now bounds-guarded (trap-safe for any
  // value-pointer incl. stale Lambda closures / mistyped speculative Unboxes), so no
  // per-op exclusion is needed. Value-typed still excluded (its tag-inspection path
  // is separately bounds-guarded only if vgValue).
  bool valGuard =
      vgEnv0 && valDef != nullptr && WJCanGetOp(be, valDef) &&
      (vgValue || valDef->type() != jit::MIRType::Value);
  // VGDBG: log the payload the value-check is about to chunk-load (Object-branch),
  // so the last line before a trap identifies the OOB culprit.
  static int vgDbg = WJGETENV("GECKO_WJ_VGDBG") ? 1 : 0;
  if (valGuard && vgDbg && valDef->type() != jit::MIRType::Value) {
    fprintf(stderr, "[wj-vgdbg-c] valDef op=%s type=%s\n", WJOpName(valDef->op()),
            StringFromMIRType(valDef->type()));
    uintptr_t hp = uintptr_t(static_cast<void*>(&gWJHelpObj));
    if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(hp)) ||
        !GetOp(e, be, valDef) || !e.writeOp(Op::I32Store) || !e.writeVarU32(2) ||
        !e.writeVarU32(0) || !e.writeOp(Op::F64Const) ||
        !e.writeFixedF64(double(WJH_DBGPTR)) || !e.writeOp(Op::F64Const) ||
        !e.writeFixedF64(0.0) || !e.writeOp(Op::Call) || !e.writeVarU32(0) ||
        !e.writeOp(Op::Drop))
      return false;
  }
  if (!noInlineBar) {
    // tenured = (chunk(obj)->storeBuffer == 0)
    if (!GetOp(e, be, objDef) || !e.writeOp(Op::I32Const) ||
        !e.writeVarS32(int32_t(chunkMaskInv)) || !e.writeOp(Op::I32And) ||
        !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(sbOff) ||
        !e.writeOp(Op::I32Eqz))
      return false;
    if (valGuard) {
      if (valDef->type() == jit::MIRType::Value) {
        // isGCThing = (JSVAL_TAG_STRING <= tag <= JSVAL_TAG_OBJECT). BOTH bounds:
        // a >= STRING-only test misclassifies a non-canonical NaN double whose high32
        // lands above the tag range as a GC thing -> chunk(mantissa-garbage)->sb load
        // traps (the deterministic deltablue "memory access out of bounds"). Real GC
        // tags occupy exactly [STRING, OBJECT]; canonical doubles have high32 well
        // below STRING, so both bounds correctly exclude them.
        if (!GetOp(e, be, valDef) || !e.writeOp(Op::I64Const) || !e.writeVarS64(32) ||
            !e.writeOp(Op::I64ShrU) || !e.writeOp(Op::I32WrapI64) ||
            !e.writeOp(Op::I32Const) ||
            !e.writeVarS32(int32_t(uint32_t(JSVAL_TAG_STRING))) ||
            !e.writeOp(Op::I32GeU))
          return false;  // tag >= STRING
        if (!GetOp(e, be, valDef) || !e.writeOp(Op::I64Const) || !e.writeVarS64(32) ||
            !e.writeOp(Op::I64ShrU) || !e.writeOp(Op::I32WrapI64) ||
            !e.writeOp(Op::I32Const) ||
            !e.writeVarS32(int32_t(uint32_t(JSVAL_TAG_OBJECT))) ||
            !e.writeOp(Op::I32LeU) || !e.writeOp(Op::I32And))
          return false;  // && tag <= OBJECT  => isGCThing
        if (!e.writeOp(Op::If) || !e.writeFixedU8(uint8_t(TypeCode::I32)))
          return false;
        // GC thing: valNursery = chunk(payload)->sb != 0
        if (!GetOp(e, be, valDef) || !e.writeOp(Op::I32WrapI64) ||
            !e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(chunkMaskInv)) ||
            !e.writeOp(Op::I32And) || !e.writeOp(Op::I32Load) || !e.writeVarU32(2) ||
            !e.writeVarU32(sbOff) || !e.writeOp(Op::I32Const) || !e.writeVarS32(0) ||
            !e.writeOp(Op::I32Ne))
          return false;
        if (!e.writeOp(Op::Else)) return false;
        if (!e.writeOp(Op::I32Const) || !e.writeVarS32(0)) return false;  // not GC -> 0
        if (!e.writeOp(Op::End)) return false;
      } else {
        // Object/String/Symbol/BigInt: payload = the pointer. TRAP-SAFE bounds guard:
        // a stale/mistyped value pointer (Lambda closure, speculative Unbox/
        // LoadElementAndUnbox) can be a non-current or non-object bit pattern whose
        // chunk(payload) lands OUTSIDE wasm memory -> the chunk-load TRAPS (earley
        // 1/10 "out of bounds"). So only load when payload is in-range; otherwise
        // assume nursery and emit the barrier (SOUND -- over-barrier never corrupts;
        // an in-range stale nursery ptr still reads its old nursery chunk -> correct).
        // Guard the ACTUAL load address (chunk(payload)+sbOff), not just payload:
        // a garbage payload in the top ~1 chunk would pass `payload<mem` yet make
        // chunkBase+sbOff exceed memory -> trap. addr = (payload & ~ChunkMask)+sbOff.
        // valNursery = (addr+4 <= memBytes) ? (load[addr] != 0) : 1
        auto emitAddr = [&]() -> bool {
          return GetOp(e, be, valDef) && e.writeOp(Op::I32Const) &&
                 e.writeVarS32(int32_t(chunkMaskInv)) && e.writeOp(Op::I32And) &&
                 e.writeOp(Op::I32Const) && e.writeVarS32(int32_t(sbOff)) &&
                 e.writeOp(Op::I32Add);
        };
        // (addr + 4) <= (memory.size << 16)
        if (!emitAddr() || !e.writeOp(Op::I32Const) || !e.writeVarS32(4) ||
            !e.writeOp(Op::I32Add) || !e.writeOp(Op::MemorySize) ||
            !e.writeVarU32(0) || !e.writeOp(Op::I32Const) || !e.writeVarS32(16) ||
            !e.writeOp(Op::I32Shl) || !e.writeOp(Op::I32LeU))
          return false;
        if (!e.writeOp(Op::If) || !e.writeFixedU8(uint8_t(TypeCode::I32)))
          return false;
        if (!emitAddr() || !e.writeOp(Op::I32Load) || !e.writeVarU32(2) ||
            !e.writeVarU32(0) || !e.writeOp(Op::I32Const) || !e.writeVarS32(0) ||
            !e.writeOp(Op::I32Ne))
          return false;  // load[addr] != 0  (chunk storeBuffer)
        if (!e.writeOp(Op::Else) || !e.writeOp(Op::I32Const) ||
            !e.writeVarS32(1) || !e.writeOp(Op::End))
          return false;  // out-of-range -> assume nursery (emit barrier)
      }
      if (!e.writeOp(Op::I32And)) return false;  // objTenured && valNursery
    }
    // Ion's one-element whole-cell cache (CodeGenerator.cpp ~5760): skip the helper
    // when obj is ALREADY the last buffered whole cell. WJH_POSTBARRIER -> PostWriteBarrier
    // -> WholeCellBuffer::put sets last_=obj on buffering, so a run of stores to the
    // same cell (splay tree-rotation re-stores the same nodes; earley/raytrace re-store
    // the same parent) only pays ONE helper call. Sound: last_ is reset at every minor
    // GC (== 0 -> obj != 0 -> barrier re-emitted); tenured cells never move so the ptr
    // compare is stable. GECKO_WJ_NOLASTCELL disables.
    static int noLastCell = WJGETENV("GECKO_WJ_NOLASTCELL") ? 1 : 0;
    if (!noLastCell && gWJWholeCellLastAddr) {
      if (!GetOp(e, be, objDef) || !e.writeOp(Op::I32Const) ||
          !e.writeVarS32(int32_t(gWJWholeCellLastAddr)) || !e.writeOp(Op::I32Load) ||
          !e.writeVarU32(2) || !e.writeVarU32(0) || !e.writeOp(Op::I32Ne) ||
          !e.writeOp(Op::I32And))
        return false;  // cond && (obj != lastBufferedWholeCell)
    }
    if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
  }
  if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(objAddr)) ||
      !GetOp(e, be, objDef) || !e.writeOp(Op::I32Store) || !e.writeVarU32(2) ||
      !e.writeVarU32(0))
    return false;
  if (!(e.writeOp(Op::F64Const) && e.writeFixedF64(double(WJH_POSTBARRIER)) &&
        e.writeOp(Op::F64Const) && e.writeFixedF64(0.0) && e.writeOp(Op::Call) &&
        e.writeVarU32(0) && e.writeOp(Op::Drop)))
    return false;
  if (!noInlineBar && !e.writeOp(Op::End)) return false;  // close the tenured `if`
  return true;
}

// True if a value of this MIRType could be a GC pointer (object/string/symbol/
// bigint, or an untyped Value box which may contain one) -- i.e. a store of it
// may need a post-write barrier. Numbers/booleans/null/undefined never do.
static bool WJValueMightBeGCThing(jit::MIRType t) {
  return t == jit::MIRType::Object || t == jit::MIRType::Value ||
         t == jit::MIRType::String || t == jit::MIRType::Symbol ||
         t == jit::MIRType::BigInt;
}

// True if the node has a side effect we must emit even though it produces no
// wasm value (or its value is unused). Returns false from EmitEffect => skip.
enum class EffectKind { Skip, Emitted, Fail };

static EffectKind EmitEffect(Encoder& e, WJBackend& be, MInstruction* ins) {
  WJSetBailOp(ins);  // attribute an internal bail to this op (+ precise op line)
  switch (ins->op()) {
    case MDefinition::Opcode::CloseIterCache: {
      // for-of early-exit cleanup: call iter's return() via js::CloseIterOperation.
      // scratch[0]=iter(boxed), site=completionKind. (uBlock's process() for-of loops.)
      auto* c = ins->toCloseIterCache();
      if (!EmitStageScratch(e, be, c->iter(), 0)) return EffectKind::Fail;
      if (!EmitHelperCallResult(e, be, ins, WJH_CLOSEITER,
                                uint32_t(c->completionKind())))
        return EffectKind::Fail;
      return e.writeOp(Op::Drop) ? EffectKind::Emitted : EffectKind::Fail;
    }
    case MDefinition::Opcode::Bail:
    case MDefinition::Opcode::Unreachable:
    case MDefinition::Opcode::ThrowWithStack:  // `throw` variant that attaches an
                                               // error stack; same cold-path deopt.
    case MDefinition::Opcode::Throw: {
      // Throw / ThrowWithStack: a cold error path (splay `throw new Error(...)`,
      // raytrace asserts). Deopt to PBL, which re-executes from the resume point and
      // performs the real throw with correct exception semantics. Lets the hot path stay JIT'd instead
      // of bailing the whole function to PBL. This is a TOP-LEVEL deopt (not inside a
      // guard's `if`), so clear inDispatchBody so EmitDeopt uses the inline-return
      // path -- the OOL `br $D` depth is calibrated for a guard's enclosing `if` and
      // would branch one scope too far here (invalid wasm: "expected 2 for branch").
      bool savedDispatch = be.inDispatchBody;
      be.inDispatchBody = false;
      bool ok = EmitDeopt(e, be);
      be.inDispatchBody = savedDispatch;
      return ok ? EffectKind::Emitted : EffectKind::Fail;
    }
    case MDefinition::Opcode::Start:
    case MDefinition::Opcode::CheckOverRecursed:
    case MDefinition::Opcode::InterruptCheck:
      return EffectKind::Skip;  // prologue / ignored (host stack guards depth)
    case MDefinition::Opcode::AddAndStoreSlot: {
      // Add an own property: transition obj->shape to the new shape (which has the
      // added slot) and store the value at slotOffset. Mirrors Ion's
      // visitAddAndStoreSlot. The shape pre-write barrier Ion emits is safely
      // ELIDED here: this is a property-ADDITION transition, so the new shape has
      // the OLD shape as its shape-tree parent -> the old shape stays reachable
      // and gets marked via the new shape (the SATB invariant holds without the
      // explicit barrier). The value store needs no pre-barrier (new init); any
      // post-write barrier is a separate MPostWriteBarrier node (handled).
      // (raytrace constructors / color+vector field init bail here otherwise.)
      auto* a = ins->toAddAndStoreSlot();
      if (a->preserveWrapper()) return EffectKind::Fail;  // proxy: rare, bail
      uintptr_t shapeSlot = WJInternShape(uintptr_t(a->shape()));
      if (!shapeSlot) return EffectKind::Fail;
      uint32_t shapeOff = uint32_t(offsetof(JS::shadow::Object, shape));
      uint32_t slotOffset = a->slotOffset();
      // obj->shape = relocated new shape
      if (!GetOp(e, be, a->object()) || !e.writeOp(Op::I32Const) ||
          !e.writeVarS32(int32_t(shapeSlot)) || !e.writeOp(Op::I32Load) ||
          !e.writeVarU32(2) || !e.writeVarU32(0) || !e.writeOp(Op::I32Store) ||
          !e.writeVarU32(2) || !e.writeVarU32(shapeOff))
        return EffectKind::Fail;
      // store value at the slot (fixed: obj+offset; dynamic: slots+offset).
      if (a->kind() == MAddAndStoreSlot::Kind::FixedSlot) {
        if (!GetOp(e, be, a->object()) || !EmitCanonStoreValue(e, be, a->value()) ||
            !e.writeOp(Op::I64Store) || !e.writeVarU32(3) ||
            !e.writeVarU32(slotOffset))
          return EffectKind::Fail;
      } else {
        if (!GetOp(e, be, a->object()) || !e.writeOp(Op::I32Load) ||
            !e.writeVarU32(2) ||
            !e.writeVarU32(uint32_t(js::NativeObject::offsetOfSlots())) ||
            !EmitCanonStoreValue(e, be, a->value()) || !e.writeOp(Op::I64Store) ||
            !e.writeVarU32(3) || !e.writeVarU32(slotOffset))
          return EffectKind::Fail;
      }
      // POST-WRITE BARRIER: Warp folds the barrier into the store op for
      // AddAndStoreSlot rather than emitting a separate MPostWriteBarrier node, so
      // we MUST emit it here. Without it, storing a nursery value into a promoted
      // (tenured) object leaves the slot out of the store buffer -> a later minor
      // GC doesn't update it -> stale pointer (raytrace's intermittent "rendered
      // incorrectly"; huge nursery / NO_NURSERY masks it). Only for GC-typed values
      // (numbers/booleans need no barrier -> keeps Vector.x/y/z field init fast).
      if (WJValueMightBeGCThing(a->value()->type()) &&
          !EmitForcePostBarrier(e, be, a->object(), a->value()))
        return EffectKind::Fail;
      return EffectKind::Emitted;
    }
    case MDefinition::Opcode::AllocateAndStoreSlot: {
      // Like AddAndStoreSlot but the added property overflows into a NEW dynamic
      // slot, so first grow the slots buffer (WJH_GROWSLOTS -> growSlotsPure), then
      // transition the shape + store the value at the (dynamic) slotOffset. Mirrors
      // Ion visitAllocateAndStoreSlot. (raytrace intersect's `new IntersectionInfo`
      // bailed here.) growSlotsPure mallocs off-heap so the object doesn't move; the
      // post-call GetOp rematerializes it and re-reads the fresh slots pointer.
      auto* a = ins->toAllocateAndStoreSlot();
      if (a->preserveWrapper()) return EffectKind::Fail;  // proxy: rare, bail
      uintptr_t shapeSlot = WJInternShape(uintptr_t(a->shape()));
      if (!shapeSlot) return EffectKind::Fail;
      uint32_t slotOffset = a->slotOffset();
      // gWJNewAux = newCapacity; scratch[0] = object; call WJH_GROWSLOTS, drop result
      if (!EmitStageScratch(e, be, a->object(), 0)) return EffectKind::Fail;
      if (!e.writeOp(Op::I32Const) ||
          !e.writeVarS32(int32_t(uintptr_t(static_cast<void*>(&gWJNewAux)))) ||
          !e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(a->numNewSlots())) ||
          !e.writeOp(Op::I32Store) || !e.writeVarU32(2) || !e.writeVarU32(0))
        return EffectKind::Fail;
      if (!EmitHelperCallResult(e, be, ins, WJH_GROWSLOTS, 0)) return EffectKind::Fail;
      if (!e.writeOp(Op::Drop)) return EffectKind::Fail;
      // obj->shape = relocated new shape
      uint32_t shapeOff = uint32_t(offsetof(JS::shadow::Object, shape));
      if (!GetOp(e, be, a->object()) || !e.writeOp(Op::I32Const) ||
          !e.writeVarS32(int32_t(shapeSlot)) || !e.writeOp(Op::I32Load) ||
          !e.writeVarU32(2) || !e.writeVarU32(0) || !e.writeOp(Op::I32Store) ||
          !e.writeVarU32(2) || !e.writeVarU32(shapeOff))
        return EffectKind::Fail;
      // store value at slots + slotOffset (dynamic slot; slots re-read post-grow)
      if (!GetOp(e, be, a->object()) || !e.writeOp(Op::I32Load) ||
          !e.writeVarU32(2) ||
          !e.writeVarU32(uint32_t(js::NativeObject::offsetOfSlots())) ||
          !EmitCanonStoreValue(e, be, a->value()) || !e.writeOp(Op::I64Store) ||
          !e.writeVarU32(3) || !e.writeVarU32(slotOffset))
        return EffectKind::Fail;
      // POST-WRITE BARRIER (see AddAndStoreSlot): emit for GC-typed values.
      if (WJValueMightBeGCThing(a->value()->type()) &&
          !EmitForcePostBarrier(e, be, a->object(), a->value()))
        return EffectKind::Fail;
      return EffectKind::Emitted;
    }
    case MDefinition::Opcode::StoreSlotByIteratorIndex: {
      // `o[Object.keys(o)[i]] = v` write via the iterator's current PropertyIndex.
      MStoreSlotByIteratorIndex* s = ins->toStoreSlotByIteratorIndex();
      return EmitIterSlotStore(e, be, s->object(), s->iterator(), nullptr,
                               s->value())
                 ? EffectKind::Emitted
                 : EffectKind::Fail;
    }
    case MDefinition::Opcode::StoreSlotByIteratorIndexIndexed: {
      MStoreSlotByIteratorIndexIndexed* s =
          ins->toStoreSlotByIteratorIndexIndexed();
      return EmitIterSlotStore(e, be, s->object(), s->iterator(), s->index(),
                               s->value())
                 ? EffectKind::Emitted
                 : EffectKind::Fail;
    }
    case MDefinition::Opcode::StoreFixedSlot: {
      MStoreFixedSlot* s = ins->toStoreFixedSlot();
      // Megamorphic recompile: this store is on a mega-convertible (passthrough)
      // GuardShape, so the receiver may be polymorphic -> a raw fixed-slot store
      // would corrupt. Route through the generic SETPROP helper (baked key, no
      // deopt, handles any shape). EFFECT -> leaves nothing on the stack.
      bool sMega = be.forceMega && WJIsShapeGuard(s->object()) &&
                   WJMegaConvertibleGuard(s->object(), /*allowStores=*/true, /*stormProven=*/be.forceMega);
      bool sHybrid = !sMega && WJShapeHybrid() && WJIsShapeGuard(s->object()) &&
                     WJMegaConvertibleGuard(s->object(), /*allowStores=*/false, /*stormProven=*/be.forceMega);
      if (sMega || sHybrid) {
        jit::MDefinition* gs = s->object();
        uint64_t keyBits = 0;
        if (WJDerivePropKey(WJGuardRepShape(gs), uint32_t(s->slot()), &keyBits) &&
            WJGuardObject(gs)->type() == MIRType::Object) {
          uint64_t keyVal = js::IdToValue(
              JS::PropertyKey::fromRawBits(uintptr_t(keyBits))).asRawBits();
          // Per-access mega: route through the inline multi-way store IC instead of
          // the generic WJH_SETPROP helper. Monomorphic receivers (richards' Task/
          // Packet field stores -- 1.9M/run) get a fast inline shape-guard + slot
          // store; polymorphic ones use up to kWJPropWays cached shapes; only true
          // mega misses hit WJH_SETPROPIC. The IC self-guards on shape so it's sound
          // for the passthrough'd (polymorphic) guard. HUGE win (richards 152->351,
          // SETPROP 1.9M->0) BUT has a GC-nursery-triggered staleness crash (NO_NURSERY
          // DEFAULT-ON 2026-07-11 (GECKO_WJ_NOMEGASTOREIC reverts): the historical
          // "null function" GC-staleness crash is FIXED (by the intervening precise-
          // rooting/store-barrier work) -- re-validated with gczeal 7,1 (minor GC on
          // EVERY allocation) on both the existing-slot AND property-ADD paths under
          // FORCEMEGA+MEGASTOREIC: result == PBL exactly, no crash. Note the store IC
          // (incl. the ADD-IC post-write barrier) is ALREADY default-on for the other
          // store callers (SetPropertyCache / MMegamorphicStoreSlot / ADDIC), so this
          // flip only extends that same validated path to the forceMega StoreFixedSlot
          // GC-value case -- removing a 4x store cliff (richards-under-FORCEMEGA
          // 511 -> 2026) for shape-storm store-heavy fns (the real-site pattern).
          // Hybrid always uses the store IC (value is provably non-GC -> no barrier).
          static int megaStoreIC = WJGETENV("GECKO_WJ_NOMEGASTOREIC") ? 0 : 1;
          static int hybStoreHelper = WJGETENV("GECKO_WJ_HYBSTOREHELPER") ? 1 : 0;
          bool useIC = (sHybrid && !hybStoreHelper) || megaStoreIC;
          uint32_t site = useIC ? WJPropSite(be) : 0;
          if (site != 0) {
            return EmitPropStoreIC(e, be, ins, WJGuardObject(gs), nullptr,
                                   s->value(), /*strict=*/false, site, keyVal)
                       ? EffectKind::Emitted
                       : EffectKind::Fail;
          }
          // Site pool exhausted (or reverted): generic helper.
          if (!EmitStageScratch(e, be, WJGuardObject(gs), 0)) return EffectKind::Fail;
          if (!EmitStageConstBoxed(e, be, keyVal, 1)) return EffectKind::Fail;
          if (!EmitStageScratch(e, be, s->value(), 2)) return EffectKind::Fail;
          if (!EmitHelperCallResult(e, be, ins, WJH_SETPROP, 0))
            return EffectKind::Fail;
          return e.writeOp(Op::Drop) ? EffectKind::Emitted : EffectKind::Fail;
        }
        // Guard was removed (passthrough) but this store can't convert by-name
        // (key underivable / receiver not MIRType::Object). A raw fixed-slot store
        // at s->slot() would write the WRONG slot for a polymorphic receiver ->
        // heap corruption (raytrace "Scene rendered incorrectly"). BAIL the whole
        // function to PBL -- matches LoadFixedSlot's behaviour; never emit the raw
        // store under a removed guard.
        if (WJGETENV("GECKO_WJ_MEGASTOREBAIL"))
          fprintf(stderr, "[wj-megastorebail] StoreFixedSlot slot=%zu objType=%d\n",
                  s->slot(), int(WJGuardObject(gs)->type()));
        return EffectKind::Fail;
      }
      int32_t objLocal = be.local(s->object());
      if (objLocal < 0) return EffectKind::Fail;
      MDefinition* v = s->value();
      if (s->needsBarrier()) {
        // Was: route through wjhelp(WJH_SETSLOT) -> NativeObject::setSlot (a C++ hop
        // per store). Now INLINE (stay in wasm), mirroring StoreDynamicSlot: gated
        // incremental pre-write barrier on the OLD value + inline store + inline-
        // gated post-write barrier. EmitGuardedValuePreBarrier only fires when the
        // zone is mid-incremental-marking; EmitForcePostBarrier skips nursery
        // containers inline. raytrace did ~1.2M of these helper calls/run.
        // GECKO_WJ_NOINLINESETSLOT reverts to the helper for A/B.
        static int noInline = WJGETENV("GECKO_WJ_NOINLINESETSLOT") ? 1 : 0;
        if (!noInline) {
          uint32_t off = uint32_t(js::NativeObject::getFixedSlotOffset(s->slot()));
          if (WJGETENV("GECKO_WJ_SLOTDBG2"))
            fprintf(stderr, "[wb-slot] StoreFixedSlot(barrier) slot=%zu off=%u objOp=%s\n",
                    s->slot(), off, WJOpName(s->object()->op()));
          if (!EmitGuardedValuePreBarrier(e, be, [&]() {
                return GetOp(e, be, s->object()) && e.writeOp(Op::I64Load) &&
                       e.writeVarU32(3) && e.writeVarU32(off);
              }))
            return EffectKind::Fail;
          if (!GetOp(e, be, s->object())) return EffectKind::Fail;
          if (!EmitCanonStoreValue(e, be, v)) return EffectKind::Fail;
          if (!e.writeOp(Op::I64Store) || !e.writeVarU32(3) || !e.writeVarU32(off))
            return EffectKind::Fail;
          // Skip the inline post-barrier when an explicit MPostWriteBarrier node
          // already covers this object (the common case) -- avoids the double-emit.
          static int dedupPB = WJGETENV("GECKO_WJ_NODEDUPPOSTBAR") ? 0 : 1;
          bool nodeCoversPB = dedupPB && WJHasPostBarrierNode(s->object());
          if (WJValueMightBeGCThing(v->type()) && !nodeCoversPB &&
              !EmitForcePostBarrier(e, be, s->object(), v))
            return EffectKind::Fail;
          return EffectKind::Emitted;
        }
        uintptr_t objAddr = uintptr_t(static_cast<void*>(&gWJHelpObj));
        uintptr_t slotAddr = uintptr_t(static_cast<void*>(&gWJHelpSlot));
        uintptr_t valAddr = uintptr_t(static_cast<void*>(&gWJHelpVal));
        if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(objAddr)) ||
            !GetOp(e, be, s->object()) ||
            !e.writeOp(Op::I32Store) || !e.writeVarU32(2) || !e.writeVarU32(0))
          return EffectKind::Fail;
        if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(slotAddr)) ||
            !e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(s->slot())) ||
            !e.writeOp(Op::I32Store) || !e.writeVarU32(2) || !e.writeVarU32(0))
          return EffectKind::Fail;
        if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(valAddr)) ||
            !EmitSpillValue(e, be, v) ||
            !e.writeOp(Op::I64Store) || !e.writeVarU32(3) || !e.writeVarU32(0))
          return EffectKind::Fail;
        if (!e.writeOp(Op::F64Const) || !e.writeFixedF64(double(WJH_SETSLOT)) ||
            !e.writeOp(Op::F64Const) || !e.writeFixedF64(0.0) ||
            !e.writeOp(Op::Call) || !e.writeVarU32(0) || !e.writeOp(Op::Drop))
          return EffectKind::Fail;
        return EffectKind::Emitted;
      }
      // Barrier-free (primitive) store: inline.
      uint32_t off = uint32_t(js::NativeObject::getFixedSlotOffset(s->slot()));
      if (WJGETENV("GECKO_WJ_SLOTDBG2"))
        fprintf(stderr, "[wb-slot] StoreFixedSlot slot=%zu off=%u objOp=%s barrier=%d\n",
                s->slot(), off, WJOpName(s->object()->op()), s->needsBarrier());
      if (!GetOp(e, be, s->object())) return EffectKind::Fail;  // addr (current)
      if (!EmitCanonStoreValue(e, be, v)) return EffectKind::Fail;    // boxed value
      if (!e.writeOp(Op::I64Store) || !e.writeVarU32(3) || !e.writeVarU32(off))
        return EffectKind::Fail;
      static int forceBarrier = WJGETENV("GECKO_WJ_FORCEBARRIER") ? 1 : 0;
      if (forceBarrier && WJTypeMaybeGC(v->type()) &&
          !EmitForcePostBarrier(e, be, s->object()))
        return EffectKind::Fail;
      return EffectKind::Emitted;
    }
    case MDefinition::Opcode::StoreFixedSlotFromOffset: {
      // storeValue(object + offset) = value. Fixed slots are inline in the
      // object. needsBarrier() => pre-write barrier on the OLD value (mirrors
      // StoreDynamicSlot); a separate MPostWriteBarrier handles the nursery side.
      MStoreFixedSlotFromOffset* s = ins->toStoreFixedSlotFromOffset();
      MDefinition* v = s->value();
      if (s->needsBarrier()) {
        if (!EmitGuardedValuePreBarrier(e, be, [&]() {
              return GetOp(e, be, s->object()) && GetOp(e, be, s->offset()) &&
                     e.writeOp(Op::I32Add) && e.writeOp(Op::I64Load) &&
                     e.writeVarU32(3) && e.writeVarU32(0);
            }))
          return EffectKind::Fail;
      }
      if (!GetOp(e, be, s->object())) return EffectKind::Fail;
      if (!GetOp(e, be, s->offset())) return EffectKind::Fail;
      if (!e.writeOp(Op::I32Add)) return EffectKind::Fail;
      if (!EmitCanonStoreValue(e, be, v)) return EffectKind::Fail;
      if (!e.writeOp(Op::I64Store) || !e.writeVarU32(3) || !e.writeVarU32(0))
        return EffectKind::Fail;
      static int forceBarrier = WJGETENV("GECKO_WJ_FORCEBARRIER") ? 1 : 0;
      if (forceBarrier && WJTypeMaybeGC(v->type()) &&
          !EmitForcePostBarrier(e, be, s->object()))
        return EffectKind::Fail;
      return EffectKind::Emitted;
    }
    case MDefinition::Opcode::StoreDynamicSlotFromOffset: {
      // storeValue(slots + offset) = value, where slots is the out-of-line slots
      // buffer (vs StoreFixedSlotFromOffset's inline object base). Ion emits an
      // UNCONDITIONAL pre-write barrier on the old value (visitStoreDynamicSlot
      // FromOffset{V,T}); the nursery post-barrier is a separate MPostWriteBarrier.
      MStoreDynamicSlotFromOffset* s = ins->toStoreDynamicSlotFromOffset();
      MDefinition* v = s->value();
      if (!EmitGuardedValuePreBarrier(e, be, [&]() {
            return GetOp(e, be, s->slots()) && GetOp(e, be, s->offset()) &&
                   e.writeOp(Op::I32Add) && e.writeOp(Op::I64Load) &&
                   e.writeVarU32(3) && e.writeVarU32(0);
          }))
        return EffectKind::Fail;
      if (!GetOp(e, be, s->slots())) return EffectKind::Fail;
      if (!GetOp(e, be, s->offset())) return EffectKind::Fail;
      if (!e.writeOp(Op::I32Add)) return EffectKind::Fail;
      if (!EmitCanonStoreValue(e, be, v)) return EffectKind::Fail;
      if (!e.writeOp(Op::I64Store) || !e.writeVarU32(3) || !e.writeVarU32(0))
        return EffectKind::Fail;
      // Nursery post-barrier is a separate MPostWriteBarrier node (this op only
      // has the slots buffer, not the owning object, so no forced post-barrier).
      return EffectKind::Emitted;
    }
    case MDefinition::Opcode::StoreDynamicSlot: {
      // slots[slot] = value. The barrier-free case (primitive value, or a
      // separate MPostWriteBarrier node follows) stores inline; a value needing
      // a pre-write barrier falls through to bail (correct, rare).
      MStoreDynamicSlot* s = ins->toStoreDynamicSlot();
      // Megamorphic recompile: if this store's Slots came off a passthrough'd
      // (mega-convertible) GuardShape, the baked slot index is unsafe for a
      // polymorphic receiver -> route to the self-guarding store IC (gbemu's
      // dominant deopt source: 686K shape-guard deopts on out-of-line field
      // stores). Mirrors the StoreFixedSlot mega path. GECKO_WJ_MEGADYNSTORE.
      if ((be.forceMega || WJShapeHybrid()) && s->slots()->isSlots()) {
        jit::MDefinition* gs = s->slots()->toSlots()->object();
        if (WJIsShapeGuard(gs) &&
            WJMegaConvertibleGuard(gs, /*allowStores=*/be.forceMega, /*stormProven=*/be.forceMega)) {
          js::Shape* gshape = WJGuardRepShape(gs);
          uint32_t nfixed = gshape ? gshape->asShared().numFixedSlots() : 0;
          uint64_t keyBits = 0;
          if (gshape &&
              WJDerivePropKey(gshape, nfixed + s->slot(), &keyBits) &&
              WJGuardObject(gs)->type() == MIRType::Object) {
            uint64_t keyVal = js::IdToValue(
                JS::PropertyKey::fromRawBits(uintptr_t(keyBits))).asRawBits();
            static int hybStoreHelper = WJGETENV("GECKO_WJ_HYBSTOREHELPER") ? 1 : 0;
            uint32_t site = (hybStoreHelper && !be.forceMega) ? 0 : WJPropSite(be);
            if (site != 0) {
              return EmitPropStoreIC(e, be, ins, WJGuardObject(gs), nullptr,
                                     s->value(), /*strict=*/false, site, keyVal)
                         ? EffectKind::Emitted
                         : EffectKind::Fail;
            }
            if (!EmitStageScratch(e, be, WJGuardObject(gs), 0)) return EffectKind::Fail;
            if (!EmitStageConstBoxed(e, be, keyVal, 1)) return EffectKind::Fail;
            if (!EmitStageScratch(e, be, s->value(), 2)) return EffectKind::Fail;
            if (!EmitHelperCallResult(e, be, ins, WJH_SETPROP, 0))
              return EffectKind::Fail;
            return e.writeOp(Op::Drop) ? EffectKind::Emitted : EffectKind::Fail;
          }
          // Guard passthrough'd but can't convert by-name -> a raw baked store
          // would corrupt a polymorphic receiver. Bail the whole fn to PBL.
          return EffectKind::Fail;
        }
      }
      uint32_t off = uint32_t(s->slot() * sizeof(JS::Value));
      if (s->needsBarrier()) {
        if (!EmitGuardedValuePreBarrier(e, be, [&]() {
              return GetOp(e, be, s->slots()) && e.writeOp(Op::I64Load) &&
                     e.writeVarU32(3) && e.writeVarU32(off);
            }))
          return EffectKind::Fail;
      }
      if (!GetOp(e, be, s->slots())) return EffectKind::Fail;
      if (!EmitCanonStoreValue(e, be, s->value())) return EffectKind::Fail;
      if (!e.writeOp(Op::I64Store) || !e.writeVarU32(3) || !e.writeVarU32(off))
        return EffectKind::Fail;
      static int forceBarrier = WJGETENV("GECKO_WJ_FORCEBARRIER") ? 1 : 0;
      if (forceBarrier && WJTypeMaybeGC(s->value()->type()) &&
          s->slots()->isSlots() &&
          !EmitForcePostBarrier(e, be, s->slots()->toSlots()->object()))
        return EffectKind::Fail;
      return EffectKind::Emitted;
    }
    case MDefinition::Opcode::StoreUnboxedScalar: {
      // Typed-array element store: data[index] = value (sized). Mirrors Ion
      // visitStoreUnboxedScalar. Int storage takes an Int32 value (wasm truncates
      // the low bits, matching JS ToInt32-then-narrow); float storage takes a
      // Double. No barrier (scalar data, no GC pointers). gbemu's Uint8Array writes.
      auto* s = ins->toStoreUnboxedScalar();
      Scalar::Type st = s->writeType();
      int32_t il = be.local(s->index());
      if (il < 0) return EffectKind::Fail;
      MDefinition* valD = s->value();
      bool valDouble = valD->type() == MIRType::Double ||
                       valD->type() == MIRType::Float32;
      bool valInt = valD->type() == MIRType::Int32 || valD->type() == MIRType::Boolean;
      bool floatStore = st == Scalar::Float32 || st == Scalar::Float64;
      // Require the value already coerced to the storage class (Warp inserts the
      // conversion); a mismatch (e.g. Double into an int array) is rare -> bail.
      if (floatStore ? !valDouble : !valInt)
        return WJBAIL("StoreUnboxedScalar value/type mismatch\n") ? EffectKind::Fail
                                                                  : EffectKind::Fail;
      uint32_t esz = uint32_t(Scalar::byteSize(st));
      if (!GetOp(e, be, s->elements()) || !GetLocal(e, uint32_t(il)) ||
          !e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(esz)) ||
          !e.writeOp(Op::I32Mul) || !e.writeOp(Op::I32Add))
        return EffectKind::Fail;  // addr on stack
      if (!GetOp(e, be, valD)) return EffectKind::Fail;  // value
      switch (st) {
        case Scalar::Int8:
        case Scalar::Uint8:
          if (!e.writeOp(Op::I32Store8) || !e.writeVarU32(0) || !e.writeVarU32(0))
            return EffectKind::Fail;
          return EffectKind::Emitted;
        case Scalar::Int16:
        case Scalar::Uint16:
          if (!e.writeOp(Op::I32Store16) || !e.writeVarU32(1) || !e.writeVarU32(0))
            return EffectKind::Fail;
          return EffectKind::Emitted;
        case Scalar::Int32:
        case Scalar::Uint32:
          if (!e.writeOp(Op::I32Store) || !e.writeVarU32(2) || !e.writeVarU32(0))
            return EffectKind::Fail;
          return EffectKind::Emitted;
        case Scalar::Float32:
          // value is f64 (Double or Float32, both f64 locals) -> demote to f32.
          if (!e.writeOp(Op::F32DemoteF64) || !e.writeOp(Op::F32Store) ||
              !e.writeVarU32(2) || !e.writeVarU32(0))
            return EffectKind::Fail;
          return EffectKind::Emitted;
        case Scalar::Float64:
          if (!e.writeOp(Op::F64Store) || !e.writeVarU32(3) || !e.writeVarU32(0))
            return EffectKind::Fail;
          return EffectKind::Emitted;
        default:
          return EffectKind::Fail;  // Uint8Clamped (needs saturation) / BigInt -> PBL
      }
    }
    case MDefinition::Opcode::StoreElement: {
      // Dense store elements[index] = value. Object values (post-write barrier)
      // still bail. needsHoleCheck is handled inline by guarding the store is
      // in-bounds + not over a hole, deopting to PBL otherwise (grow/sparse) --
      // for pre-sized dense arrays (crypto bignums) the guards always pass, so
      // the hot store stays in JIT instead of bailing the whole function to PBL.
      MStoreElement* s = ins->toStoreElement();
      int32_t idx = be.local(s->index());
      if (idx < 0) return EffectKind::Fail;
      if (s->needsHoleCheck()) {
        // deopt unless (uint32)index < (uint32)initializedLength
        if (!GetOp(e, be, s->elements()) || !e.writeOp(Op::I32Const) ||
            !e.writeVarS32(js::ObjectElements::offsetOfInitializedLength()) ||
            !e.writeOp(Op::I32Add) || !e.writeOp(Op::I32Load) ||
            !e.writeVarU32(2) || !e.writeVarU32(0))
          return EffectKind::Fail;
        if (!GetLocal(e, uint32_t(idx)) || !e.writeOp(Op::I32LeU))
          return EffectKind::Fail;  // initLen <= index  -> out of bounds
        if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return EffectKind::Fail;
        if (!EmitDeopt(e, be)) return EffectKind::Fail;
        if (!e.writeOp(Op::End)) return EffectKind::Fail;
        // (No separate hole-value load: a dense in-bounds store just fills the
        // slot, matching Ion -- the bounds check above IS the needsHoleCheck.)
      }
      if (s->needsBarrier()) {
        // pre-write barrier on the OLD element value (incremental GC marking).
        if (!EmitGuardedValuePreBarrier(e, be, [&]() {
              return GetOp(e, be, s->elements()) && GetLocal(e, uint32_t(idx)) &&
                     e.writeOp(Op::I32Const) && e.writeVarS32(8) &&
                     e.writeOp(Op::I32Mul) && e.writeOp(Op::I32Add) &&
                     e.writeOp(Op::I64Load) && e.writeVarU32(3) && e.writeVarU32(0);
            }))
          return EffectKind::Fail;
      }
      if (!GetOp(e, be, s->elements()) || !GetLocal(e, uint32_t(idx)) ||
          !e.writeOp(Op::I32Const) || !e.writeVarS32(8) ||
          !e.writeOp(Op::I32Mul) || !e.writeOp(Op::I32Add))
        return EffectKind::Fail;  // addr = elements + index*8
      // CANONDBL: store an Int32-typed number as a DOUBLE (0 -> 0.0). int32 0 and
      // double 0.0 are indistinguishable in JS, so this is always sound; it keeps
      // number arrays all-double so element loads can be infallible f64 (navier).
      static int canonDbl = WJGETENV("GECKO_WJ_CANONDBL") ? 1 : 0;
      if (canonDbl && s->value()->type() == MIRType::Int32) {
        int32_t vl = be.local(s->value());
        if (vl >= 0) {
          if (!GetLocal(e, uint32_t(vl)) || !e.writeOp(Op::F64ConvertI32S) ||
              !e.writeOp(Op::I64ReinterpretF64))
            return EffectKind::Fail;
          if (!e.writeOp(Op::I64Store) || !e.writeVarU32(3) || !e.writeVarU32(0))
            return EffectKind::Fail;
          return EffectKind::Emitted;
        }
      }
      if (!EmitSpillValue(e, be, s->value())) return EffectKind::Fail;
      if (!e.writeOp(Op::I64Store) || !e.writeVarU32(3) || !e.writeVarU32(0))
        return EffectKind::Fail;
      // DEFAULT-ON post-write barrier for a maybe-GC value stored into a dense
      // element. Warp may ELIDE its separate MPostWriteBarrier node based on a
      // nursery-elements assumption that is WRONG at runtime: box2d's island arrays
      // (m_bodies/m_contacts/m_joints, AddBody/AddContact @box2d.js:236) are TENURED
      // but get nursery body/contact objects stored -> the missing barrier leaves the
      // array slot unrecorded -> a minor GC moves the object -> stale slot -> wrong
      // sim -> intermittent ERR=Box2D (100% under NURSERY_MB=1; NO_NURSERY masks it).
      // EmitForcePostBarrier is objTenured-gated (number/Double arrays skip it, so
      // navier etc. pay nothing) and store-buffer-idempotent. GECKO_WJ_NOSTOREELEMBARRIER
      // reverts (= the old default, which was unsound for tenured-array<-nursery-value).
      static int noStoreElemBarrier = WJGETENV("GECKO_WJ_NOSTOREELEMBARRIER") ? 1 : 0;
      if (!noStoreElemBarrier && WJTypeMaybeGC(s->value()->type()) &&
          s->elements()->isElements() &&
          !EmitForcePostBarrier(e, be, s->elements()->toElements()->object()))
        return EffectKind::Fail;
      return EffectKind::Emitted;
    }
    case MDefinition::Opcode::StoreElementHole: {
      // arr[i] = v where i may be at/just past the dense initialized length (append)
      // or over a hole. In-bounds dense store stays in JIT (gbemu's pre-sized
      // this.memory[addr]=v writes + the memoryRead/WriteJumpCompile cache tables);
      // the grow/append/hole case (index >= initializedLength) deopts to PBL which
      // grows the array correctly. A GC-pointer VALUE gets the proven-sound inline
      // post-write barrier after the store (objTenured-only gate, same path as
      // StoreFixedSlot's default HYBSTORE) -- uBlock's parser stores AST-node objects
      // into dense arrays, so bailing them all to PBL was the dominant StoreElementHole
      // cost. gbemu's element values are numbers (barrier skipped via WJTypeMaybeGC).
      MStoreElementHole* s = ins->toStoreElementHole();
      if (WJGETENV("GECKO_WJ_NOSEHGC") && WJTypeMaybeGC(s->value()->type()))
        return EffectKind::Fail;
      int32_t idx = be.local(s->index());
      if (idx < 0) return EffectKind::Fail;
      // bounds guard: deopt unless (uint32)index < (uint32)initializedLength
      if (!GetOp(e, be, s->elements()) || !e.writeOp(Op::I32Const) ||
          !e.writeVarS32(js::ObjectElements::offsetOfInitializedLength()) ||
          !e.writeOp(Op::I32Add) || !e.writeOp(Op::I32Load) ||
          !e.writeVarU32(2) || !e.writeVarU32(0))
        return EffectKind::Fail;
      if (!GetLocal(e, uint32_t(idx)) || !e.writeOp(Op::I32LeU))
        return EffectKind::Fail;  // initLen <= index -> grow/hole
      if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return EffectKind::Fail;
      // GROW/HOLE branch (index >= initializedLength). Inline the Ion-style dense
      // APPEND fast path: when index == initializedLength AND index < capacity the
      // store is a pure append into spare dense capacity, so do it INLINE (store value,
      // bump initializedLength, grow length if needed, inline post-write barrier). This
      // is GC-FREE (no allocation) -- unlike the SetObjectElement helper it adds NO new
      // GC safepoint (so no 3d-cube-style staleness), and unlike EmitDeopt it does NOT
      // replay the appending region (the deopt-resume bug that duplicated hash-map
      // entries / corrupted crypto-aes -- task #15). hash-map inserts keys 0..N in
      // order with index==key, so every store is exactly this append (capacity
      // 131072 > 90000) -> fully in JIT + correct. Holes (index > initializedLength)
      // and realloc (index >= capacity) fall through to deopt (PBL grows correctly);
      // 3d-cube's small arrays realloc -> deopt -> correct.
      //   GECKO_WJ_SEHAPPEND   -> inline dense-append fast path for index==initLen.
      //   GECKO_WJ_NOSEHHELPER -> revert to pure deopt for the grow branch.
      // DEFAULT (2026-06-28) = SetObjectElement helper for the grow branch. The plain
      // deopt is UNSOUND here: the StoreElementHole's resume point is a resume-AFTER
      // (or, when no nearer rp exists, the enclosing LOOP HEAD), so resuming re-runs
      // preceding side-effecting ops in the same statement/iteration -- e.g. hash-map
      // _rehash replays `entry._next = newData[index]` before re-reading
      // `next = entry._next`, corrupting the chain -> dup/lost entries; crypto-aes
      // (jetstream) wrong result; 3d-cube intermittent (task #15). The helper does the
      // grow in C++ with no deopt and no replay, so it is correct everywhere. It adds
      // a GC safepoint, which used to destabilize 3d-cube -- but the precise-rooting
      // default (2026-06-28) fixed that latent rooting bug, so the helper is now sound.
      // 2026-06-29: SEHAPPEND (inline GC-FREE dense append for index==initLen, helper
      // fallback for rehash/realloc) is now DEFAULT-ON. The old "hangs crypto-octane"
      // blocker was the SAME pre-precise-rooting GC-staleness the helper hit; precise
      // rooting fixed it. VALIDATED: crypto-aes 211->120ms (~1.75x), octane neutral +
      // no-ERR (all 10), crypto-octane no-hang x6, hash-map/3d-cube/crypto-aes OK under
      // tiny-nursery + GCZeal=7. The inline append has NO GC safepoint, so it is strictly
      // safer than the helper grow it replaces. Revert: GECKO_WJ_NOSEHAPPEND.
      static int sehAppend = WJGETENV("GECKO_WJ_NOSEHAPPEND") ? 0 : 1;
      static int sehHelper = WJGETENV("GECKO_WJ_NOSEHHELPER") ? 0 : 1;
      bool didAppend = false;
      if (sehAppend) {
        // cond = (index == initializedLength) & (capacity > index)
        if (!GetOp(e, be, s->elements()) || !e.writeOp(Op::I32Const) ||
            !e.writeVarS32(js::ObjectElements::offsetOfInitializedLength()) ||
            !e.writeOp(Op::I32Add) || !e.writeOp(Op::I32Load) ||
            !e.writeVarU32(2) || !e.writeVarU32(0))
          return EffectKind::Fail;
        if (!GetLocal(e, uint32_t(idx)) || !e.writeOp(Op::I32Eq))
          return EffectKind::Fail;
        if (!GetOp(e, be, s->elements()) || !e.writeOp(Op::I32Const) ||
            !e.writeVarS32(js::ObjectElements::offsetOfCapacity()) ||
            !e.writeOp(Op::I32Add) || !e.writeOp(Op::I32Load) ||
            !e.writeVarU32(2) || !e.writeVarU32(0))
          return EffectKind::Fail;
        if (!GetLocal(e, uint32_t(idx)) || !e.writeOp(Op::I32GtU))
          return EffectKind::Fail;
        if (!e.writeOp(Op::I32And)) return EffectKind::Fail;
        if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return EffectKind::Fail;
        // elements[index] = boxed value
        if (!GetOp(e, be, s->elements()) || !GetLocal(e, uint32_t(idx)) ||
            !e.writeOp(Op::I32Const) || !e.writeVarS32(8) ||
            !e.writeOp(Op::I32Mul) || !e.writeOp(Op::I32Add))
          return EffectKind::Fail;
        if (!EmitSpillValue(e, be, s->value())) return EffectKind::Fail;
        if (!e.writeOp(Op::I64Store) || !e.writeVarU32(3) || !e.writeVarU32(0))
          return EffectKind::Fail;
        // initializedLength = index + 1
        if (!GetOp(e, be, s->elements()) || !e.writeOp(Op::I32Const) ||
            !e.writeVarS32(js::ObjectElements::offsetOfInitializedLength()) ||
            !e.writeOp(Op::I32Add) || !GetLocal(e, uint32_t(idx)) ||
            !e.writeOp(Op::I32Const) || !e.writeVarS32(1) || !e.writeOp(Op::I32Add) ||
            !e.writeOp(Op::I32Store) || !e.writeVarU32(2) || !e.writeVarU32(0))
          return EffectKind::Fail;
        // if (length <= index) length = index + 1
        if (!GetOp(e, be, s->elements()) || !e.writeOp(Op::I32Const) ||
            !e.writeVarS32(js::ObjectElements::offsetOfLength()) ||
            !e.writeOp(Op::I32Add) || !e.writeOp(Op::I32Load) ||
            !e.writeVarU32(2) || !e.writeVarU32(0))
          return EffectKind::Fail;
        if (!GetLocal(e, uint32_t(idx)) || !e.writeOp(Op::I32LeU))
          return EffectKind::Fail;
        if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return EffectKind::Fail;
        if (!GetOp(e, be, s->elements()) || !e.writeOp(Op::I32Const) ||
            !e.writeVarS32(js::ObjectElements::offsetOfLength()) ||
            !e.writeOp(Op::I32Add) || !GetLocal(e, uint32_t(idx)) ||
            !e.writeOp(Op::I32Const) || !e.writeVarS32(1) || !e.writeOp(Op::I32Add) ||
            !e.writeOp(Op::I32Store) || !e.writeVarU32(2) || !e.writeVarU32(0))
          return EffectKind::Fail;
        if (!e.writeOp(Op::End)) return EffectKind::Fail;
        // GC-value append: inline post-write barrier (no GC safepoint).
        if (WJTypeMaybeGC(s->value()->type()) && !WJHasPostBarrierNode(s->object()) &&
            !EmitForcePostBarrier(e, be, s->object()))
          return EffectKind::Fail;
        if (!e.writeOp(Op::Else)) return EffectKind::Fail;
        didAppend = true;
      }
      // NON-APPEND grow (hole-fill or realloc): deopt (PBL), or helper if opted in.
      if (!sehHelper) {
        if (!EmitDeopt(e, be)) return EffectKind::Fail;
      } else {
        if (!EmitStageScratch(e, be, s->object(), 0)) return EffectKind::Fail;
        if (!EmitStageScratch(e, be, s->index(), 1)) return EffectKind::Fail;
        if (!EmitStageScratch(e, be, s->value(), 2)) return EffectKind::Fail;
        if (!EmitHelperCallResult(e, be, ins, WJH_SETPROP, 0))
          return EffectKind::Fail;
        if (!e.writeOp(Op::Drop)) return EffectKind::Fail;
      }
      if (didAppend && !e.writeOp(Op::End)) return EffectKind::Fail;  // close append-If
      if (!e.writeOp(Op::Else)) return EffectKind::Fail;
      // IN-BOUNDS branch: store inline. addr = elements + index*8; boxed value.
      if (!GetOp(e, be, s->elements()) || !GetLocal(e, uint32_t(idx)) ||
          !e.writeOp(Op::I32Const) || !e.writeVarS32(8) ||
          !e.writeOp(Op::I32Mul) || !e.writeOp(Op::I32Add))
        return EffectKind::Fail;
      if (!EmitSpillValue(e, be, s->value())) return EffectKind::Fail;
      if (!e.writeOp(Op::I64Store) || !e.writeVarU32(3) || !e.writeVarU32(0))
        return EffectKind::Fail;
      // GC-pointer value: post-write barrier on the array object (tenured-array <-
      // nursery-node edge). objTenured-only gate, proven sound. Skip if Warp already
      // emitted a PostWriteBarrier node for this object (dedup).
      if (WJTypeMaybeGC(s->value()->type()) && !WJHasPostBarrierNode(s->object()) &&
          !EmitForcePostBarrier(e, be, s->object()))
        return EffectKind::Fail;
      if (!e.writeOp(Op::End)) return EffectKind::Fail;
      return EffectKind::Emitted;
    }
    case MDefinition::Opcode::GuardGlobalGeneration: {
      // DEFAULT-ON (2026-07-02): bailing instead cost octane 10-15x (richards
      // 2326->137 -- global-heavy fns went wholesale to PBL). The earlier
      // "silent exit" seen with this inline was the canonical-callee resume bug
      // (see runtime-callee fix), not this guard. GECKO_WJ_NOGGG bails.
      if (WJGETENV("GECKO_WJ_NOGGG")) return EffectKind::Fail;
      // Deopt unless *generationAddr == expected (the realm's global was not
      // replaced/invalidated). Previously fell into the silent Skip default --
      // an UNSOUND drop like GuardElementsArePacked below.
      MGuardGlobalGeneration* g = ins->toGuardGlobalGeneration();
      if (!e.writeOp(Op::I32Const) ||
          !e.writeVarS32(int32_t(uintptr_t(g->generationAddr()))) ||
          !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0) ||
          !e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(g->expected())) ||
          !e.writeOp(Op::I32Ne))
        return EffectKind::Fail;
      if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return EffectKind::Fail;
      if (!EmitDeopt(e, be)) return EffectKind::Fail;
      if (!e.writeOp(Op::End)) return EffectKind::Fail;
      return EffectKind::Emitted;
    }
    case MDefinition::Opcode::GuardElementNotHole: {
      if (WJGETENV("GECKO_WJ_NOGENH")) return EffectKind::Fail;
      // Deopt if elements[index] is the hole magic (guards a following
      // hole-unaware dense store/read).
      MDefinition* elemsD = ins->getOperand(0);
      int32_t il = be.local(ins->getOperand(1));
      if (il < 0) return EffectKind::Fail;
      if (!GetOp(e, be, elemsD) || !GetLocal(e, uint32_t(il)) ||
          !e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(sizeof(JS::Value))) ||
          !e.writeOp(Op::I32Mul) || !e.writeOp(Op::I32Add) ||
          !e.writeOp(Op::I64Load) || !e.writeVarU32(3) || !e.writeVarU32(0) ||
          !e.writeOp(Op::I64Const) ||
          !e.writeVarS64(int64_t(JS::MagicValue(JS_ELEMENTS_HOLE).asRawBits())) ||
          !e.writeOp(Op::I64Eq))
        return EffectKind::Fail;
      if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return EffectKind::Fail;
      if (!EmitDeopt(e, be)) return EffectKind::Fail;
      if (!e.writeOp(Op::End)) return EffectKind::Fail;
      return EffectKind::Emitted;
    }
    case MDefinition::Opcode::BoundsCheckLower: {
      if (WJGETENV("GECKO_WJ_NOBCL")) return EffectKind::Fail;
      // Deopt if index < minimum (Ion visitBoundsCheckLower). Infallible form
      // needs nothing.
      MBoundsCheckLower* b = ins->toBoundsCheckLower();
      if (!b->fallible()) return EffectKind::Skip;
      int32_t il = be.local(b->index());
      if (il < 0) return EffectKind::Fail;
      if (!GetLocal(e, uint32_t(il)) || !e.writeOp(Op::I32Const) ||
          !e.writeVarS32(b->minimum()) || !e.writeOp(Op::I32LtS))
        return EffectKind::Fail;
      if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return EffectKind::Fail;
      if (!EmitDeopt(e, be)) return EffectKind::Fail;
      if (!e.writeOp(Op::End)) return EffectKind::Fail;
      return EffectKind::Emitted;
    }
    case MDefinition::Opcode::PostWriteElementBarrier: {
      if (WJGETENV("GECKO_WJ_NOPWEB")) return EffectKind::Fail;
      // Whole-cell store-buffer entry for the container object -- a sound
      // superset of Ion's element-precise barrier (mirrors PostWriteBarrier).
      if (!EmitForcePostBarrier(e, be, ins->getOperand(0), ins->getOperand(1)))
        return EffectKind::Fail;
      return EffectKind::Emitted;
    }
    case MDefinition::Opcode::GuardElementsArePacked: {
      if (WJGETENV("GECKO_WJ_NOGEAP")) return EffectKind::Fail;
      // Deopt unless the ELEMENTS header has no holes (NON_PACKED clear). Ion
      // pairs this with a hole-check-free LoadElement (self-hosted
      // ArrayIteratorNext `a[index]`); silently dropping it (the old Skip
      // default for None-typed guards) let a raw JS_ELEMENTS_HOLE magic flow
      // out of a holey array into user values -> the webtooling Set crash.
      MDefinition* elemsD = ins->getOperand(0);
      if (!GetOp(e, be, elemsD) || !e.writeOp(Op::I32Const) ||
          !e.writeVarS32(js::ObjectElements::offsetOfFlags()) ||
          !e.writeOp(Op::I32Add) || !e.writeOp(Op::I32Load) || !e.writeVarU32(2) ||
          !e.writeVarU32(0))
        return EffectKind::Fail;
      if (!e.writeOp(Op::I32Const) ||
          !e.writeVarS32(int32_t(js::ObjectElements::NON_PACKED)) ||
          !e.writeOp(Op::I32And))
        return EffectKind::Fail;
      if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return EffectKind::Fail;
      if (!EmitDeopt(e, be)) return EffectKind::Fail;
      if (!e.writeOp(Op::End)) return EffectKind::Fail;
      return EffectKind::Emitted;
    }
    case MDefinition::Opcode::StoreHoleValueElement: {
      // Store the hole magic at elements[index] AND mark the elements header
      // NON_PACKED (Ion visitStoreHoleValueElement). This op is NOT
      // isEffectful() (an infallible alias-set store), so it used to fall into
      // the Skip default: the array literal `[,-0]` then built with initLen
      // covering a slot that was never hole-marked -- the flag stayed PACKED,
      // and C++ packed fast paths (Set-from-array) read the raw hole magic ->
      // MOZ_RELEASE_ASSERT(whyMagic()==why) crash (webtooling lodash createSet).
      MDefinition* elemsD = ins->getOperand(0);
      MDefinition* idxD = ins->getOperand(1);
      int32_t il = be.local(idxD);
      if (il < 0) return EffectKind::Fail;
      // elements[index] = MagicValue(JS_ELEMENTS_HOLE)
      if (!GetOp(e, be, elemsD) || !GetLocal(e, uint32_t(il)) ||
          !e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(sizeof(JS::Value))) ||
          !e.writeOp(Op::I32Mul) || !e.writeOp(Op::I32Add) ||
          !e.writeOp(Op::I64Const) ||
          !e.writeVarS64(int64_t(JS::MagicValue(JS_ELEMENTS_HOLE).asRawBits())) ||
          !e.writeOp(Op::I64Store) || !e.writeVarU32(3) || !e.writeVarU32(0))
        return EffectKind::Fail;
      // elements->flags |= NON_PACKED. offsetOfFlags() is NEGATIVE (header is
      // before elements_) -> I32Add as a signed const, not a load offset.
      if (!GetOp(e, be, elemsD) || !e.writeOp(Op::I32Const) ||
          !e.writeVarS32(js::ObjectElements::offsetOfFlags()) ||
          !e.writeOp(Op::I32Add) || !e.writeOp(Op::LocalSet) ||
          !e.writeVarU32(be.propTaggedLocal))
        return EffectKind::Fail;
      if (!GetLocal(e, be.propTaggedLocal) || !GetLocal(e, be.propTaggedLocal) ||
          !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0) ||
          !e.writeOp(Op::I32Const) ||
          !e.writeVarS32(int32_t(js::ObjectElements::NON_PACKED)) ||
          !e.writeOp(Op::I32Or) ||
          !e.writeOp(Op::I32Store) || !e.writeVarU32(2) || !e.writeVarU32(0))
        return EffectKind::Fail;
      return EffectKind::Emitted;
    }
    case MDefinition::Opcode::SetInitializedLength: {
      // elements->initializedLength = index + 1 (Ion SetLengthFromIndex). Used to
      // finalize a dense array literal after its elements are stored (splay
      // GeneratePayloadTree's `[0,1,..,9]`).
      MDefinition* elems = ins->getOperand(0);
      MDefinition* idx = ins->getOperand(1);
      if (WJGETENV("GECKO_WJ_SILDBG"))
        fprintf(stderr, "[wj-sil] idx const=%d val=%d\n", idx->isConstant(),
                idx->isConstant() ? idx->toConstant()->toInt32() : -1);
      if (WJGETENV("GECKO_WJ_SILNOP")) return EffectKind::Emitted;  // isolate: no store
      int32_t off = js::ObjectElements::offsetOfInitializedLength();
      if (!GetOp(e, be, elems) || !e.writeOp(Op::I32Const) ||
          !e.writeVarS32(off) || !e.writeOp(Op::I32Add))
        return EffectKind::Fail;  // addr = elements + offsetOfInitializedLength
      if (!GetOp(e, be, idx) || !e.writeOp(Op::I32Const) || !e.writeVarS32(1) ||
          !e.writeOp(Op::I32Add))
        return EffectKind::Fail;  // value = index + 1
      if (!e.writeOp(Op::I32Store) || !e.writeVarU32(2) || !e.writeVarU32(0))
        return EffectKind::Fail;
      return EffectKind::Emitted;
    }
    case MDefinition::Opcode::MegamorphicStoreSlot: {
      MMegamorphicStoreSlot* m = ins->toMegamorphicStoreSlot();
      if (WJGETENV("GECKO_WJ_STOREDBG"))
        fprintf(stderr, "[wj-store] MegamorphicStoreSlot objType=%s\n",
                StringFromMIRType(m->object()->type()));
      uint64_t nameBits = js::IdToValue(m->name()).asRawBits();
      // Inline per-site polymorphic store IC (shape-compare -> direct slot store, no
      // C++ hop) -- richards' scheduler does 1.56M of these "megamorphic" named
      // stores, which were each a full VM SetObjectElement. Warp's megamorphic mark
      // is conservative; the real shape count per site is small, so the 4-way IC
      // mostly hits. WJH_SETPROPIC fills on miss. GECKO_WJ_NOMEGAIC reverts.
      static int noMegaIC = WJGETENV("GECKO_WJ_NOMEGAIC") ? 1 : 0;
      if (!noMegaIC && (m->object()->type() == MIRType::Object ||
                        m->object()->type() == MIRType::Value)) {
        uint32_t site = WJPropSite(be);
        if (site != 0) {
          return EmitPropStoreIC(e, be, ins, m->object(), nullptr, m->rhs(),
                                 m->strict(), site, nameBits)
                     ? EffectKind::Emitted
                     : EffectKind::Fail;
        }
      }
      if (!EmitStageScratch(e, be, m->object(), 0)) return EffectKind::Fail;
      if (!EmitStageConstBoxed(e, be, nameBits, 1)) return EffectKind::Fail;
      if (!EmitStageScratch(e, be, m->rhs(), 2)) return EffectKind::Fail;
      if (!EmitHelperCallResult(e, be, ins, WJH_SETPROP, m->strict() ? 1 : 0))
        return EffectKind::Fail;
      return e.writeOp(Op::Drop) ? EffectKind::Emitted : EffectKind::Fail;
    }
    case MDefinition::Opcode::MegamorphicSetElement: {
      // obj[index] = value (megamorphic). SetObjectElement handles by-value set +
      // accessors correctly (superset of the Ion fast path). op0=obj,1=index,2=value.
      if (WJGETENV("GECKO_WJ_NO_MSE")) return EffectKind::Fail;
      MMegamorphicSetElement* m = ins->toMegamorphicSetElement();
      if (!EmitStageScratch(e, be, m->object(), 0)) return EffectKind::Fail;
      if (!EmitStageScratch(e, be, m->index(), 1)) return EffectKind::Fail;
      if (!EmitStageScratch(e, be, m->value(), 2)) return EffectKind::Fail;
      if (!EmitHelperCallResult(e, be, ins, WJH_SETPROP, m->strict() ? 1 : 0))
        return EffectKind::Fail;
      return e.writeOp(Op::Drop) ? EffectKind::Emitted : EffectKind::Fail;
    }
    case MDefinition::Opcode::IteratorEnd: {
      // for-in loop cleanup: unlink the NativeIterator (js::CloseIterator). Non-
      // throwing for property iterators; cold -> minimal effect helper. (Speedometer
      // + octane hit this; un-bails the whole for-in fn. IteratorMore not needed --
      // for-of uses the iterator protocol, only for-in close surfaces here.)
      if (!EmitStageScratch(e, be, ins->getOperand(0), 0)) return EffectKind::Fail;  // iterator obj
      if (!EmitHelperCallResult(e, be, ins, WJH_ITEREND, 0)) return EffectKind::Fail;
      return e.writeOp(Op::Drop) ? EffectKind::Emitted : EffectKind::Fail;
    }
    case MDefinition::Opcode::CallSetArrayLength: {
      // arr.length = rhs. op0 = obj (guaranteed array), op1 = rhs (Value).
      if (WJGETENV("GECKO_WJ_NO_CSAL")) return EffectKind::Fail;
      MCallSetArrayLength* m = ins->toCallSetArrayLength();
      if (!EmitStageScratch(e, be, m->obj(), 0)) return EffectKind::Fail;
      if (!EmitStageScratch(e, be, m->rhs(), 1)) return EffectKind::Fail;
      if (!EmitHelperCallResult(e, be, ins, WJH_SETARRLEN, m->strict() ? 1 : 0))
        return EffectKind::Fail;
      return e.writeOp(Op::Drop) ? EffectKind::Emitted : EffectKind::Fail;
    }
    case MDefinition::Opcode::SetPropertyCache: {
      MSetPropertyCache* s = ins->toSetPropertyCache();
      // Inline the per-site store IC for object receivers (writable own data
      // property -> direct slot store + barriers, no C++ hop); else generic set.
      static int noPropIC = WJGETENV("GECKO_WJ_NOPROPIC") ? 1 : 0;
      if (WJGETENV("GECKO_WJ_STOREDBG"))
        fprintf(stderr, "[wj-store] SetPropertyCache objType=%s idvalType=%s idvalConst=%d valType=%s\n",
                StringFromMIRType(s->object()->type()),
                StringFromMIRType(s->idval()->type()), s->idval()->isConstant(),
                StringFromMIRType(s->value()->type()));
      // Property-INIT ops (InitElem/InitProp families -- incl. the self-hosted
      // DefineDataProperty intrinsic, which compiles to InitElem bytecode) have
      // DEFINE semantics: an OOB typed-array define must THROW where a plain
      // set silently skips (jit-test typedarray/define-property-oob). Route
      // them through the generic helper with the init attrs encoded in the
      // site word (bit0 strict, bit1 init, attrs<<2); the inline store IC is
      // set-semantics only. Init-op SetPropertyCache is rare (object-literal
      // inits normally transpile to template stores), so no perf concern.
      JSOp sop = JSOp::Nop;
      bool haveOp = false;
      if (MResumePoint* srp = ins->resumePoint()) {
        sop = JSOp(*srp->pc());
        haveOp = true;
      } else if (be.curRp) {
        sop = JSOp(*be.curRp->pc());
        haveOp = true;
      }
      bool isInit = haveOp && IsPropertyInitOp(sop);
      unsigned initAttrs = 0;
      if (isInit) {
        switch (sop) {
          case JSOp::InitProp:
          case JSOp::InitElem:
          case JSOp::InitElemInc:
            initAttrs = JSPROP_ENUMERATE;
            break;
          case JSOp::InitLockedProp:
          case JSOp::InitLockedElem:
            initAttrs = JSPROP_PERMANENT | JSPROP_READONLY;
            break;
          case JSOp::InitHiddenProp:
          case JSOp::InitHiddenElem:
            initAttrs = 0;
            break;
          default:
            // Unexpected init-family op: bail to PBL rather than guess attrs.
            js::wasm::gWJBailReason = "setpropcache-init-op";
            return EffectKind::Fail;
        }
      }
      if (!noPropIC && !isInit &&
          (s->object()->type() == MIRType::Object ||
           s->object()->type() == MIRType::Value)) {
        uint32_t site = WJPropSite(be);
        if (site != 0) {
          return EmitPropStoreIC(e, be, ins, s->object(), s->idval(), s->value(),
                                 s->strict(), site)
                     ? EffectKind::Emitted
                     : EffectKind::Fail;
        }
      }
      if (!EmitStageScratch(e, be, s->object(), 0)) return EffectKind::Fail;
      if (!EmitStageScratch(e, be, s->idval(), 1)) return EffectKind::Fail;
      if (!EmitStageScratch(e, be, s->value(), 2)) return EffectKind::Fail;
      uint32_t hsite = (s->strict() ? 1u : 0u) | (isInit ? 2u : 0u) |
                       (initAttrs << 2);
      if (!EmitHelperCallResult(e, be, ins, WJH_SETPROP, hsite))
        return EffectKind::Fail;
      return e.writeOp(Op::Drop) ? EffectKind::Emitted : EffectKind::Fail;
    }
    case MDefinition::Opcode::PostWriteBarrier: {
      // Record the container in the GC store buffer after an object was stored
      // into one of its slots/elements. WITHOUT this, a tenured container's
      // edge to a nursery value is invisible to a minor GC, which frees the
      // value -> the slot reads back garbage (was the raytrace `dot` crash).
      int32_t ol = be.local(ins->getOperand(0));
      if (ol < 0) return EffectKind::Fail;
      // Inline isTenured() gate (skip the helper for nursery containers).
      if (!EmitForcePostBarrier(e, be, ins->getOperand(0), ins->getOperand(1)))
        return EffectKind::Fail;
      return EffectKind::Emitted;
    }
    case MDefinition::Opcode::KeepAliveObject:
      // Pure GC-liveness marker; our GC-root shadow stack already keeps call
      // operands alive, so this is a no-op for codegen.
      return EffectKind::Skip;
    default:
      // Pure node whose value is unused (DCE leftover, or a type we don't map to
      // a wasm local): safe to skip. Unknown EFFECTFUL node: bail the function.
      // An unimplemented alias-set STORE (not isEffectful() -- infallible stores
      // like StoreHoleValueElement carry no resume point) must ALSO bail:
      // skipping it silently drops a memory write and corrupts the object
      // (webtooling lodash `[,-0]` built packed-flagged with a hole). Same for
      // an unimplemented GUARD: a None-typed guard (GuardElementsArePacked)
      // has no value uses, so the Skip default silently DROPPED the check the
      // downstream specialized code relies on (hole-check-free LoadElement).
      if (ins->isEffectful() || ins->getAliasSet().isStore() || ins->isGuard()) {
        WJTallyBail(ins->op());
        return EffectKind::Fail;
      }
      return EffectKind::Skip;
  }
}

// Store the (boxed) return value to gWJScratch[result] and return ok (0.0).
// Pop the persistent env-root shadow slot at a function exit (restore gWJRootSP
// to its entry value, envRootIdx). No-op unless the function rooted its env.
// Net-zero on the wasm value stack (pure memory store), so safe to interleave
// with an in-progress [flag,result] on the stack.
static bool EmitEnvRootPop(Encoder& e, WJBackend& be) {
  if (!be.useEnvRoot) return true;
  uintptr_t spAddr = uintptr_t(static_cast<void*>(&gWJRootSP));
  return e.writeOp(Op::I32Const) && e.writeVarS32(int32_t(spAddr)) &&
         GetLocal(e, be.envRootIdx) && e.writeOp(Op::I32Store) &&
         e.writeVarU32(2) && e.writeVarU32(0);
}

// Restore gWJJitDepth to this frame's entry value (no-op unless the entry block
// emitted the recursion save/check). Net-zero on the wasm value stack.
static bool EmitDepthPop(Encoder& e, WJBackend& be) {
  if (!be.depthChecked) return true;
  return e.writeOp(Op::I32Const) &&
         e.writeVarS32(int32_t(uintptr_t(static_cast<void*>(&gWJJitDepth)))) &&
         GetLocal(e, be.savedDepthLocal) && e.writeOp(Op::I32Store) &&
         e.writeVarU32(2) && e.writeVarU32(0);
}

// Once-per-function (entry block): savedDepthLocal = gWJJitDepth; gWJJitDepth += 1;
// if (gWJJitDepth > limit) deopt-to-PBL. Bounds JIT->JIT recursion below V8's wasm
// stack; the deep tail runs in PBL (64MB shadow stack, real over-recursion guard).
static bool EmitDepthCheck(Encoder& e, WJBackend& be) {
  // DEFAULT-ON (GECKO_WJ_NOSTACKGUARD reverts). Bounds JIT->JIT recursion below
  // V8's wasm-stack hard limit: the guard raises a CATCHABLE over-recursion (throw
  // mode) instead of the uncatchable V8 stack crash; the deep tail runs in PBL
  // (64MB shadow stack, real over-recursion quota). Cheap because of LEAF-SKIP
  // (only call-containing fns are guarded) and correct in dispatch bodies too.
  // Byte-accounting: gWJJitDepth accumulates estimated frame BYTES, not counts.
  // (History: was an opt-in prototype; the early flag-2-valve regressions --
  // crypto/acorn ERR -- were resolved by throw-mode + the deopt-restore fix.)
  static int enabled =
      WJGETENV("GECKO_WJ_NOSTACKGUARD") ? 0 : 1;
  if (!enabled || !be.hasCalls) return true;
  // BYTE-ACCOUNTING guard: gWJJitDepth accumulates estimated V8 wasm-stack BYTES
  // (this frame's frameBytesEst), not a flat count -- so the limit fires by
  // approximate stack usage (like a real SP check) and correctly distinguishes a
  // big-frame fn (few levels to overflow) from a small one. The limit is in BYTES
  // ~= V8's wasm execution-stack size minus a safety margin (GECKO_WJ_DEPTHLIMIT
  // overrides, in bytes).
  static int limit = WJGETENV("GECKO_WJ_DEPTHLIMIT")
                         ? atoi(WJGETENV("GECKO_WJ_DEPTHLIMIT")) : 2500000;
  int32_t fb = int32_t(be.frameBytesEst);
  int32_t dAddr = int32_t(uintptr_t(static_cast<void*>(&gWJJitDepth)));
  // savedDepthLocal = gWJJitDepth
  if (!e.writeOp(Op::I32Const) || !e.writeVarS32(dAddr) || !e.writeOp(Op::I32Load) ||
      !e.writeVarU32(2) || !e.writeVarU32(0) || !e.writeOp(Op::LocalSet) ||
      !e.writeVarU32(be.savedDepthLocal))
    return false;
  // gWJJitDepth = savedDepthLocal + frameBytesEst
  if (!e.writeOp(Op::I32Const) || !e.writeVarS32(dAddr) ||
      !GetLocal(e, be.savedDepthLocal) || !e.writeOp(Op::I32Const) ||
      !e.writeVarS32(fb) || !e.writeOp(Op::I32Add) || !e.writeOp(Op::I32Store) ||
      !e.writeVarU32(2) || !e.writeVarU32(0))
    return false;
  be.depthChecked = true;  // pairs the restore at every exit
  // if (savedDepthLocal + frameBytesEst > limit): set the catchable over-recursion
  // exception (js::ReportOverRecursed, exactly what PBL throws at its native-stack
  // quota) and propagate via the exception exit. We THROW here, not deopt-to-PBL: a
  // deopt re-runs THIS frame in PBL but the recursive callee is still JIT-compiled,
  // so PBL calls back into the JIT each level and V8's wasm stack keeps growing ->
  // still crashes. Throwing unwinds the whole recursion catchably (== PBL behavior).
  if (!GetLocal(e, be.savedDepthLocal) || !e.writeOp(Op::I32Const) ||
      !e.writeVarS32(fb) || !e.writeOp(Op::I32Add) || !e.writeOp(Op::I32Const) ||
      !e.writeVarS32(limit) || !e.writeOp(Op::I32GtS))
    return false;
  if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
  // THROW is the DEFAULT (GECKO_WJ_NODEPTHTHROW reverts to the flag-2 valve): a throw
  // unwinds V8's wasm stack cleanly; the valve regrows it (PBL re-enters the JIT) and
  // still crashes at deep recursion.
  static int depthThrow = WJGETENV("GECKO_WJ_NODEPTHTHROW") ? 0 : 1;
  if (depthThrow) {
    // THROW MODE: raise the catchable over-recursion exception and propagate out.
    // A throw UNWINDS V8's wasm execution stack cleanly (each frame returns the
    // throw flag), unlike the flag-2 valve below which routes to PBL that then
    // RE-ENTERS the JIT-compiled callee -> regrows V8's stack -> still crashes at
    // deep recursion (sum(15000)). Restore depth + pop the env root (as the valve
    // does), set the pending exception (js::ReportOverRecursed via WJH_OVERRECURSED),
    // then return the throw flag [1.0,0] (or EHABI wasm-exn). Pick a limit BELOW the
    // crash but ABOVE any benchmark's real recursion so the guard only fires for
    // genuinely-too-deep recursion. GECKO_WJ_DEPTHTHROW enables (default = valve).
    if (!e.writeOp(Op::I32Const) || !e.writeVarS32(dAddr) ||
        !GetLocal(e, be.savedDepthLocal) || !e.writeOp(Op::I32Store) ||
        !e.writeVarU32(2) || !e.writeVarU32(0))
      return false;
    if (!EmitEnvRootPop(e, be)) return false;
    if (!e.writeOp(Op::F64Const) || !e.writeFixedF64(double(WJH_OVERRECURSED)) ||
        !e.writeOp(Op::F64Const) || !e.writeFixedF64(0.0) ||
        !e.writeOp(Op::Call) || !e.writeVarU32(0) || !e.writeOp(Op::Drop))
      return false;
    if constexpr (kWJEHABI) {
      if (!EmitThrowWjexn(e)) return false;
    } else {
      if (!e.writeOp(Op::F64Const) || !e.writeFixedF64(1.0) ||
          !e.writeOp(Op::I64Const) || !e.writeVarS64(0) || !e.writeOp(Op::Return))
        return false;
    }
  } else {
    // FLAG-2 "didn't run": restore the depth (this frame is aborting) and let the
    // CALLER route the call to the interpreter/PBL (slow path). Each deeper level
    // repeats this, so the deep tail runs interpreted on small C++ frames under
    // the engine's real (catchable) recursion quota -- no V8-stack crash, and no
    // false "too much recursion" for legitimately-deep-but-finite recursion.
    if (!e.writeOp(Op::I32Const) || !e.writeVarS32(dAddr) ||
        !GetLocal(e, be.savedDepthLocal) || !e.writeOp(Op::I32Store) ||
        !e.writeVarU32(2) || !e.writeVarU32(0))
      return false;
    // The depth check runs AFTER the prologue's env/callee root push: pop it, or
    // every valved call leaks gWJRootSP slots (gWJCallRoots overflowed -> silent
    // corruption on deep-tail churn).
    if (!EmitEnvRootPop(e, be)) return false;
    if (!e.writeOp(Op::F64Const) || !e.writeFixedF64(2.0) ||
        !e.writeOp(Op::I64Const) ||
        !e.writeVarS64(int64_t(JS::UndefinedValue().asRawBits())) ||
        !e.writeOp(Op::Return))
      return false;
  }
  if (!e.writeOp(Op::End)) return false;
  return true;
}

static bool EmitReturn(Encoder& e, WJBackend& be, MDefinition* val) {
  // EHABI: return just the i64 boxed result (exceptions via wasm EH, no flag). Old ABI:
  // return [f64 flag=0.0, i64 boxed-result] in registers.
  if constexpr (!kWJEHABI) {
    if (!e.writeOp(Op::F64Const) || !e.writeFixedF64(0.0)) return false;  // ok flag
  }
  if (!GetOp(e, be, val)) return false;
  if (val->type() != MIRType::Value && val->type() != MIRType::Int64) {
    if (!EmitBoxFromStack(e, val->type())) return false;
  }
  if (!EmitEnvRootPop(e, be)) return false;
  if (!EmitDepthPop(e, be)) return false;
  return e.writeOp(Op::Return);  // EHABI: [result]; old: [flag, result]
}

// Per-def/phi VALUE tracer emit (GECKO_WJ_VALUETRACE=<lineno>): store the local's
// value bits to gWJTraceVal, then call wjhelp(251, id) so the runtime records (id,val).
static bool EmitValueTrace(Encoder& e, WJBackend& be, uint32_t localIdx, uint8_t ty,
                           uint32_t id) {
  static int vtLine = -2;
  if (vtLine == -2) {
    const char* s = WJGETENV("GECKO_WJ_VALUETRACE");
    vtLine = s ? atoi(s) : -1;
  }
  if (vtLine < 0 || !be.info || !be.info->script() ||
      int(be.info->script()->lineno()) != vtLine)
    return true;
  if (ty != uint8_t(TypeCode::I32) && ty != uint8_t(TypeCode::I64) &&
      ty != uint8_t(TypeCode::F64))
    return true;
  if (!e.writeOp(Op::I32Const) ||
      !e.writeVarS32(int32_t(uintptr_t(&js::wasm::gWJTraceVal))))
    return false;
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(localIdx)) return false;
  if (ty == uint8_t(TypeCode::I32)) {
    if (!e.writeOp(Op::I64ExtendI32S)) return false;
  } else if (ty == uint8_t(TypeCode::F64)) {
    if (!e.writeOp(Op::I64ReinterpretF64)) return false;
  }
  if (!e.writeOp(Op::I64Store) || !e.writeVarU32(3) || !e.writeVarU32(0)) return false;
  return e.writeOp(Op::F64Const) && e.writeFixedF64(251.0) &&
         e.writeOp(Op::F64Const) && e.writeFixedF64(double(id)) &&
         e.writeOp(Op::Call) && e.writeVarU32(0) && e.writeOp(Op::Drop);
}

static bool EmitEdgeCopies(Encoder& e, WJBackend& be, MBasicBlock* from,
                           MBasicBlock* to) {
  uint32_t k = to->getPredecessorIndex(from);
  std::vector<int32_t> dsts;
  std::vector<MPhi*> phis;
  for (MPhiIterator p = to->phisBegin(); p != to->phisEnd(); p++) {
    MPhi* phi = *p;
    int32_t dst = be.local(phi);
    if (dst < 0) continue;
    if (!GetOp(e, be, phi->getOperand(k))) return false;  // push old value
    dsts.push_back(dst);
    phis.push_back(phi);
  }
  for (size_t i = dsts.size(); i-- > 0;) {
    if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(uint32_t(dsts[i]))) return false;
  }
  // VALUE TRACE (GECKO_WJ_VALUETRACE): record each phi's value on this edge, so the
  // switch-merge phi (the reloop miscompile) is visible. phiId encoded as id (biased
  // by 1000000 so it reads distinctly from straight-line defs in the dump).
  for (size_t i = 0; i < phis.size(); i++) {
    if (!EmitValueTrace(e, be, uint32_t(dsts[i]), WJValType(phis[i]->type()),
                        1000000 + phis[i]->id()))
      return false;
  }
  return true;
}

static bool GotoBlock(Encoder& e, uint32_t bidLocal, uint32_t loopDepth,
                      uint32_t targetIdx) {
  return e.writeOp(Op::I32Const) && e.writeVarS32(int32_t(targetIdx)) &&
         e.writeOp(Op::LocalSet) && e.writeVarU32(bidLocal) && e.writeOp(Op::Br) &&
         e.writeVarU32(loopDepth);
}

// Push i32 JS-truthiness (0/1) of the boxed Value in wasm local `v`. Covers
// int32/boolean/object/undefined/null/double; deopts to PBL for the rarer
// string/symbol/bigint (which need a length/zero check we don't emit yet).
static bool EmitValueTruthy(Encoder& e, WJBackend& be, uint32_t v) {
  const uint8_t kI32 = uint8_t(TypeCode::I32);
  auto tag = [&]() -> bool {
    return GetLocal(e, v) && e.writeOp(Op::I64Const) && e.writeVarS64(32) &&
           e.writeOp(Op::I64ShrU) && e.writeOp(Op::I32WrapI64);
  };
  auto tagEq = [&](JSValueType t) -> bool {
    return tag() && e.writeOp(Op::I32Const) &&
           e.writeVarS32(int32_t(TagWord(t))) && e.writeOp(Op::I32Eq);
  };
  if (!tagEq(JSVAL_TYPE_INT32)) return false;
  if (!e.writeOp(Op::If) || !e.writeFixedU8(kI32)) return false;
  if (!GetLocal(e, v) || !e.writeOp(Op::I32WrapI64) || !e.writeOp(Op::I32Const) ||
      !e.writeVarS32(0) || !e.writeOp(Op::I32Ne))
    return false;
  if (!e.writeOp(Op::Else)) return false;
  if (!tagEq(JSVAL_TYPE_BOOLEAN)) return false;
  if (!e.writeOp(Op::If) || !e.writeFixedU8(kI32)) return false;
  if (!GetLocal(e, v) || !e.writeOp(Op::I32WrapI64)) return false;
  if (!e.writeOp(Op::Else)) return false;
  if (!tagEq(JSVAL_TYPE_OBJECT)) return false;
  if (!e.writeOp(Op::If) || !e.writeFixedU8(kI32)) return false;
  if (!e.writeOp(Op::I32Const) || !e.writeVarS32(1)) return false;
  if (!e.writeOp(Op::Else)) return false;
  if (!tagEq(JSVAL_TYPE_UNDEFINED)) return false;
  if (!tagEq(JSVAL_TYPE_NULL)) return false;
  if (!e.writeOp(Op::I32Or)) return false;
  if (!e.writeOp(Op::If) || !e.writeFixedU8(kI32)) return false;
  if (!e.writeOp(Op::I32Const) || !e.writeVarS32(0)) return false;
  if (!e.writeOp(Op::Else)) return false;
  // double: tag u<= CLEAR => (d != 0) & (d == d)  [d==d is false for NaN]
  if (!tag() || !e.writeOp(Op::I32Const) ||
      !e.writeVarS32(int32_t(uint32_t(JSVAL_TAG_CLEAR))) || !e.writeOp(Op::I32LeU))
    return false;
  if (!e.writeOp(Op::If) || !e.writeFixedU8(kI32)) return false;
  if (!GetLocal(e, v) || !e.writeOp(Op::F64ReinterpretI64) ||
      !e.writeOp(Op::F64Const) || !e.writeFixedF64(0.0) || !e.writeOp(Op::F64Ne))
    return false;
  if (!GetLocal(e, v) || !e.writeOp(Op::F64ReinterpretI64) || !GetLocal(e, v) ||
      !e.writeOp(Op::F64ReinterpretI64) || !e.writeOp(Op::F64Eq))
    return false;
  if (!e.writeOp(Op::I32And)) return false;
  if (!e.writeOp(Op::Else)) return false;
  // not-double => string / symbol / bigint. INLINE string truthiness (len != 0) and
  // symbol (always truthy) instead of deopting -- this is what Ion does, AND it avoids
  // a deopt-RESUME that is unsound for the `!(x = call())` shape: pdfjs Lexer_getObj's
  // `if(!(ch=stream.getChar()))` test deopted here on every (string) char, and the
  // resume-after-call reconstruction returned a wrong token ("Malformed PDF"). Inlining
  // removes the deopt entirely -> correct + faster, and it unblocks STRCMP2 (the
  // Compare_String coverage that let getObj compile far enough to reach this test).
  // BigInt still deopts (rare; truthiness needs the digit/length check). GECKO_WJ_
  // NOSTRTRUTHY reverts to the old deopt-all path.
  static int noStrTruthy = WJGETENV("GECKO_WJ_NOSTRTRUTHY") ? 1 : 0;
  if (!noStrTruthy) {
    uint32_t lenOff = uint32_t(JSString::offsetOfLength());
    if (!tagEq(JSVAL_TYPE_STRING)) return false;
    if (!e.writeOp(Op::If) || !e.writeFixedU8(kI32)) return false;  // (6) string
    if (!GetLocal(e, v) || !e.writeOp(Op::I32WrapI64) || !e.writeOp(Op::I32Load) ||
        !e.writeVarU32(2) || !e.writeVarU32(lenOff) || !e.writeOp(Op::I32Const) ||
        !e.writeVarS32(0) || !e.writeOp(Op::I32Ne))
      return false;  // len != 0
    if (!e.writeOp(Op::Else)) return false;
    if (!tagEq(JSVAL_TYPE_SYMBOL)) return false;
    if (!e.writeOp(Op::If) || !e.writeFixedU8(kI32)) return false;  // (7) symbol
    if (!e.writeOp(Op::I32Const) || !e.writeVarS32(1)) return false;  // always truthy
    if (!e.writeOp(Op::Else)) return false;
  }
  // bigint (+ anything unexpected) -> PBL. The deopt sits 5 (or 7 with the string/
  // symbol inline above) `if(result i32)` levels deep, so the OOL deopt br-depth needs
  // +4 (or +6) over its usual single-`if` assumption.
  be.deoptExtraNest = noStrTruthy ? 4 : 6;
  bool dok = EmitDeopt(e, be);
  be.deoptExtraNest = 0;
  if (!dok) return false;
  if (!e.writeOp(Op::I32Const) || !e.writeVarS32(0)) return false;  // dead
  if (!noStrTruthy) {
    if (!e.writeOp(Op::End)) return false;  // (7) symbol if
    if (!e.writeOp(Op::End)) return false;  // (6) string if
  }
  for (int i = 0; i < 5; i++)
    if (!e.writeOp(Op::End)) return false;
  return true;
}

// Emit one block's non-control instructions. Returns false on unsupported node.
// Op/block-level BISECTION tool: GECKO_WJ_FORCEDEOPTOFF=<byteoff> (+ optional
// GECKO_WJ_FORCEDEOPTLINE=<fn lineno> to scope to one function) forces a
// deopt-at-entry (run the rest in PBL, like alwaysBails) for every block whose
// entry bytecode offset >= <byteoff>. Blocks BEFORE the offset stay in JIT; blocks
// at/after run PBL. Binary-search <byteoff> to localize a miscompiled op: the
// smallest offset at which the result becomes WRONG is just past the bad op (it
// just entered JIT). Deterministic from (block, env), so the terminator-skip sites
// check it too (a force-deopted block emits no terminator, like alwaysBails).
static bool WJBlockForceDeopt(WJBackend& be, MBasicBlock* b) {
  static int off = WJGETENV("GECKO_WJ_FORCEDEOPTOFF")
                       ? atoi(WJGETENV("GECKO_WJ_FORCEDEOPTOFF"))
                       : -1;
  // RANGE mode (GECKO_WJ_FORCEDEOPTRANGE=lo,hi): deopt ONLY blocks whose entry
  // offset is in [lo,hi), keeping the rest (incl. the loop latch) in JIT. This
  // isolates a single bad block WITHOUT the loop-escape confound of the suffix
  // (>=off) mode -- correct result iff the bad block is in [lo,hi).
  static int rlo = -1, rhi = -1;
  static bool rangeInit = false;
  if (!rangeInit) {
    rangeInit = true;
    if (const char* r = WJGETENV("GECKO_WJ_FORCEDEOPTRANGE")) {
      rlo = atoi(r);
      const char* c = strchr(r, ',');
      rhi = c ? atoi(c + 1) : (rlo + 1);
    }
  }
  // FN-LINENO-RANGE mode (GECKO_WJ_FORCEDEOPTLINERANGE=lo,hi): force EVERY block of
  // any function whose script lineno is in [lo,hi] to deopt-at-entry (=> that whole
  // function runs in PBL). Binary-search lo/hi to find which function miscompiles.
  static int flo = -1, fhi = -1;
  static bool flrInit = false;
  if (!flrInit) {
    flrInit = true;
    if (const char* r = WJGETENV("GECKO_WJ_FORCEDEOPTLINERANGE")) {
      flo = atoi(r);
      const char* c = strchr(r, ',');
      fhi = c ? atoi(c + 1) : (flo + 1);
    }
  }
  if (off < 0 && rlo < 0 && flo < 0) return false;
  static int wantLine = WJGETENV("GECKO_WJ_FORCEDEOPTLINE")
                            ? atoi(WJGETENV("GECKO_WJ_FORCEDEOPTLINE"))
                            : 0;
  uint32_t fnLine =
      be.info && be.info->script() ? be.info->script()->lineno() : 0;
  if (flo >= 0)
    return fnLine >= uint32_t(flo) && fnLine <= uint32_t(fhi) && !b->alwaysBails();
  if (wantLine && fnLine != uint32_t(wantLine)) return false;
  if (b->alwaysBails()) return false;  // already handled
  MResumePoint* rp = b->entryResumePoint();
  if (!rp || !rp->pc()) return false;
  JSScript* s = rp->block()->info().script();
  if (!s) return false;
  uint32_t boff = uint32_t(rp->pc() - s->code());
  if (rlo >= 0) return boff >= uint32_t(rlo) && boff < uint32_t(rhi);
  return boff >= uint32_t(off);
}

static bool EmitBlockBody(Encoder& e, WJBackend& be, MBasicBlock* b) {
  // Remat cache: clear at every block boundary (a value cached on one CF path is not
  // valid at a merge; conservative + sound).
  be.rematInvalidate();
  // BLOCKDUMP: print this block's entry offset + its MIR ops (scoped to the fn at
  // GECKO_WJ_BLOCKDUMP=<lineno>). Localizes which op is in a bisected bad block.
  if (const char* bd = WJGETENV("GECKO_WJ_BLOCKDUMP")) {
    uint32_t wl = uint32_t(atoi(bd));
    uint32_t fl = be.info && be.info->script() ? be.info->script()->lineno() : 0;
    MResumePoint* erp = b->entryResumePoint();
    uint32_t bo = (erp && erp->pc() && b->info().script())
                      ? uint32_t(erp->pc() - b->info().script()->code()) : 999999;
    if (fl == wl) {
      fprintf(stderr, "[blockdump] fn:%u boff=%u ops:", fl, bo);
      for (MInstructionIterator it = b->begin(); it != b->end(); it++)
        fprintf(stderr, " %s", WJOpName(it->op()));
      fprintf(stderr, "\n");
    }
  }
  // JIT execution tracer (GECKO_WJ_TRACEFN=<lineno>): emit WJH_TRACE(blockOffset) at
  // this block's entry, so the runtime ring buffer records the exact block-execution
  // PATH of this function. Wasm-level (doesn't perturb JS warmup like JS wrappers do).
  static int traceFn = WJGETENV("GECKO_WJ_TRACEFN") ? atoi(WJGETENV("GECKO_WJ_TRACEFN")) : -1;
  if (traceFn > 0 && be.info && be.info->script() &&
      int(be.info->script()->lineno()) == traceFn) {
    MResumePoint* trp = b->entryResumePoint();
    uint32_t tbo = (trp && trp->pc() && b->info().script())
                       ? uint32_t(trp->pc() - b->info().script()->code()) : 0xFFFF;
    if (!e.writeOp(Op::F64Const) || !e.writeFixedF64(250.0 /*WJH_TRACE, unique sentinel;
        MUST NOT collide with any WJHelpKind -- was 40 == WJH_ISARRAY, which broke
        Array.isArray*/) ||
        !e.writeOp(Op::F64Const) || !e.writeFixedF64(double(tbo)) ||
        !e.writeOp(Op::Call) || !e.writeVarU32(0) || !e.writeOp(Op::Drop))
      return false;
  }
  // Bisection: force this block to deopt-at-entry (see WJBlockForceDeopt).
  if (WJBlockForceDeopt(be, b)) {
    be.curRp = b->entryResumePoint();
    bool sd = be.inDispatchBody;
    be.inDispatchBody = false;
    bool ok = EmitDeopt(e, be);
    be.inDispatchBody = sd;
    return ok;
  }
  // alwaysBails block: Warp couldn't type this (cold) path and marked it to bail
  // unconditionally. Emit a deopt-at-entry (resume to PBL from the block's entry
  // pc) and skip the body+terminator -- the caller checks alwaysBails() to elide
  // the terminator. Force INLINE resume (ends in `return`, no br-depth dependence
  // on the guard-if nesting the OOL path assumes). Keeps the function's hot blocks
  // compiled; this block only runs (deopts) if actually reached. (navier fix.)
  if (b->alwaysBails()) {
    gWJHadAlwaysBails = true;  // valve: a storming fn with this -> PBL, not recompile
    static int abdbg = WJGETENV("GECKO_WJ_ABDBG") ? 1 : 0;
    be.curRp = b->entryResumePoint();
    if (!be.curRp) {
      if (abdbg) {
        JSScript* s = be.info ? be.info->script() : nullptr;
        fprintf(stderr, "[wj-abdbg] %s:%u alwaysBails block has NO entry RP\n",
                s ? s->filename() : "?", s ? unsigned(s->lineno()) : 0);
      }
      js::wasm::gWJBailReason = "alwaysbails-no-entry-rp";
      return false;
    }
    bool savedDispatch = be.inDispatchBody;
    be.inDispatchBody = false;
    bool ok = EmitDeopt(e, be);
    be.inDispatchBody = savedDispatch;
    if (!ok && abdbg) {
      JSScript* s = be.info ? be.info->script() : nullptr;
      fprintf(stderr, "[wj-abdbg] %s:%u alwaysBails EmitDeopt FAILED\n",
              s ? s->filename() : "?", s ? unsigned(s->lineno()) : 0);
    }
    return ok;
  }
  // Track the MOST-RECENT resume point in program order. A deopt at an op WITHOUT
  // its own resume point must resume from the last one SEEN (which reflects the
  // state AFTER the last committed side effect), NOT the block-entry RP -- else
  // PBL re-runs from the block start and RE-APPLIES already-committed stores
  // (randtest seed `+=`, navier loop stores -> wrong values). This is the sound
  // resume-point selection (mirrors Ion, where every bailing op has the correct
  // nearest dominating resume point).
  MResumePoint* lastRp = b->entryResumePoint();
  // RP-FALLBACK: some blocks (esp. post-call split blocks in raytrace's construct/
  // method-call path, and gbemu's polymorphic dispatch) have a null
  // entryResumePoint. A fallible guard there can't emit its deopt (curRp null) ->
  // the WHOLE function bailed to PBL. For a single-predecessor block the
  // predecessor's exit rp IS the correct entry resume state (re-execute from
  // there), so carry it forward. Multi-predecessor merges get their rp from Warp.
  // Gated until validated (GECKO_WJ_NORPFALLBACK reverts).
  if (!lastRp && !WJGETENV("GECKO_WJ_NORPFALLBACK") && b->numPredecessors() == 1) {
    auto it = be.blockExitRp.find(b->getPredecessor(0));
    if (it != be.blockExitRp.end()) lastRp = it->second;
  }
  be.curRp = lastRp;  // valid even for an empty block (terminator uses it)
  // Recursion guard (once per fn, at the first block reached with a valid resume
  // point = the entry): bump gWJJitDepth and deopt the whole fn to PBL past the
  // limit, so deep JIT->JIT recursion can't overflow V8's wasm stack uncatchably.
  // Fires for call-containing fns ONLY (leaf-skip = cheap). Fires even when the
  // entry is inside a relooper dispatch body (the prior bypass that let cdjs's
  // recursion skip the guard) -- inDispatchBody is cleared so EmitDeopt uses the
  // inline-return form (a wasm `return`, valid at any block nesting).
  if (!be.depthChecked && be.curRp && be.hasCalls) {
    bool sd = be.inDispatchBody;
    be.inDispatchBody = false;
    bool ok = EmitDepthCheck(e, be);
    be.inDispatchBody = sd;
    if (!ok) return false;
  }
  for (MInstructionIterator it = b->begin(); it != b->end(); it++) {
    MInstruction* ins = *it;
    if (ins->isControlInstruction()) continue;
    MResumePoint* insRp = ins->resumePoint();
    if (WJGETENV("GECKO_WJ_RPDBG") && be.info && be.info->script() &&
        be.info->script()->lineno() == uint32_t(atoi(WJGETENV("GECKO_WJ_RPDBG")))) {
      fprintf(stderr, "[wj-rp] b%u op#%u(%s) rp=%s mode=%d\n", b->id(),
              unsigned(ins->op()), WJOpName(ins->op()), insRp ? "Y" : "n",
              insRp ? int(insRp->mode()) : -1);
    }
    // Deopt resume-point selection. Advance the deopt rp to this op's rp BEFORE
    // the op ONLY for resume-AT points: a mid-op guard then re-executes this op in
    // PBL from its own pc. For resume-AFTER points, KEEP the PRIOR rp during the op
    // -- the resume-after point's captured stack includes this op's RESULT, but a
    // mid-op guard deopts BEFORE the result is computed (before the local.set
    // below), so spilling the resume-after frame would spill the op's STALE/uninit
    // result local as the captured stack top -> PBL resumes with a garbage stack
    // value. The prior rp (the previous op's resume-after) captures all prior
    // effects and resumes just before this op, so PBL correctly re-executes ONLY
    // this op. (deltablue's LoadDynamicSlotAndUnbox fallible-unbox tag-guard hit
    // this: the bad spilled result fed the constraint solver -> "Cycle encountered"
    // / hang.) Advance to insRp AFTER the op completes (its result now in-local).
    bool spillAfter = insRp && IsResumeAfter(insRp->mode());
    if (insRp && !spillAfter) lastRp = insRp;
    be.curRp = lastRp;
    be.curOp = uint32_t(ins->op());
    if (be.info && be.info->script())
      gWJCurScriptLine = uint32_t(be.info->script()->lineno());
    int32_t l = be.local(ins);
    if (l < 0) {
      // No value local: either a no-op typed node or an effectful one.
      EffectKind k = EmitEffect(e, be, ins);
      if (k == EffectKind::Fail) {
        fprintf(stderr, "[wb-be] unsupported effect op#%u %s\n",
                unsigned(ins->op()), WJOpName(ins->op()));
        return false;
      }
    } else {
      if (!EmitValue(e, be, ins)) {
        JSScript* bs = be.info ? be.info->script() : nullptr;
        fprintf(stderr, "[wb-be] unsupported value op#%u %s type%u fn@%u\n",
                unsigned(ins->op()), WJOpName(ins->op()), unsigned(ins->type()),
                bs ? unsigned(bs->lineno()) : 0u);
        return false;
      }
      if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(uint32_t(l))) {
        WJSetBailOp(ins);
        js::wasm::gWJBailReason = "value-localset";
        return false;
      }
      // Per-def VALUE tracer (GECKO_WJ_VALUETRACE=<lineno>): see EmitValueTrace.
      if (!EmitValueTrace(e, be, uint32_t(l), WJValType(ins->type()), ins->id()))
        return false;
    }
    // Resume-AFTER op completed: its result is now in-local, so the resume-after
    // frame is consistent. Advance the deopt rp for subsequent ops, the terminator
    // (which keeps the body's last rp), + the shadow post-spill below.
    if (spillAfter) {
      lastRp = insRp;
      be.curRp = lastRp;
    }
  }
  be.blockExitRp[b] = lastRp;  // for the single-pred rp-fallback (see above)
  return true;
}

// Emit a block terminator. loopDepth = depth of the dispatch loop $L from here.
// Push the br_table i32 index for a boxed-Value switch discriminant held in
// local `dl`: an INT32-tagged value uses its payload; a DOUBLE whose f64 is an
// EXACT int32 uses the converted int (incl -0 -> 0: switch is `===` and
// -0===0, so `case 0` must match -0 and `case 2` must match 2.0 --
// ion/bug1556571); anything else pushes `defaultIdx` (routes to the default
// after the `- low` adjustment). Reuses be.propTaggedLocal as scratch (safe at
// a terminator: no prop-IC emit is in flight).
static bool EmitValueSwitchIndex(Encoder& e, WJBackend& be, uint32_t dl,
                                 int32_t defaultIdx) {
  auto pushTag = [&]() -> bool {
    return GetLocal(e, dl) && e.writeOp(Op::I64Const) && e.writeVarS64(32) &&
           e.writeOp(Op::I64ShrU) && e.writeOp(Op::I32WrapI64);
  };
  if (!pushTag() || !e.writeOp(Op::I32Const) ||
      !e.writeVarS32(int32_t(TagWord(JSVAL_TYPE_INT32))) ||
      !e.writeOp(Op::I32Eq))
    return false;
  if (!e.writeOp(Op::If) || !e.writeFixedU8(uint8_t(TypeCode::I32)))
    return false;
  if (!GetLocal(e, dl) || !e.writeOp(Op::I32WrapI64)) return false;  // payload
  if (!e.writeOp(Op::Else)) return false;
  if (!pushTag() || !e.writeOp(Op::I32Const) ||
      !e.writeVarS32(int32_t(uint32_t(JSVAL_TAG_CLEAR))) ||
      !e.writeOp(Op::I32LeU))
    return false;  // tag u<= JSVAL_TAG_CLEAR => f64 (nunbox32)
  if (!e.writeOp(Op::If) || !e.writeFixedU8(uint8_t(TypeCode::I32)))
    return false;
  if (!GetLocal(e, dl) || !e.writeOp(Op::F64ReinterpretI64) ||
      !e.writeOp(MiscOp::I32TruncSatF64S) || !e.writeOp(Op::LocalTee) ||
      !e.writeVarU32(be.propTaggedLocal))
    return false;  // i = trunc_sat(d)
  if (!e.writeOp(Op::F64ConvertI32S) || !GetLocal(e, dl) ||
      !e.writeOp(Op::F64ReinterpretI64) || !e.writeOp(Op::F64Eq))
    return false;  // (double)i == d
  if (!e.writeOp(Op::If) || !e.writeFixedU8(uint8_t(TypeCode::I32)))
    return false;
  if (!GetLocal(e, be.propTaggedLocal)) return false;
  if (!e.writeOp(Op::Else)) return false;
  if (!e.writeOp(Op::I32Const) || !e.writeVarS32(defaultIdx)) return false;
  if (!e.writeOp(Op::End)) return false;
  if (!e.writeOp(Op::Else)) return false;
  if (!e.writeOp(Op::I32Const) || !e.writeVarS32(defaultIdx)) return false;
  if (!e.writeOp(Op::End)) return false;
  return e.writeOp(Op::End);
}

static bool EmitTerminator(Encoder& e, WJBackend& be,
                           std::unordered_map<MBasicBlock*, uint32_t>& blockIdx,
                           uint32_t bidLocal, uint32_t loopDepth, MBasicBlock* b) {
  MControlInstruction* t = b->lastIns();
  if (t->resumePoint()) be.curRp = t->resumePoint();  // else keep body last RP
  if (t->isReturn()) {
    return EmitReturn(e, be, t->getOperand(0));
  }
  if (t->isUnreachable() || t->isThrow()) {
    // A reachable Throw/Unreachable means deopt to PBL to run real semantics.
    return EmitDeopt(e, be);
  }
  if (t->isGoto()) {
    MBasicBlock* s = t->toGoto()->target();
    if (!EmitEdgeCopies(e, be, b, s)) return false;
    return GotoBlock(e, bidLocal, loopDepth, blockIdx[s]);
  }
  if (t->isTest()) {
    MTest* test = t->toTest();
    MDefinition* cond = test->getOperand(0);
    MBasicBlock* tb = test->ifTrue();
    MBasicBlock* fb = test->ifFalse();
    if (WJValType(cond->type()) == uint8_t(TypeCode::I32)) {
      if (!GetOp(e, be, cond)) return false;
    } else if (cond->type() == MIRType::Value) {
      int32_t cl = be.local(cond);
      if (cl < 0) return false;
      if (!EmitValueTruthy(e, be, uint32_t(cl))) return false;
    } else {
      return false;
    }
    if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
    if (!EmitEdgeCopies(e, be, b, tb)) return false;
    if (!GotoBlock(e, bidLocal, loopDepth + 1, blockIdx[tb])) return false;
    if (!e.writeOp(Op::Else)) return false;
    if (!EmitEdgeCopies(e, be, b, fb)) return false;
    if (!GotoBlock(e, bidLocal, loopDepth + 1, blockIdx[fb])) return false;
    if (!e.writeOp(Op::End)) return false;
    return e.writeOp(Op::Unreachable);
  }
  if (t->isTableSwitch()) {
    // switch(value): case k (k in [0,numCases)) -> case block; else -> default.
    // Lower to a wasm br_table over (value - low): open numCases+1 nested blocks
    // (default outermost, case0 innermost), br_table inside the innermost, then
    // close each block and emit that case's edge-copies + GotoBlock($L). br_table
    // treats the index as UNSIGNED, so value<low wraps huge -> hits the default.
    // (gbemu executeIteration's `switch(this.IRQEnableDelay)` -- was bailing the
    // whole hot CPU loop to PBL on the unhandled terminator.)
    // Boxed-Value discriminants go through EmitValueSwitchIndex (int32 payload,
    // exact-int32 double incl -0, else default).
    MTableSwitch* ts = t->toTableSwitch();
    uint32_t n = uint32_t(ts->numCases());
    for (uint32_t i = 0; i <= n; i++) {
      if (!e.writeOp(Op::Block) || !e.writeFixedU8(0x40)) return false;
    }
    MDefinition* disc = ts->getOperand(0);
    // The br_table index is `value - low` as i32. If the discriminant is an i32
    // node, read it directly. If it is a boxed Value (i64) -- e.g. gbemu
    // executeIteration's `switch(this.IRQEnableDelay)` where `this` is polymorphic
    // so the field loads boxed -- emitting i32.sub on the i64 is INVALID wasm and
    // V8 rejects the WHOLE module (the entire CPU loop fell to PBL). Extract an i32
    // index from the Value: an INT32-tagged value uses its payload; anything else
    // (double / non-number) is routed to the default (push low+n so the `- low`
    // below yields n). JS `switch` is `===`, so only an int32 matches an integer
    // case label here (an integer-valued double discriminant would go to default --
    // acceptable: switch discriminants reaching this path are int32 in practice).
    if (WJValType(disc->type()) == uint8_t(TypeCode::I32)) {
      if (!GetOp(e, be, disc)) return false;  // i32 value
    } else if (disc->type() == MIRType::Value) {
      int32_t dl = be.local(disc);
      if (dl < 0) { js::wasm::gWJBailReason = "tableswitch-value-nolocal"; return false; }
      if (!EmitValueSwitchIndex(e, be, uint32_t(dl),
                                int32_t(ts->low()) + int32_t(n)))
        return false;
    } else {
      js::wasm::gWJBailReason = "tableswitch-nonI32-disc";
      return false;
    }
    if (ts->low() != 0) {
      if (!e.writeOp(Op::I32Const) || !e.writeVarS32(ts->low()) ||
          !e.writeOp(Op::I32Sub))
        return false;
    }
    if (!e.writeOp(Op::BrTable) || !e.writeVarU32(n)) return false;
    for (uint32_t i = 0; i < n; i++) {
      if (!e.writeVarU32(i)) return false;  // case i -> label $case_i (depth i)
    }
    if (!e.writeVarU32(n)) return false;  // default -> label $default (depth n)
    for (uint32_t i = 0; i < n; i++) {
      if (!e.writeOp(Op::End)) return false;  // close $case_i; now in $case_{i+1}
      MBasicBlock* tgt = ts->getCase(i);
      if (!EmitEdgeCopies(e, be, b, tgt)) return false;
      // n+1 blocks opened, i+1 closed -> n-i still open between here and $L.
      if (!GotoBlock(e, bidLocal, loopDepth + (n - i), blockIdx[tgt]))
        return false;
    }
    if (!e.writeOp(Op::End)) return false;  // close $default
    MBasicBlock* dflt = ts->getDefault();
    if (!EmitEdgeCopies(e, be, b, dflt)) return false;
    return GotoBlock(e, bidLocal, loopDepth, blockIdx[dflt]);
  }
  return false;
}

// ---------------------------------------------------------------------------
// Structured-control-flow emitter ("relooper"). Reconstructs nested
// loop/block/if from the reducible MIR CFG so V8 sees real loops -- letting it
// register-allocate across iterations and keep the values OptimizeMIR's LICM
// already hoisted, which the flat br_table dispatch loop defeats. Each block
// gets a wasm `block` scope ending at it (so forward branches become `br`);
// each loop header also gets a `loop` scope (back-edges `br` to it). Begins are
// pulled back until all scopes nest. Returns false either before emitting
// (sets *started=false -> caller uses the dispatch loop) or mid-emit on an
// unsupported node (*started=true -> caller bails the whole compile).
static bool WJEmitStructured(Encoder& e, WJBackend& be,
                             std::vector<MBasicBlock*>& blocks,
                             std::unordered_map<MBasicBlock*, uint32_t>& blockIdx,
                             bool* started) {
  *started = false;
  uint32_t n = uint32_t(blocks.size());
  std::vector<int32_t> idomIdx(n, -1);
  std::vector<int32_t> loopEnd(n, -1);  // loopEnd[h] = end pos if h is a header
  for (uint32_t i = 0; i < n; i++) {
    MBasicBlock* d = blocks[i]->immediateDominator();
    if (d) {
      auto it = blockIdx.find(d);
      if (it != blockIdx.end()) idomIdx[i] = int32_t(it->second);
    }
    if (blocks[i]->isLoopHeader()) {
      MBasicBlock* bk = blocks[i]->backedge();
      if (!bk) return false;
      auto bit = blockIdx.find(bk);
      if (bit == blockIdx.end()) return false;
      loopEnd[i] = int32_t(bit->second) + 1;
    }
  }
  std::vector<uint32_t> bbegin(n, 0);  // block-scope begin position per block
  for (uint32_t t = 1; t < n; t++) {
    if (idomIdx[t] < 0) {
      js::wasm::gWJBailReason = "no-idom";
      return false;  // unreachable/no idom: bail to dispatch
    }
    bbegin[t] = uint32_t(idomIdx[t]);
  }
  // Nesting fix-up: two scopes [a,b) [c,d) improperly overlap iff a<c<b<d (or
  // symmetric). Pull a block scope's begin back to the other's begin until none
  // overlap. Loop begins are fixed (must sit at the header).
  auto improper = [](uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
    return (a < c && c < b && b < d) || (c < a && a < d && d < b);
  };
  for (uint32_t iter = 0; iter <= n; iter++) {
    bool changed = false;
    for (uint32_t t = 1; t < n; t++) {
      for (uint32_t u = 1; u < n; u++) {
        if (u != t && improper(bbegin[u], u, bbegin[t], t)) {
          // Resolve by GROWING both scopes to the earlier start (never SHRINK a
          // scope forward -- that strands forward branches into it, e.g. splay_'s
          // block 4 -> merge-block 11 when 11's begin was pushed past 4). Pulling
          // both to min(begins) gives them a common start => properly nested by end.
          uint32_t m = bbegin[u] < bbegin[t] ? bbegin[u] : bbegin[t];
          if (bbegin[t] != m || bbegin[u] != m) {
            bbegin[t] = m;
            bbegin[u] = m;
            changed = true;
          }
        }
      }
      for (uint32_t h = 0; h < n; h++) {
        if (loopEnd[h] >= 0 &&
            improper(h, uint32_t(loopEnd[h]), bbegin[t], t)) {
          // Only GROW t's scope (pull begin back to the loop header). If that would
          // push it FORWARD (t starts before the loop, ends inside -- crosses the
          // boundary), leave it: the pre-validation bails -> dispatch loop (safe),
          // never a shrunk scope that strands a forward branch.
          if (uint32_t(h) < bbegin[t]) {
            bbegin[t] = h;
            changed = true;
          }
        }
      }
    }
    if (!changed) break;
  }
  struct Rec {
    uint32_t end;
    bool isLoop;
    uint32_t header;
  };
  std::vector<std::vector<Rec>> beginsAt(n + 1), endsAt(n + 1);
  for (uint32_t t = 1; t < n; t++) {
    Rec r{t, false, 0};
    beginsAt[bbegin[t]].push_back(r);
    endsAt[t].push_back(r);
  }
  for (uint32_t h = 0; h < n; h++) {
    if (loopEnd[h] < 0) continue;
    Rec r{uint32_t(loopEnd[h]), true, h};
    beginsAt[h].push_back(r);
    endsAt[uint32_t(loopEnd[h])].push_back(r);
  }
  // Open outermost (largest end) first; close innermost (smallest end) first.
  for (uint32_t p = 0; p <= n; p++) {
    std::sort(beginsAt[p].begin(), beginsAt[p].end(),
              [](const Rec& a, const Rec& b) { return a.end > b.end; });
    std::sort(endsAt[p].begin(), endsAt[p].end(),
              [](const Rec& a, const Rec& b) { return a.end < b.end; });
  }

  std::vector<Rec> stk;
  auto depthOfBlock = [&](uint32_t tIdx) -> int32_t {
    for (int32_t k = int32_t(stk.size()) - 1; k >= 0; k--)
      if (!stk[k].isLoop && stk[k].end == tIdx) return int32_t(stk.size()) - 1 - k;
    return -1;
  };
  auto depthOfLoop = [&](uint32_t hIdx) -> int32_t {
    for (int32_t k = int32_t(stk.size()) - 1; k >= 0; k--)
      if (stk[k].isLoop && stk[k].header == hIdx) return int32_t(stk.size()) - 1 - k;
    return -1;
  };
  auto brTo = [&](MBasicBlock* s, uint32_t pos, uint32_t extra) -> int32_t {
    uint32_t sIdx = blockIdx[s];
    int32_t d = (sIdx > pos) ? depthOfBlock(sIdx) : depthOfLoop(sIdx);
    return d < 0 ? -1 : int32_t(uint32_t(d) + extra);
  };
  // PRE-VALIDATE (no bytes emitted, *started still false): dry-run the same
  // scope-stack walk and confirm every terminator branch target resolves to an
  // open scope. If a branch can't (irreducible/forward edge the structuring
  // can't express -> brTo == -1), return false here so the caller falls back to
  // the dispatch loop. Without this, such CFGs strand mid-emit (*started=true ->
  // whole-compile failure), and because a failed compile is retried on the next
  // warm-up tick, the function recompile-storms (deltablue hung under reloop).
  // This is the gate-widening enabler: with depth bails caught up-front, call-
  // heavy shallow-loop functions can safely reloop (crypto/splay/richards win).
  for (uint32_t pos = 0; pos <= n; pos++) {
    for (size_t i = 0; i < endsAt[pos].size(); i++) {
      if (stk.empty()) { js::wasm::gWJBailReason = "reloop-pv-underflow"; return false; }
      stk.pop_back();
    }
    for (Rec& r : beginsAt[pos]) stk.push_back(r);
    if (pos == n) break;
    MBasicBlock* b = blocks[pos];
    if (b->alwaysBails() || WJBlockForceDeopt(be, b)) continue;
    MControlInstruction* t = b->lastIns();
    if (t->isGoto()) {
      if (brTo(t->toGoto()->target(), pos, 0) < 0) {
        js::wasm::gWJBailReason = "reloop-goto-depth"; return false;
      }
    } else if (t->isTest()) {
      int32_t dvt = brTo(t->toTest()->ifTrue(), pos, 1);
      int32_t dvf = brTo(t->toTest()->ifFalse(), pos, 1);
      if (dvt < 0 || dvf < 0) {
        if (WJGETENV("GECKO_WJ_RELOOPDBG")) {
          uint32_t ti = blockIdx[t->toTest()->ifTrue()];
          uint32_t fi = blockIdx[t->toTest()->ifFalse()];
          fprintf(stderr,
                  "[reloopdbg] test-depth FAIL pos=%u (n=%u): ifTrue=%u(fwd=%d dvt=%d bbegin=%u) "
                  "ifFalse=%u(fwd=%d dvf=%d bbegin=%u) stk=%zu\n",
                  pos, n, ti, int(ti > pos), dvt, ti < n ? bbegin[ti] : 0,
                  fi, int(fi > pos), dvf, fi < n ? bbegin[fi] : 0, stk.size());
        }
        js::wasm::gWJBailReason = "reloop-test-depth"; return false;
      }
    } else if (t->isTableSwitch()) {
      MTableSwitch* ts = t->toTableSwitch();
      for (uint32_t i = 0; i < ts->numCases(); i++) {
        if (brTo(ts->getCase(i), pos, 0) < 0) {
          js::wasm::gWJBailReason = "reloop-tableswitch-depth"; return false;
        }
      }
      if (brTo(ts->getDefault(), pos, 0) < 0) {
        js::wasm::gWJBailReason = "reloop-tableswitch-depth"; return false;
      }
    }
  }
  if (!stk.empty()) { js::wasm::gWJBailReason = "reloop-pv-residual"; return false; }
  stk.clear();

  *started = true;
  const uint8_t kVoid = 0x40;
  // Out-of-line deopt: wrap the whole structured body in `block $D`. A guard miss
  // inside a hot block body records its resume site and brs to $D (depth =
  // open-relooper-scopes + the guard `if`), where the post-body dispatcher runs the
  // matching resume -- keeping the hot block bodies tiny (no inline spill bloat).
  // This is what makes the relooper viable for call-heavy functions (inline deopt
  // at every guard regressed richards under reloop). NORELOOPOOL/NOOOL revert.
  // OOL deopt in the relooper is opt-in: it adds the $D wrapper + dispatcher, which
  // is a net win only for call-heavy / guard-heavy functions (where inline deopt at
  // every guard bloats the body). For the small tight loops the gate currently
  // reloops (crypto/splay), inline deopt is leaner -- so default to it and enable
  // OOL only under GECKO_WJ_RELOOPOOL (used when the gate is widened to call-heavy
  // functions). NOOOL/NORELOOPOOL force it off.
  static int noOol = WJGETENV("GECKO_WJ_NOOOL") ? 1 : 0;
  static int wantReloopOol = WJGETENV("GECKO_WJ_RELOOPOOL") ? 1 : 0;
  // NOTE: auto-enabling OOL for call-heavy reloop'd functions recovered the
  // richards/navier regression from the widened gate BUT exposed a correctness
  // bug in the OOL deopt-resume path for crypto (ERR=Crypto). So OOL stays
  // opt-in (GECKO_WJ_RELOOPOOL). The widened-gate path (GECKO_WJ_RELOOPCALLS)
  // therefore uses INLINE deopt -- correct + fast for crypto (+61%) but a net
  // loss for richards/navier; making it default-safe needs either the OOL
  // resume bug fixed or a reliable per-function inline-vs-OOL discriminator.
  // Auto-OOL for call-heavy reloop'd functions: inline deopt at every guard
  // bloats call-heavy bodies (regressed richards/navier under the widened gate);
  // OOL moves the spill+resume out of line. This was the documented
  // "reliable per-function inline-vs-OOL discriminator" -- safe now that the
  // OOL deopt-resume miscompile (ERR=Crypto) is fixed (it was the WJCollectRoots
  // id-order bug, [[wjcollectroots-listorder-fix]]). Tight loops (no calls) keep
  // inline deopt (leaner). GECKO_WJ_NOOOL forces inline everywhere.
  be.oolDeopt = (wantReloopOol || be.reloopHasCalls) && !noOol;
  be.reloopOol = be.oolDeopt;
  if (be.reloopOol) {
    if (!e.writeOp(Op::Block) || !e.writeFixedU8(kVoid)) return false;  // $D
  }
  for (uint32_t pos = 0; pos <= n; pos++) {
    for (size_t i = 0; i < endsAt[pos].size(); i++) {
      if (stk.empty()) return false;
      stk.pop_back();
      if (!e.writeOp(Op::End)) return false;
    }
    for (Rec& r : beginsAt[pos]) {
      if (!e.writeOp(r.isLoop ? Op::Loop : Op::Block) || !e.writeFixedU8(kVoid))
        return false;
      stk.push_back(r);
    }
    if (pos == n) break;
    MBasicBlock* b = blocks[pos];
    be.reloopScopeDepth = uint32_t(stk.size());  // scopes enclosing this body (excl $D)
    be.inDispatchBody = be.reloopOol;             // in-body guards go out-of-line to $D
    be.inReloopBody = true;                       // new deopt sites here would hit #16
    bool bodyOk = EmitBlockBody(e, be, b);
    be.inReloopBody = false;
    be.inDispatchBody = false;                    // terminator deopts stay inline
    if (!bodyOk) return false;
    if (b->alwaysBails() || WJBlockForceDeopt(be, b)) continue;  // emitted its own deopt; no terminator
    MControlInstruction* t = b->lastIns();
    if (t->resumePoint()) be.curRp = t->resumePoint();  // else keep body last RP
    if (t->isReturn()) {
      if (!EmitReturn(e, be, t->getOperand(0))) return false;
    } else if (t->isUnreachable() || t->isThrow()) {
      if (!EmitDeopt(e, be)) return false;
    } else if (t->isGoto()) {
      MBasicBlock* s = t->toGoto()->target();
      if (!EmitEdgeCopies(e, be, b, s)) return false;
      int32_t d = brTo(s, pos, 0);
      if (d < 0) { js::wasm::gWJBailReason = "reloop-goto-depth"; return false; }
      if (!e.writeOp(Op::Br) || !e.writeVarU32(uint32_t(d))) return false;
    } else if (t->isTest()) {
      MTest* test = t->toTest();
      MDefinition* cond = test->getOperand(0);
      MBasicBlock* tb = test->ifTrue();
      MBasicBlock* fb = test->ifFalse();
      // Push the i32 condition. I32-typed conditions are read directly; Value-typed
      // ones go through EmitValueTruthy (which emits its own balanced if-chain and
      // leaves a single i32, possibly an inline deopt for string/symbol/bigint).
      // Terminator deopts are inline (inDispatchBody was cleared above), so the
      // EmitValueTruthy deopt path's br-depth is irrelevant here.
      if (WJValType(cond->type()) == uint8_t(TypeCode::I32)) {
        if (!GetOp(e, be, cond)) return false;
      } else if (cond->type() == MIRType::Value) {
        int32_t cl = be.local(cond);
        if (cl < 0) { js::wasm::gWJBailReason = "reloop-test-cond-local"; return false; }
        if (!EmitValueTruthy(e, be, uint32_t(cl))) return false;
      } else {
        js::wasm::gWJBailReason = "reloop-test-cond";
        return false;
      }
      if (!e.writeOp(Op::If) || !e.writeFixedU8(kVoid)) return false;
      if (!EmitEdgeCopies(e, be, b, tb)) return false;
      int32_t dt = brTo(tb, pos, 1);
      if (dt < 0) { js::wasm::gWJBailReason = "reloop-test-depth-t"; return false; }
      if (!e.writeOp(Op::Br) || !e.writeVarU32(uint32_t(dt))) return false;
      if (!e.writeOp(Op::Else)) return false;
      if (!EmitEdgeCopies(e, be, b, fb)) return false;
      int32_t df = brTo(fb, pos, 1);
      if (df < 0) { js::wasm::gWJBailReason = "reloop-test-depth-f"; return false; }
      if (!e.writeOp(Op::Br) || !e.writeVarU32(uint32_t(df))) return false;
      if (!e.writeOp(Op::End)) return false;
    } else if (t->isTableSwitch()) {
      // switch(value) in the relooper: open nc+1 wasm blocks (default outermost,
      // case0 innermost), br_table inside the innermost, then close each block and
      // br to that case target's relooper scope (depth = brTo(tgt) + the switch
      // blocks still open). Mirrors the dispatch-path lowering (EmitTerminator)
      // but branches to structured scopes instead of GotoBlock($L). This is what
      // lets gbemu's 335KB executeIteration (switch on IRQEnableDelay) reloop
      // instead of falling to the dispatch loop.
      MTableSwitch* ts = t->toTableSwitch();
      uint32_t nc = uint32_t(ts->numCases());
      for (uint32_t i = 0; i <= nc; i++) {
        if (!e.writeOp(Op::Block) || !e.writeFixedU8(kVoid)) return false;
      }
      MDefinition* disc = ts->getOperand(0);
      if (WJValType(disc->type()) == uint8_t(TypeCode::I32)) {
        if (!GetOp(e, be, disc)) return false;
      } else if (disc->type() == MIRType::Value) {
        int32_t dl = be.local(disc);
        if (dl < 0) {
          js::wasm::gWJBailReason = "reloop-tableswitch-nolocal";
          return false;
        }
        if (!EmitValueSwitchIndex(e, be, uint32_t(dl),
                                  int32_t(ts->low()) + int32_t(nc)))
          return false;
      } else {
        js::wasm::gWJBailReason = "reloop-tableswitch-disc";
        return false;
      }
      if (ts->low() != 0) {
        if (!e.writeOp(Op::I32Const) || !e.writeVarS32(ts->low()) ||
            !e.writeOp(Op::I32Sub))
          return false;
      }
      if (!e.writeOp(Op::BrTable) || !e.writeVarU32(nc)) return false;
      for (uint32_t i = 0; i < nc; i++) {
        if (!e.writeVarU32(i)) return false;
      }
      if (!e.writeVarU32(nc)) return false;  // default
      for (uint32_t i = 0; i < nc; i++) {
        if (!e.writeOp(Op::End)) return false;  // close $case_i
        MBasicBlock* tgt = ts->getCase(i);
        if (!EmitEdgeCopies(e, be, b, tgt)) return false;
        // nc+1 opened, i+1 closed -> nc-i switch-blocks still open here.
        int32_t d = brTo(tgt, pos, nc - i);
        if (d < 0) {
          js::wasm::gWJBailReason = "reloop-tableswitch-case-depth";
          return false;
        }
        if (!e.writeOp(Op::Br) || !e.writeVarU32(uint32_t(d))) return false;
      }
      if (!e.writeOp(Op::End)) return false;  // close $default
      MBasicBlock* dflt = ts->getDefault();
      if (!EmitEdgeCopies(e, be, b, dflt)) return false;
      int32_t dd = brTo(dflt, pos, 0);
      if (dd < 0) {
        js::wasm::gWJBailReason = "reloop-tableswitch-default-depth";
        return false;
      }
      if (!e.writeOp(Op::Br) || !e.writeVarU32(uint32_t(dd))) return false;
    } else {
      js::wasm::gWJBailReason = "reloop-terminator";
      return false;
    }
  }
  if (!stk.empty()) return false;
  if (!e.writeOp(Op::Unreachable)) return false;
  if (be.reloopOol) {
    if (!e.writeOp(Op::End)) return false;  // end $D
    be.inDispatchBody = false;
    for (uint32_t s = 0; s < be.deoptSites.size(); s++) {
      if (!GetLocal(e, be.deoptSiteLocal) || !e.writeOp(Op::I32Const) ||
          !e.writeVarS32(int32_t(s)) || !e.writeOp(Op::I32Eq) ||
          !e.writeOp(Op::If) || !e.writeFixedU8(0x40))
        return false;
      be.curRp = be.deoptSites[s].first;
      be.curOp = be.deoptSites[s].second;
      if (!EmitDeoptResumeInline(e, be)) {
        if (!js::wasm::gWJBailReason ||
            !strstr(js::wasm::gWJBailReason, "resume"))
          js::wasm::gWJBailReason = "reloop-deopt-resume";
        return false;
      }
      if (!e.writeOp(Op::End)) return false;
    }
    if (!e.writeOp(Op::Unreachable)) return false;
  }
  return e.writeOp(Op::End);  // function body end
}

// === Int32-mis-speculation widening (navier/raytrace float arrays) ===========
// Warp speculates a float array's element loads + their arithmetic as Int32
// because the arrays warm up all-zero (int32) and integer-valued doubles like
// dt*0=0.0 canonicalize to int32. At steady state the doubles DEOPT every load/
// ToNumberInt32 -> the whole function runs in PBL (no Baseline tier). The IC
// never widens (PBL's inline fast paths bypass the IC fallback). So we re-type
// here, Ion-style: partition Int32 nodes into INDEX-locked (backward reachable
// from an array index/length -- must stay Int32) vs VALUE (widen to Double, with
// the no-deopt number-unbox). Abort (leave MIR unchanged) on anything unhandled,
// so a function we can't safely widen just keeps its old (deopting) codegen --
// never a wrong result. Gated by gWJForceNumberArith / GECKO_WJ_NUMARITH.
static bool WJWidenInt32Values(MIRGenerator& mir, MIRGraph& graph) {
  using jit::MDefinition;
  using jit::MInstruction;
  using Op_ = MDefinition::Opcode;
  jit::TempAllocator& alloc = graph.alloc();

  auto isI32 = [](MDefinition* d) { return d && d->type() == MIRType::Int32; };

  // 1. INDEX-locked set: backward closure from array index/length operands.
  std::unordered_set<MDefinition*> locked;
  std::vector<MDefinition*> wl;
  auto lock = [&](MDefinition* d) {
    if (isI32(d) && locked.insert(d).second) wl.push_back(d);
  };
  for (auto bIt = graph.rpoBegin(); bIt != graph.rpoEnd(); bIt++) {
    for (auto it = bIt->begin(); it != bIt->end(); it++) {
      MInstruction* ins = *it;
      switch (ins->op()) {
        case Op_::LoadElement:
        case Op_::LoadElementAndUnbox:
        case Op_::LoadElementHole:
          lock(ins->getOperand(1));
          break;
        case Op_::StoreElement:
        case Op_::StoreElementHole:
          lock(ins->getOperand(1));
          break;
        case Op_::BoundsCheck:
          lock(ins->getOperand(0));
          lock(ins->getOperand(1));
          break;
        case Op_::BoundsCheckLower:
        case Op_::SpectreMaskIndex:
          for (size_t i = 0; i < ins->numOperands(); i++)
            lock(ins->getOperand(i));
          break;
        case Op_::InitializedLength:
        case Op_::ArrayLength:
          lock(ins);  // an Int32 length producer -- never a float value
          break;
        case Op_::SetInitializedLength:
          lock(ins->getOperand(1));
          break;
        // Int32-wraparound / truncation sinks: their operands MUST stay Int32 so
        // widening never feeds a double into a bitwise/truncating op (which would
        // lose >2^53 precision). Backward-closing from here keeps int-only code
        // (crypto bignums) fully Int32 -- correct AND no boundary deopts.
        case Op_::BitOr:
        case Op_::BitAnd:
        case Op_::BitXor:
        case Op_::Lsh:
        case Op_::Rsh:
        case Op_::Ursh:
        case Op_::BitNot:
        case Op_::TruncateToInt32:
        case Op_::WasmTruncateToInt32:
        case Op_::SignExtendInt32:
        // Typed-array / DataView / push: index and (int-typed) value operands
        // must stay Int32. Conservatively lock all operands (over-locking only
        // reduces widening, never breaks correctness).
        case Op_::LoadUnboxedScalar:
        case Op_::StoreUnboxedScalar:
        case Op_::LoadDataViewElement:
        case Op_::StoreDataViewElement:
        case Op_::LoadTypedArrayElementHole:
        case Op_::StoreTypedArrayElementHole:
        case Op_::ArrayPush:
        case Op_::GuardInt32IsNonNegative:
          for (size_t i = 0; i < ins->numOperands(); i++)
            lock(ins->getOperand(i));
          break;
        default:
          break;
      }
    }
  }
  while (!wl.empty()) {
    MDefinition* d = wl.back();
    wl.pop_back();
    for (size_t i = 0; i < d->numOperands(); i++) lock(d->getOperand(i));
  }

  // 2. VALUE-widen candidates: Int32 numeric value nodes not index-locked.
  std::unordered_set<MDefinition*> widen;
  auto widenable = [&](MDefinition* d) -> bool {
    if (!isI32(d) || locked.count(d)) return false;
    switch (d->op()) {
      case Op_::LoadElement:
      case Op_::LoadElementAndUnbox:
      case Op_::Add:
      case Op_::Sub:
      case Op_::Mul:
      case Op_::Div:
      case Op_::Mod:
      case Op_::Unbox:           // param/value unbox (e.g. lin_solve `a`)
      case Op_::ToNumberInt32:   // bridge double->int; bypass on widen
      case Op_::Phi:             // value accumulator carried across iterations
        return true;
      default:
        return false;
    }
  };
  std::unordered_set<MDefinition*> cand;  // widenable Int32 nodes (not locked)
  for (auto bIt = graph.rpoBegin(); bIt != graph.rpoEnd(); bIt++)
    for (auto it = bIt->begin(); it != bIt->end(); it++)
      if (widenable(*it)) cand.insert(*it);
  if (cand.empty()) return false;

  // FLOAT-TAINT (the principled, safe criterion -- not a heuristic): only widen
  // an Int32 node that PARTICIPATES in floating-point arithmetic, i.e. is
  // connected through arith/phi/conversion edges to a Double/Float value. A
  // pure-integer chain (richards counters, crypto bignums) touches no Double ->
  // never tainted -> never widened -> its Int32 semantics are preserved exactly.
  // Taint flows both ways across arith: a Double operand taints the arith node
  // (and thus its other Int32 operands), and a tainted arith taints its Int32
  // load/operand producers. This is exactly "this int is really a float".
  auto isFP = [](MDefinition* d) {
    return d->type() == MIRType::Double || d->type() == MIRType::Float32;
  };
  auto isNumericOp = [&](MDefinition* d) {
    switch (d->op()) {
      case Op_::Add: case Op_::Sub: case Op_::Mul: case Op_::Div:
      case Op_::Mod: case Op_::Phi: case Op_::ToNumberInt32:
      case Op_::ToDouble: case Op_::ToFloat32:  // Int32 sum -> ToDouble -> f64 mul
        return true;
      default:
        return false;
    }
  };
  std::unordered_set<MDefinition*> tainted;
  for (auto bIt = graph.rpoBegin(); bIt != graph.rpoEnd(); bIt++)
    for (auto it = bIt->begin(); it != bIt->end(); it++)
      if (isFP(*it)) tainted.insert(*it);  // seed: every Double/Float node
  // CMP-SEED: also seed an Int32 value-unbox whose uses are ALL Double-safe sinks
  // (number compares / Double arith), even with no FP node nearby. These are number
  // properties Warp mis-typed Int32 because it observed an int32 0 first (raytrace
  // material.reflection = 0 for matte, 0.5 for shiny) -> the strict Int32 unbox
  // deopt-STORMS on the double, dragging the whole hot function (rayTrace) to PBL.
  // Widening to a number (Double) unbox is sound -- 0 and 0.5 compare/arith
  // identically as doubles -- and kills the storm. Restricting the seed to nodes
  // whose every use is Double-safe. DEFAULT OFF (opt-in GECKO_WJ_WIDENCMP):
  // FUNDAMENTALLY too broad -- pure-integer code (richards scheduler counters,
  // navier loop/index values) also feeds only compares + int arithmetic, which
  // ARE "Double-safe" sinks, so this widens genuine Int32 to Double (richards
  // 120->13, navier 3900->26, deltablue hang). There is no STATIC signal that
  // separates "an Int32 that's really a float at runtime" (raytrace
  // material.reflection = 0 | 0.5) from "a genuine Int32 that happens to feed
  // number ops" -- only float-taint (an actual FP connection) is sound. The
  // reflection deopt storm needs an IC-level number-speculation fix, not widening.
  if (WJGETENV("GECKO_WJ_WIDENCMP")) {
    auto allUsesDoubleSafe = [&](MDefinition* d) -> bool {
      bool any = false;
      for (auto u = d->usesBegin(); u != d->usesEnd(); u++) {
        if (!u->consumer()->isDefinition()) continue;  // resume pt: boxed by type
        MDefinition* user = u->consumer()->toDefinition();
        any = true;
        switch (user->op()) {
          case Op_::Compare:
          case Op_::Add: case Op_::Sub: case Op_::Mul: case Op_::Div: case Op_::Mod:
          case Op_::ToDouble: case Op_::ToFloat32:
            if (user->type() == MIRType::Int32) return false;  // int sink -> unsafe
            break;
          default:
            return false;
        }
      }
      return any;
    };
    for (MDefinition* c : cand)
      if ((c->op() == Op_::Unbox || c->op() == Op_::LoadElementAndUnbox ||
           c->op() == Op_::LoadFixedSlotAndUnbox ||
           c->op() == Op_::LoadDynamicSlotAndUnbox) &&
          allUsesDoubleSafe(c))
        tainted.insert(c);
  }
  bool changed = true;
  while (changed) {
    changed = false;
    for (MDefinition* c : cand) {
      if (tainted.count(c)) continue;
      bool adj = false;
      // a numeric node is tainted if any operand is tainted (Double feeds it)
      if (isNumericOp(c))
        for (size_t i = 0; i < c->numOperands() && !adj; i++)
          if (tainted.count(c->getOperand(i))) adj = true;
      // any node is tainted if it FEEDS a tainted numeric op (load -> float arith)
      for (auto u = c->usesBegin(); u != c->usesEnd() && !adj; u++) {
        if (!u->consumer()->isDefinition()) continue;
        MDefinition* user = u->consumer()->toDefinition();
        if (tainted.count(user) && isNumericOp(user)) adj = true;
      }
      if (adj) {
        tainted.insert(c);
        changed = true;
      }
    }
  }
  for (MDefinition* c : cand)
    if (tainted.count(c)) widen.insert(c);
  if (widen.empty()) return false;
  // SAFE: int-wraparound/truncation sinks (bitops, ToInt32, indices, typed
  // arrays) are backward-locked, so a widened Double never feeds a truncating op
  // (no >2^53 precision loss). Int32 operands of widened arith -> MToDouble
  // (exact: int32 < 2^53). Any Int32 USE of a widened value -> MToNumberInt32
  // (step d), which DEOPTS on a non-exact-int32 -> correctness preserved even for
  // a missed sink (worst case a slow deopt, never a wrong value).

  static int wdbg = WJGETENV("GECKO_WJ_WIDEN") ? 1 : 0;
  // Only ARITH and Phi widen nodes have numeric VALUE operands that must become
  // Double. Loads/Unbox have STRUCTURAL operands (array ptr, Int32 index, boxed
  // value) that must stay as-is -- never convert a load's index!
  auto hasNumericOperands = [](MDefinition* d) {
    switch (d->op()) {
      case Op_::Add: case Op_::Sub: case Op_::Mul: case Op_::Div:
      case Op_::Mod: case Op_::Phi:
        return true;
      default:
        return false;
    }
  };

  // 3b. PURITY CHECK (safety keystone): a widened value (now a raw Double) must
  //    flow ONLY into a STRICT WHITELIST of Double-safe sinks. Anything else --
  //    Box / call arg / object-slot store / Return / a guard / an Int32 op --
  //    could later read the value as a boxed Value/object (mandreel+splay
  //    `Value::toObject` OOB crash) or need a deopt-prone MToNumberInt32. If ANY
  //    widen node has a non-whitelisted use, ABORT the whole function (leave MIR
  //    untouched -> keeps its old, correct codegen). navier's value chains feed
  //    only element stores + double arith + compares -> never abort.
  for (MDefinition* w : widen) {
    for (auto u = w->usesBegin(); u != w->usesEnd(); u++) {
      if (!u->consumer()->isDefinition()) continue;  // resume point: boxed-by-type
      MDefinition* user = u->consumer()->toDefinition();
      if (widen.count(user)) continue;
      bool ok = false;
      switch (user->op()) {
        // Double-accepting numeric sinks. Int32-typed arith is rejected below
        // (it would need an MToNumberInt32 deopt boundary).
        case Op_::Add: case Op_::Sub: case Op_::Mul: case Op_::Div:
        case Op_::Mod:
        case Op_::ToDouble: case Op_::ToFloat32: case Op_::ToNumberInt32:
        case Op_::Compare:
        case Op_::StoreElement: case Op_::StoreElementHole:
        case Op_::PostWriteElementBarrier:
          ok = (user->type() != MIRType::Int32);
          break;
        default:
          ok = false;
      }
      if (!ok) {
        if (wdbg)
          fprintf(stderr, "[wj-widen-abort] %s:%u unsafe-use op=%s ty=%s\n",
                  mir.outerInfo().script() ? mir.outerInfo().script()->filename()
                                           : "?",
                  mir.outerInfo().script()
                      ? unsigned(mir.outerInfo().script()->lineno())
                      : 0,
                  WJOpName(user->op()), StringFromMIRType(user->type()));
        return false;  // not a proven-safe Double sink -> abort
      }
    }
  }

  // 4. APPLY. (a) retype widen nodes to Double. (b) ToNumberInt32 in widen ->
  //    replace with its (now-Double) input. (c) numeric Int32 operands of widen
  //    arith/phi -> MToDouble. (No Int32-use step: aborted above.)
  // (a)+(b)
  std::vector<MInstruction*> toRemove;
  for (MDefinition* w : widen) {
    if (w->op() == Op_::ToNumberInt32) continue;  // handled in pass (b)
    w->setResultType(MIRType::Double);
  }
  for (MDefinition* w : widen) {
    if (w->op() != Op_::ToNumberInt32) continue;
    MDefinition* in = w->getOperand(0);  // Double input
    w->replaceAllUsesWith(in);
    toRemove.push_back(w->toInstruction());
  }
  // (c) widen-node Int32 operands that aren't widen -> MToDouble. For a Phi the
  //     conversion must sit at the END of the matching predecessor block (phi
  //     operands flow from predecessors); for a normal node, right before it.
  for (MDefinition* w : widen) {
    if (!hasNumericOperands(w)) continue;  // loads/unbox: structural operands
    if (w->isPhi()) {
      jit::MPhi* phi = w->toPhi();
      for (size_t i = 0; i < phi->numOperands(); i++) {
        MDefinition* o = phi->getOperand(i);
        if (widen.count(o) || isFP(o) || !isI32(o)) continue;
        jit::MBasicBlock* pred = phi->block()->getPredecessor(i);
        auto* cvt = jit::MToDouble::New(alloc, o);
        pred->insertBefore(pred->lastIns(), cvt);
        phi->replaceOperand(i, cvt);
      }
      continue;
    }
    MInstruction* wi = w->toInstruction();
    for (size_t i = 0; i < wi->numOperands(); i++) {
      MDefinition* o = wi->getOperand(i);
      if (widen.count(o) || isFP(o) || !isI32(o)) continue;
      auto* cvt = jit::MToDouble::New(alloc, o);
      wi->block()->insertBefore(wi, cvt);
      wi->replaceOperand(i, cvt);
    }
  }
  for (MInstruction* r : toRemove) r->block()->discard(r);

  if (wdbg)
    fprintf(stderr, "[wj-widen] %s:%u widened %zu nodes\n",
            mir.outerInfo().script() ? mir.outerInfo().script()->filename() : "?",
            mir.outerInfo().script() ? unsigned(mir.outerInfo().script()->lineno())
                                     : 0,
            widen.size());
  return true;
}

// Elide fallible Unbox:Object whose ONLY uses are property-cache RECEIVERS
// (MGetPropertyCache/MSetPropertyCache operand 0). Those caches lower to the
// generic WJH_GETPROP/WJH_SETPROP helper, which does a full VM GetProperty/
// SetProperty on a BOXED value -- correct for ANY receiver type (object ->
// lookup, number/string -> proto, null/undef -> throw), NO deopt. So Warp's
// Unbox:Object before them is pointless and only adds a deopt-to-PBL (gbemu's
// chained `a.b.c` storm: 109k+ Unbox deopts -> whole fns run in PBL -> 0.94x).
// Rewire the cache's receiver to the Unbox's boxed input and drop the dead
// Unbox. Default-on; GECKO_WJ_NOPROPUNBOX reverts.
static void WJElideUnboxForPropCache(MIRGenerator& mir, MIRGraph& graph) {
  using jit::MDefinition;
  using Op_ = MDefinition::Opcode;
  if (WJGETENV("GECKO_WJ_NOPROPUNBOX")) return;
  std::vector<jit::MInstruction*> dead;
  for (auto bIt = graph.rpoBegin(); bIt != graph.rpoEnd(); bIt++) {
    for (auto it = bIt->begin(); it != bIt->end(); it++) {
      MDefinition* u = *it;
      if (u->op() != Op_::Unbox || u->type() != MIRType::Object) continue;
      if (u->toUnbox()->mode() != jit::MUnbox::Fallible) continue;
      // every DEFINITION use must be a prop-cache with u as the RECEIVER (op0)
      bool ok = true;
      for (auto use = u->usesBegin(); use != u->usesEnd() && ok; use++) {
        if (!use->consumer()->isDefinition()) continue;  // resume use: fine
        MDefinition* c = use->consumer()->toDefinition();
        bool isCache = (c->op() == Op_::GetPropertyCache ||
                        c->op() == Op_::SetPropertyCache);
        if (!isCache || c->getOperand(0) != u) ok = false;
      }
      if (!ok) continue;
      u->replaceAllUsesWith(u->toUnbox()->input());  // boxed value -> helper
      dead.push_back(u->toInstruction());
    }
  }
  for (jit::MInstruction* d : dead) d->block()->discard(d);
  if (!dead.empty() && WJGETENV("GECKO_WJ_PROPUNBOXDBG"))
    fprintf(stderr, "[wj-propunbox] %s:%u elided %zu Unbox:Object\n",
            mir.outerInfo().script() ? mir.outerInfo().script()->filename() : "?",
            mir.outerInfo().script() ? unsigned(mir.outerInfo().script()->lineno())
                                     : 0,
            dead.size());
}

// Replace MSpectreMaskIndex with its index input. WarpCacheIRTranspiler inserts it
// after every element bounds check as a Spectre-v1 mitigation, but it's dead weight
// here: MBoundsCheck already guarantees the index is in range, and the JIT'd code
// runs inside the host's wasm engine (its own speculation sandbox) -- there is no
// native mispredict to exploit. The backend can't lower it (-> [wb-be] bail), so
// elide it: the masked index equals the index.
static void WJElideSpectreMaskIndex(MIRGenerator& mir, MIRGraph& graph) {
  using jit::MDefinition;
  using Op_ = MDefinition::Opcode;
  std::vector<jit::MInstruction*> dead;
  for (auto bIt = graph.rpoBegin(); bIt != graph.rpoEnd(); bIt++) {
    for (auto it = bIt->begin(); it != bIt->end(); it++) {
      MDefinition* s = *it;
      if (s->op() != Op_::SpectreMaskIndex) continue;
      s->replaceAllUsesWith(s->toSpectreMaskIndex()->index());
      dead.push_back(s->toInstruction());
    }
  }
  for (jit::MInstruction* d : dead) d->block()->discard(d);
  if (!dead.empty() && WJGETENV("GECKO_WJ_SPECTREDBG"))
    fprintf(stderr, "[wj-spectre] %s:%u elided %zu SpectreMaskIndex\n",
            mir.outerInfo().script() ? mir.outerInfo().script()->filename() : "?",
            mir.outerInfo().script() ? unsigned(mir.outerInfo().script()->lineno())
                                     : 0,
            dead.size());
}

}  // namespace

// MIR opcode -> name for the deopt/crash histograms in WasmJitRuntime.cpp
// (that TU can't include jit/MIR.h). WJOpName (anon-namespace, above) is
// visible here; its switch returns "?" for any unmatched value.
const char* js::wasm::WJMirOpName(uint32_t op) {
  return WJOpName(jit::MDefinition::Opcode(op));
}

void js::wasm::WJDumpDeoptSiteHist() { WJDumpSiteHist(); }

bool js::wasm::WJEmitTrampoline(Encoder& e) {
  if constexpr (kWJEHABI) {
    // EHABI trampoline: main returns just i64 (or throws a wasm exn). Wrap `call $main`
    // in try(result i64)/catch_all: on catch restore gWJRootSP (wholesale cleanup of all
    // unwound frames' leaked root pushes) + set the return flag to 1 (the JS exception is
    // on cx); WasmJitRunCall reads cx. Locals: 1=i32 sbI32, 2=f64 flag, 3=i64 result,
    // 4=i32 savedRootSP.
    if (!e.writeVarU32(4) || !e.writeVarU32(1) ||
        !e.writeFixedU8(uint8_t(TypeCode::I32)) || !e.writeVarU32(1) ||
        !e.writeFixedU8(uint8_t(TypeCode::F64)) || !e.writeVarU32(1) ||
        !e.writeFixedU8(uint8_t(TypeCode::I64)) || !e.writeVarU32(1) ||
        !e.writeFixedU8(uint8_t(TypeCode::I32)))
      return false;
    uintptr_t spAddr = uintptr_t(static_cast<void*>(&gWJRootSP));
    // sbI32(1) = (i32)param0
    if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(0) ||
        !e.writeOp(MiscOp::I32TruncSatF64S) || !e.writeOp(Op::LocalSet) ||
        !e.writeVarU32(1))
      return false;
    // flag(2) = 0
    if (!e.writeOp(Op::F64Const) || !e.writeFixedF64(0.0) ||
        !e.writeOp(Op::LocalSet) || !e.writeVarU32(2))
      return false;
    // savedRootSP(4) = gWJRootSP
    if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(spAddr)) ||
        !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0) ||
        !e.writeOp(Op::LocalSet) || !e.writeVarU32(4))
      return false;
    // try (result i64)
    if (!e.writeOp(Op::Try) || !e.writeFixedU8(0x7E)) return false;
    // push main args: sb (f64), callee (i64, scratch[kWJCalleeSlot]), this, args.
    if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(0)) return false;  // sb
    if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(1) || !e.writeOp(Op::I64Load) ||
        !e.writeVarU32(3) || !e.writeVarU32(kWJCalleeSlot * 8))
      return false;  // callee
    if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(1) || !e.writeOp(Op::I64Load) ||
        !e.writeVarU32(3) || !e.writeVarU32(kWJThisOff))
      return false;  // this
    for (uint32_t i = 0; i < kWJMaxArgs; i++) {
      if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(1) || !e.writeOp(Op::I64Load) ||
          !e.writeVarU32(3) || !e.writeVarU32(i * 8))
        return false;  // arg i
    }
    if (!e.writeOp(Op::Call) || !e.writeVarU32(1)) return false;  // main -> i64
    // catch_all: restore SP, flag=1, push dummy i64 result.
    if (!e.writeOp(Op::CatchAll)) return false;
    if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(spAddr)) ||
        !e.writeOp(Op::LocalGet) || !e.writeVarU32(4) || !e.writeOp(Op::I32Store) ||
        !e.writeVarU32(2) || !e.writeVarU32(0))
      return false;  // gWJRootSP = savedRootSP
    if (!e.writeOp(Op::F64Const) || !e.writeFixedF64(1.0) ||
        !e.writeOp(Op::LocalSet) || !e.writeVarU32(2))
      return false;  // flag = 1
    if (!e.writeOp(Op::I64Const) || !e.writeVarS64(0)) return false;  // dummy result
    if (!e.writeOp(Op::End)) return false;  // end try
    if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(3)) return false;  // result
    // gWJScratch[kWJResultOff] = result
    if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(1) || !e.writeOp(Op::LocalGet) ||
        !e.writeVarU32(3) || !e.writeOp(Op::I64Store) || !e.writeVarU32(3) ||
        !e.writeVarU32(kWJResultOff))
      return false;
    if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(2)) return false;  // return flag
    return e.writeOp(Op::End);
  }
  // locals: i32 sbI32 (1), f64 flag (2), i64 result (3). Param 0 = scratch-base.
  if (!e.writeVarU32(3) || !e.writeVarU32(1) ||
      !e.writeFixedU8(uint8_t(TypeCode::I32)) || !e.writeVarU32(1) ||
      !e.writeFixedU8(uint8_t(TypeCode::F64)) || !e.writeVarU32(1) ||
      !e.writeFixedU8(uint8_t(TypeCode::I64))) {
    return false;
  }
  // sbI32 (local 1) = (i32)param0
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(0)) return false;
  if (!e.writeOp(MiscOp::I32TruncSatF64S)) return false;
  if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(1)) return false;
  // Push main's args: sb (f64), this (i64), arg0..arg[kWJMaxArgs-1] (i64).
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(0)) return false;  // sb
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(1) || !e.writeOp(Op::I64Load) ||
      !e.writeVarU32(3) || !e.writeVarU32(kWJThisOff)) {
    return false;  // this
  }
  for (uint32_t i = 0; i < kWJMaxArgs; i++) {
    if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(1) ||
        !e.writeOp(Op::I64Load) || !e.writeVarU32(3) || !e.writeVarU32(i * 8)) {
      return false;  // arg i
    }
  }
  if (!e.writeOp(Op::Call) || !e.writeVarU32(1)) return false;  // main -> [flag,result]
  if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(3)) return false;  // result
  if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(2)) return false;  // flag
  // Store the boxed result to gWJScratch[kWJResultOff] for the host entry path.
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(1) || !e.writeOp(Op::LocalGet) ||
      !e.writeVarU32(3) || !e.writeOp(Op::I64Store) || !e.writeVarU32(3) ||
      !e.writeVarU32(kWJResultOff)) {
    return false;
  }
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(2)) return false;  // return flag
  return e.writeOp(Op::End);
}

bool js::wasm::WJEmitBody(MIRGenerator& mir, MIRGraph& graph, uint32_t nargs,
                          bool useThis, WarpSnapshot* snapshot, Encoder& e) {
  const uint8_t kVoid = 0x40;
  {
    JSScript* brs = mir.outerInfo().script();
    js::wasm::gWJBailLine = brs ? brs->lineno() : 0;
    js::wasm::gWJBailReason = "unknown";
  }
  // Bisection: GECKO_WJ_BAILLINE=<n[,n...]> bails (stays PBL) any function defined
  // at one of those source lines. Used to find which function's JIT codegen is
  // wrong (e.g. navier advect/project miscompute exposed by the number-unbox).
  if (const char* bl = WJGETENV("GECKO_WJ_BAILLINE")) {
    JSScript* s = mir.outerInfo().script();
    if (s) {
      uint32_t ln = s->lineno();
      for (const char* p = bl; *p;) {
        uint32_t v = uint32_t(atoi(p));
        if (v == ln) return false;
        while (*p && *p != ',') p++;
        if (*p == ',') p++;
      }
    }
  }
  // Threshold bisection (for big files like earley): BAILGT=N bails fns at
  // lineno > N; BAILLT=N bails fns at lineno < N. Binary-search N to localize a
  // miscompiled function's region.
  {
    JSScript* s = mir.outerInfo().script();
    if (s) {
      uint32_t ln = s->lineno();
      if (const char* g = WJGETENV("GECKO_WJ_BAILGT"))
        if (ln > uint32_t(atoi(g))) return false;
      if (const char* l = WJGETENV("GECKO_WJ_BAILLT"))
        if (ln < uint32_t(atoi(l))) return false;
      // BAILRANGE=lo,hi[;lo2,hi2...] bails fns with lineno inside ANY range
      // (keeps the rest JIT'd, so the run stays fast while bisecting a
      // sub-region of a big file; multiple ranges pin one culprit while
      // bisecting for another).
      if (const char* r = WJGETENV("GECKO_WJ_BAILRANGE")) {
        const char* c = r;
        while (*c) {
          uint32_t lo = uint32_t(atoi(c));
          while (*c && *c != ',' && *c != ';') c++;
          uint32_t hi = lo;
          if (*c == ',') {
            hi = uint32_t(atoi(c + 1));
            while (*c && *c != ';') c++;
          }
          if (ln >= lo && ln <= hi) {
            if (WJGETENV("GECKO_WJ_BAILDBG")) {
              fprintf(stderr, "[wj-baildbg] BAILRANGE bails %s:%u\n",
                      s->filename() ? s->filename() : "?", ln);
            }
            return false;
          }
          if (*c == ';') c++;
        }
      }
      // COLUMN bisection (for MINIFIED bundles where the whole library is on one
      // line -- marked.min.js line 1 is ~50k chars, so BAILLINE/GT/LT can't
      // separate functions). BAILCOLGT=N bails fns at column > N; BAILCOLLT=N at
      // column < N; BAILCOLRANGE=lo,hi bails lo<=column<=hi. Pair with BAILLINE to
      // scope to one line first. Binary-search the column to localize the fn.
      uint32_t col = s->column().oneOriginValue();
      if (const char* g = WJGETENV("GECKO_WJ_BAILCOLGT"))
        if (col > uint32_t(atoi(g))) return false;
      if (const char* l = WJGETENV("GECKO_WJ_BAILCOLLT"))
        if (col < uint32_t(atoi(l))) return false;
      // Multi-range "a,b;c,d" like BAILRANGE (pin one culprit, bisect another).
      if (const char* r = WJGETENV("GECKO_WJ_BAILCOLRANGE")) {
        const char* c = r;
        while (*c) {
          uint32_t lo = uint32_t(atoi(c));
          while (*c && *c != ',' && *c != ';') c++;
          uint32_t hi = lo;
          if (*c == ',') {
            hi = uint32_t(atoi(c + 1));
            while (*c && *c != ';') c++;
          }
          if (col >= lo && col <= hi) return false;
          if (*c == ';') c++;
        }
      }
    }
  }
  // MIR dumper: GECKO_WJ_MIRDUMP=<lineno> prints every instruction (op + operand
  // ops + type) of the function defined at that line -- to inspect a specific
  // codegen pattern (e.g. closure-variable reads after FunctionEnvironment).
  if (const char* md = WJGETENV("GECKO_WJ_MIRDUMP")) {
    JSScript* s = mir.outerInfo().script();
    uint32_t want = uint32_t(atoi(md));
    // Match the OUTER script's lineno OR any INLINEE block's script lineno -- a
    // function that's only ever inlined (never top-level compiled) has its blocks
    // under the inlinee's CompileInfo, not mir.outerInfo(). (prepack's culprit
    // cli.js:416815 is inline-only; the old outer-only match never fired.)
    bool match = (s && s->lineno() == want);
    if (!match) {
      for (ReversePostorderIterator b = graph.rpoBegin(); b != graph.rpoEnd();
           b++) {
        JSScript* bs = b->info().script();
        if (bs && bs->lineno() == want) { match = true; break; }
      }
    }
    fprintf(stderr, "[wb-mir-fn] compiling lineno=%u (want %s) match=%d\n",
            s ? s->lineno() : 0, md, int(match));
    if (match) {
      fprintf(stderr, "[wb-mir] === %s:%u (want %u, inlinee-aware) ===\n",
              s ? s->filename() : "?", s ? s->lineno() : 0, want);
      for (ReversePostorderIterator b = graph.rpoBegin(); b != graph.rpoEnd();
           b++) {
        // Annotate each block with ITS script:lineno so inlinee blocks (the ones
        // whose script lineno == want) are identifiable in the dump.
        JSScript* bs = b->info().script();
        fprintf(stderr, "[wb-mir] --block%u script=%s:%u%s\n", b->id(),
                bs ? bs->filename() : "?", bs ? bs->lineno() : 0,
                (bs && bs->lineno() == want) ? " <== TARGET" : "");
        // Predecessors (CFG reconstruction: phi operand k comes from pred k).
        char preds[128];
        int pn = 0;
        for (size_t i = 0; i < b->numPredecessors() && pn < 100; i++)
          pn += snprintf(preds + pn, sizeof(preds) - pn, "b%u ",
                         b->getPredecessor(i)->id());
        fprintf(stderr, "[wb-mir] block%u preds=[%s]\n", b->id(), preds);
        // Phis with per-operand block+id (so a wrong-edge Phi is visible).
        for (MPhiIterator p = b->phisBegin(); p != b->phisEnd(); p++) {
          MPhi* phi = *p;
          char ops[256];
          int n = 0;
          for (size_t i = 0; i < phi->numOperands() && n < 200; i++)
            n += snprintf(ops + n, sizeof(ops) - n, "b%u:%s#%u:%s ",
                          b->getPredecessor(i)->id(),
                          WJOpName(phi->getOperand(i)->op()),
                          phi->getOperand(i)->id(),
                          StringFromMIRType(phi->getOperand(i)->type()));
          fprintf(stderr, "[wb-mir]   PHI#%u:%s <- %s\n", phi->id(),
                  StringFromMIRType(phi->type()), ops);
        }
        for (MInstructionIterator it = b->begin(); it != b->end(); it++) {
          char ops[256];
          int n = 0;
          for (size_t i = 0; i < it->numOperands() && n < 200; i++) {
            n += snprintf(ops + n, sizeof(ops) - n, "%s#%u:%s ",
                          WJOpName(it->getOperand(i)->op()),
                          it->getOperand(i)->id(),
                          StringFromMIRType(it->getOperand(i)->type()));
          }
          // Slot number for fixed-slot ops (match a store to its load by slot).
          char slot[24];
          slot[0] = 0;
          if (it->isStoreFixedSlot())
            snprintf(slot, sizeof(slot), " [slot=%u]",
                     unsigned(it->toStoreFixedSlot()->slot()));
          else if (it->isLoadFixedSlot())
            snprintf(slot, sizeof(slot), " [slot=%u]",
                     unsigned(it->toLoadFixedSlot()->slot()));
          else if (it->isLoadFixedSlotAndUnbox())
            snprintf(slot, sizeof(slot), " [slot=%u]",
                     unsigned(it->toLoadFixedSlotAndUnbox()->slot()));
          fprintf(stderr, "[wb-mir]   %s#%u:%s <- %s%s\n", WJOpName(it->op()),
                  it->id(), StringFromMIRType(it->type()), ops, slot);
        }
      }
    }
  }
  // Aggregate MIR boxing-structure analysis (GECKO_WJ_MIRSTATS): across all compiled
  // functions, quantify (1) result-type mix, (2) what PRODUCES Value-typed defs, (3)
  // Unbox sites (boxed->typed round-trips) by input type, (4) how many Value defs are
  // consumed by an Unbox-to-T (a removable round-trip if the producer were typed T),
  // (5) typed vs untyped slot reads. Answers: is ubo's boxing inherent (polymorphic
  // Value) or exploitable (producer type known/speculable)? Prints every 64 fns.
  if (WJGETENV("GECKO_WJ_MIRSTATS")) {
    static std::map<std::string, uint64_t> resTy, valProd, unboxIn, boxIn, valConsumer;
    static uint64_t nFns = 0, nDefs = 0, nValDefs = 0, nUnbox = 0, nUnboxFallible = 0,
                    nBox = 0, nValRoundTrip = 0, nLFS = 0, nLFSUnbox = 0, nLDS = 0,
                    nLDSUnbox = 0;
    nFns++;
    for (ReversePostorderIterator b = graph.rpoBegin(); b != graph.rpoEnd(); b++) {
      for (MInstructionIterator it = b->begin(); it != b->end(); it++) {
        MInstruction* d = *it;
        nDefs++;
        resTy[StringFromMIRType(d->type())]++;
        const char* opn = WJOpName(d->op());
        if (d->op() == MDefinition::Opcode::LoadFixedSlot) nLFS++;
        if (d->op() == MDefinition::Opcode::LoadFixedSlotAndUnbox) nLFSUnbox++;
        if (d->op() == MDefinition::Opcode::LoadDynamicSlot) nLDS++;
        if (d->op() == MDefinition::Opcode::LoadDynamicSlotAndUnbox) nLDSUnbox++;
        if (d->op() == MDefinition::Opcode::Box) {
          nBox++;
          boxIn[StringFromMIRType(d->toBox()->input()->type())]++;
        }
        if (d->op() == MDefinition::Opcode::Unbox) {
          nUnbox++;
          if (d->toUnbox()->mode() == MUnbox::Fallible) nUnboxFallible++;
          unboxIn[StringFromMIRType(d->toUnbox()->input()->type())]++;
        }
        if (d->type() == MIRType::Value) {
          nValDefs++;
          valProd[opn]++;
          // classify consumers: all-Unbox (pure removable round-trip) vs mixed/escapes
          bool anyUnbox = false, anyNonUnbox = false;
          for (MUseIterator u = d->usesBegin(); u != d->usesEnd(); u++) {
            if (!u->consumer()->isDefinition()) { anyNonUnbox = true; continue; }
            MDefinition* c = u->consumer()->toDefinition();
            if (c->op() == MDefinition::Opcode::Unbox) anyUnbox = true;
            else anyNonUnbox = true;
          }
          if (anyUnbox) {
            nValRoundTrip++;
            valConsumer[anyNonUnbox ? "Unbox+other" : "Unbox-only"]++;
          } else {
            valConsumer["no-unbox"]++;
          }
        }
      }
    }
    if ((nFns % 64) == 0) {
      auto dump = [&](const char* lbl, std::map<std::string, uint64_t>& m) {
        std::vector<std::pair<uint64_t, std::string>> v;
        for (auto& kv : m) v.push_back({kv.second, kv.first});
        std::sort(v.rbegin(), v.rend());
        fprintf(stderr, "[mirstats] %s:", lbl);
        for (size_t i = 0; i < v.size() && i < 8; i++)
          fprintf(stderr, " %s=%llu", v[i].second.c_str(),
                  (unsigned long long)v[i].first);
        fprintf(stderr, "\n");
      };
      fprintf(stderr,
              "[mirstats] fns=%llu defs=%llu valDefs=%llu(%.0f%%) unbox=%llu(fallible=%llu) "
              "box=%llu valRoundTrip=%llu LFS=%llu LFSunbox=%llu LDS=%llu LDSunbox=%llu\n",
              (unsigned long long)nFns, (unsigned long long)nDefs,
              (unsigned long long)nValDefs, 100.0 * nValDefs / (nDefs ? nDefs : 1),
              (unsigned long long)nUnbox, (unsigned long long)nUnboxFallible,
              (unsigned long long)nBox, (unsigned long long)nValRoundTrip,
              (unsigned long long)nLFS, (unsigned long long)nLFSUnbox,
              (unsigned long long)nLDS, (unsigned long long)nLDSUnbox);
      dump("resultType", resTy);
      dump("Value-producers", valProd);
      dump("Unbox-input", unboxIn);
      dump("Box-input", boxIn);
      dump("Value-consumers", valConsumer);
    }
  }
  // Int32-mis-speculation widening (navier float arrays): re-type float-array
  // value ops Int32->Double so they stop deopting the whole fn to PBL (navier
  // 31->112 = ~4x). Self-selects to functions with a Double element load, so
  // int-only code (crypto) is untouched. DEFAULT-ON; GECKO_WJ_NOWIDEN disables.
  if (!WJGETENV("GECKO_WJ_NOWIDEN")) {
    WJWidenInt32Values(mir, graph);
  }
  // Elide deopt-prone Unbox:Object feeding generic property-cache helpers (gbemu
  // chained property access). See WJElideUnboxForPropCache.
  WJElideUnboxForPropCache(mir, graph);
  // Drop MSpectreMaskIndex (native-CPU mitigation, dead weight here; the backend
  // can't lower it). See WJElideSpectreMaskIndex.
  WJElideSpectreMaskIndex(mir, graph);

  // Bisection knobs: bail (stay PBL) any function containing a disabled op kind.
  static int noCall = WJGETENV("GECKO_WJ_NOCALL") ? 1 : 0;
  static int noStore = WJGETENV("GECKO_WJ_NOSTORE") ? 1 : 0;
  static int noProp = WJGETENV("GECKO_WJ_NOPROP") ? 1 : 0;
  static int noObjConst = WJGETENV("GECKO_WJ_NOOBJCONST") ? 1 : 0;
  static int noGuardShape = WJGETENV("GECKO_WJ_NOGUARDSHAPE") ? 1 : 0;
  if (noCall || noStore || noProp || noObjConst || noGuardShape) {
    for (ReversePostorderIterator b = graph.rpoBegin(); b != graph.rpoEnd(); b++) {
      for (MInstructionIterator it = b->begin(); it != b->end(); it++) {
        auto op = it->op();
        if (noCall && op == MDefinition::Opcode::Call) return false;
        if (noGuardShape && op == MDefinition::Opcode::GuardShape) return false;
        if (noStore && op == MDefinition::Opcode::StoreFixedSlot) return false;
        if (noObjConst && op == MDefinition::Opcode::Constant &&
            it->type() == MIRType::Object)
          return false;
        if (noProp && (op == MDefinition::Opcode::GuardShape ||
                       op == MDefinition::Opcode::Slots ||
                       op == MDefinition::Opcode::LoadFixedSlot ||
                       op == MDefinition::Opcode::LoadFixedSlotAndUnbox ||
                       op == MDefinition::Opcode::LoadDynamicSlotAndUnbox ||
                       op == MDefinition::Opcode::Elements))
          return false;
      }
    }
  }

  gWJHadAlwaysBails = false;  // set true if any block is emitted as alwaysBails deopt
  gWJEmitShapeDeopts = 0;
  gWJEmitTotalDeopts = 0;
  WJBackend be;
  be.info = &mir.outerInfo();
  be.snapshot = snapshot;
  // Params: 0 = scratch-base (f64); EHABI also 1 = callee (i64); then this (i64) and
  // args (i64). Old ABI: 0=sb, 1=this, 2..=args. (kWJThisParam/kWJArg0Param encode this.)
  be.paramCount = (kWJEHABI ? 3 : 2) + kWJMaxArgs;
  be.scratchBase = be.reserve(uint8_t(TypeCode::I32));  // first declared local
  be.unboxScratch = be.reserve(uint8_t(TypeCode::I64));  // unbox temp for fused loads
  be.dynArgAcc = be.reserve(uint8_t(TypeCode::I64));     // dyn GetFrameArgument cascade acc
  be.dynArgIdx = uint32_t(be.reserve(uint8_t(TypeCode::I32)));  // dyn GetFrameArgument index
  be.instofProto = uint32_t(be.reserve(uint8_t(TypeCode::I32)));   // multi-hop instanceof walk
  be.instofCount = uint32_t(be.reserve(uint8_t(TypeCode::I32)));
  be.instofResult = uint32_t(be.reserve(uint8_t(TypeCode::I32)));
  be.iterIdx = uint32_t(be.reserve(uint8_t(TypeCode::I32)));    // Load/StoreSlotByIteratorIndex temps
  be.iterKind = uint32_t(be.reserve(uint8_t(TypeCode::I32)));
  be.iterAddr = uint32_t(be.reserve(uint8_t(TypeCode::I32)));
  be.callScratch = be.reserve(uint8_t(TypeCode::I32));   // call-site IC callee ptr
  be.callArgBase = uint32_t(be.reserve(uint8_t(TypeCode::I64)));  // boxed this
  for (uint32_t i = 0; i < kWJMaxArgs; i++) {
    be.reserve(uint8_t(TypeCode::I64));  // boxed arg i
  }
  be.callFlagLocal = uint32_t(be.reserve(uint8_t(TypeCode::F64)));
  be.callResultLocal = uint32_t(be.reserve(uint8_t(TypeCode::I64)));
  be.envLocal = uint32_t(be.reserve(uint8_t(TypeCode::I32)));  // saved env (see struct)
  be.envRootIdx = uint32_t(be.reserve(uint8_t(TypeCode::I32)));  // persistent env-root slot idx
  be.deoptSiteLocal = uint32_t(be.reserve(uint8_t(TypeCode::I32)));  // out-of-line deopt site id
  be.propObjLocal = uint32_t(be.reserve(uint8_t(TypeCode::I32)));   // prop-IC: receiver ptr
  be.propShapeLocal = uint32_t(be.reserve(uint8_t(TypeCode::I32)));  // prop-IC: receiver shape
  be.propTaggedLocal = uint32_t(be.reserve(uint8_t(TypeCode::I32)));  // prop-IC: tagged slot offset
  be.propBVKeyLocal = uint32_t(be.reserve(uint8_t(TypeCode::I32)));  // by-value mega-probe runtime key
  be.propKeyLocal = uint32_t(be.reserve(uint8_t(TypeCode::I64)));  // element-store IC: boxed key
  be.allocPosLocal = uint32_t(be.reserve(uint8_t(TypeCode::I32)));    // inline-alloc old position
  be.allocObjLocal = uint32_t(be.reserve(uint8_t(TypeCode::I32)));    // inline-alloc result obj
  be.rootBaseLocal = uint32_t(be.reserve(uint8_t(TypeCode::I32)));    // GC-root spill base addr
  be.savedDepthLocal = uint32_t(be.reserve(uint8_t(TypeCode::I32)));  // gWJJitDepth at entry
  be.actualArgcLocal = uint32_t(be.reserve(uint8_t(TypeCode::I32)));  // actual argc snapshot (ArgumentsLength)

  // Leaf-skip for the recursion guard: a function with NO call can't recurse, so
  // it never needs the per-entry depth check (this is what makes the guard cheap --
  // the hot tiny accessors/arith fns are leaves and pay nothing). Conservative: any
  // MCall (direct/indirect/construct) -> not a leaf -> guarded.
  // Same pass estimates the V8 wasm FRAME SIZE for the recursion byte-guard: each
  // value-producing MIR instruction becomes ~1 wasm local (V8 spills each to an
  // 8-byte machine slot), plus params + the ~32 dedicated reserved locals + fixed
  // frame overhead. This makes the guard fire by approximate STACK BYTES (what a
  // real SP check measures) instead of a flat frame count that can't tell a big
  // frame (few levels to overflow) from a small one (many).
  uint32_t valInsts = 0;
  for (ReversePostorderIterator b = graph.rpoBegin(); b != graph.rpoEnd(); b++) {
    for (MInstructionIterator it = b->begin(); it != b->end(); it++) {
      if (it->isCall()) be.hasCalls = true;
      if (it->type() != jit::MIRType::None) valInsts++;
    }
  }
  be.frameBytesEst = (be.paramCount + valInsts + 32) * 8 + 256;

  std::vector<MBasicBlock*> blocks;
  std::unordered_map<MBasicBlock*, uint32_t> blockIdx;
  bool anyAlwaysBails = false;
  for (ReversePostorderIterator b = graph.rpoBegin(); b != graph.rpoEnd(); b++) {
    // A block that always bails (Warp gave up on a COLD path it couldn't type --
    // e.g. navier lin_solve's `if (a===0)` branch, never taken at runtime) is
    // emitted as a deopt-at-entry (resume to PBL) instead of bailing the WHOLE
    // function to PBL. This mirrors Ion, which keeps the hot path compiled and
    // only bails the cold block if it is ever reached. Bailing the whole function
    // here was the navier 1x ceiling: lin_solve/advect/project all have such a
    // cold branch, so none of them compiled. See EmitBlockBody (alwaysBails ->
    // EmitDeopt). The entry block bailing means the function always deopts on
    // entry, so there's no hot path to preserve -- bail it whole.
    if (b->alwaysBails()) anyAlwaysBails = true;
    blockIdx[*b] = uint32_t(blocks.size());
    blocks.push_back(*b);
  }
  const uint32_t n = uint32_t(blocks.size());
  if (n == 0 || blocks[0]->alwaysBails()) {
    js::wasm::gWJBailReason = (n == 0) ? "no-blocks" : "entry-alwaysBails";
    return false;
  }

  // Assign locals: phis first (edge copies target them), then value nodes.
  for (MBasicBlock* b : blocks) {
    for (MPhiIterator p = b->phisBegin(); p != b->phisEnd(); p++) {
      if (WJValType((*p)->type()) != 0) be.assign(*p);
    }
  }
  uint32_t maxLoopDepth = 0;
  bool hasCalls = false;
  // try/catch: does this script have a real catch/finally handler? (Only reached when
  // GECKO_WJ_NOTRYBAIL removed the compile bail.) If so, route call/helper exceptions
  // through deopt-in-error -> PBL HandleException runs the catch (or propagates).
  if (JSScript* wjsc = mir.outerInfo().script()) {
    for (const js::TryNote& tn : wjsc->trynotes()) {
      if (tn.kind() == js::TryNoteKind::Catch ||
          tn.kind() == js::TryNoteKind::Finally) {
        be.hasCatchTrynote = true;
        break;
      }
    }
  }
  for (MBasicBlock* b : blocks) {
    if (b->loopDepth() > maxLoopDepth) maxLoopDepth = b->loopDepth();
    for (MInstructionIterator it = b->begin(); it != b->end(); it++) {
      MInstruction* ins = *it;
      if (ins->op() == MDefinition::Opcode::FunctionEnvironment)
        be.useEnvRoot = true;  // env in envLocal must be rooted across calls
      if (MResumePoint* rp = ins->resumePoint()) {
        if (rp->caller()) be.hasInlinedFrames = true;  // multi-frame => inlined callee
      }
      switch (ins->op()) {
        case MDefinition::Opcode::Call:
        case MDefinition::Opcode::GetPropertyCache:
        case MDefinition::Opcode::SetPropertyCache:
          hasCalls = true;  // reloop's structural overhead is a loss when a
          break;            // C++/helper hop dominates the loop (richards)
        case MDefinition::Opcode::ArgumentsLength:
          be.usesArgc = true;  // -> snapshot gWJCallArgc into actualArgcLocal at entry
          // Entry >kWJMaxArgs guard + arg roots: the deopt-resume actuals-graft can
          // only re-stage 8 actuals (wider calls -> PBL) and must read GC-current
          // values (rooted slots, not raw params).
          if (b->info().script() == be.info->script()) {
            be.needsArgcGuard = true;
            be.needsArgRoots = true;
            be.useEnvRoot = true;
          }
          break;
        case MDefinition::Opcode::Callee:
          if (b->info().script() == be.info->script()) {
            be.usesCallee = true;   // runtime callee, rooted next to the env
            be.useEnvRoot = true;   // the callee root slot lives at envRootIdx+1
          }
          break;
        case MDefinition::Opcode::NewNamedLambdaObject:
          // Needs the runtime callee (envRootIdx+1) staged for WJH_NEWNAMEDLAMBDA.
          if (b->info().script() == be.info->script()) {
            be.usesCallee = true;
            be.useEnvRoot = true;
          }
          break;
        case MDefinition::Opcode::CreateArgumentsObject:
          // Materializing `arguments` needs the actual arg count (snapshot),
          // the runtime callee (envRootIdx+1) and the env (envRootIdx) staged
          // for WJH_NEWARGUMENTS -> ArgumentsObject::createForWasmJit.
          if (b->info().script() == be.info->script()) {
            be.usesArgc = true;
            be.usesCallee = true;
            be.useEnvRoot = true;
            be.needsArgcGuard = true;  // entry argc>kWJMaxArgs -> flag-2 (see prologue)
            be.needsArgRoots = true;   // resume actuals-graft reads the rooted slots
          }
          break;
        case MDefinition::Opcode::Rest:
          // The rest-parameter array is built from the staged actuals; needs the
          // argc snapshot + the entry argc>kWJMaxArgs guard (WJH_REST). Roots the
          // actuals too: the deopt-resume actuals-graft must read GC-CURRENT values
          // (raw params go stale across a pre-deopt GC -- compartment-check crash).
          if (b->info().script() == be.info->script()) {
            be.usesArgc = true;
            be.needsArgcGuard = true;
            be.needsArgRoots = true;
            be.useEnvRoot = true;
          }
          break;
        case MDefinition::Opcode::GuardGlobalGeneration:
          if (!be.gggAddr) {
            be.gggAddr = uint32_t(uintptr_t(
                ins->toGuardGlobalGeneration()->generationAddr()));
            be.gggExpected = ins->toGuardGlobalGeneration()->expected();
          }
          break;
        case MDefinition::Opcode::GetFrameArgument:
          // A DYNAMIC-index frame-arg read (arguments[i], i runtime) is served by a
          // select-cascade over the kWJMaxArgs staged params -> needs the entry
          // argc>kWJMaxArgs guard so idx (<argc) is always < kWJMaxArgs.
          // BOTH forms (const too) now set usesArgc + the entry guard: the deopt-
          // resume actuals-graft needs the argc snapshot, and >kWJMaxArgs calls must
          // stay out of the JIT (only 8 actuals can be re-staged on a resume).
          if (b->info().script() == be.info->script()) {
            be.usesArgc = true;
            be.needsArgcGuard = true;
            be.needsArgRoots = true;  // root the 8 actuals: cascade reads + the
            be.useEnvRoot = true;     // resume actuals-graft need GC-current values
          }
          break;
        case MDefinition::Opcode::FunctionEnvironment:
          // env-rooted fns also root the runtime callee: the deopt-resume frame
          // needs it even when the COMPILED region has no MCallee (JSOp::Callee
          // can live in the uncompiled tail PBL resumes into).
          be.usesCallee = true;
          break;
        default:
          break;
      }
      if (ins->isControlInstruction()) continue;
      if (WJValType(ins->type()) == 0) continue;
      be.assign(ins);
    }
  }

  // alwaysBails-as-deopt: Warp marks a block alwaysBails when it ends in a cold
  // IC (never-profiled) whose transpile yields buildBailoutForColdIC. Ion handles
  // this by compiling the hot path and emitting a cheap bailout-to-Baseline for
  // the cold block; we mirror that by converting the alwaysBails block to a
  // deopt-at-block-entry (skip body+terminator) and compiling the rest of the fn.
  // This is the Ion-correct behavior -- a cold block should NOT bail the whole
  // function to the interpreter. The deopt-under-fast-call GC crash this used to
  // guard against is FIXED (always-push a fresh JitActivation in
  // WasmJitResumeViaPBL). GECKO_WJ_ABGATE=1 restores the old whole-function bail
  // (shallow-loop + alwaysBails -> PBL) for A/B debugging.
  if (anyAlwaysBails && maxLoopDepth < 2 && WJGETENV("GECKO_WJ_ABGATE")) {
    js::wasm::gWJBailReason = "ab-gate";
    return false;
  }

  // Switch-in-loop functions compile and run correctly with deopts live since the
  // PBL TableSwitch resume-base fix (2026-07-10, see reloop-bug-notes.md) -- the
  // old reloop-switch-arraypush bail, SELFBOUNDSWITCH clamp and PRERESERVE remat
  // masks are gone.

  // Diagnostic: per-function MIR opcode histogram (GECKO_WJ_OPHIST). Used to see
  // whether hot property/array accesses fold to inline loads (LoadFixedSlot/
  // LoadElement) vs fall to C++ helper hops (GetPropertyCache/MegamorphicLoadSlot)
  // -- the codegen-quality ceiling for splay/navier/raytrace.
  static int opHist = WJGETENV("GECKO_WJ_OPHIST") ? 1 : 0;
  if (opHist) {
    using Op_ = MDefinition::Opcode;
    auto cat = [](Op_ o) -> const char* {
      switch (o) {
        case Op_::GetPropertyCache: return "GetPropCache.HELPER";
        case Op_::SetPropertyCache: return "SetPropCache.HELPER";
        case Op_::MegamorphicLoadSlot: return "MegaLoad.HELPER";
        case Op_::MegamorphicSetElement: return "MegaSet.HELPER";
        case Op_::CallGetIntrinsicValue: return "other";
        case Op_::LoadFixedSlot: return "LoadFixedSlot.inline";
        case Op_::LoadFixedSlotAndUnbox: return "LoadFixedSlotUnbox.inline";
        case Op_::StoreFixedSlot: return "StoreFixedSlot.inline";
        case Op_::LoadDynamicSlot: return "LoadDynSlot.inline";
        case Op_::LoadDynamicSlotAndUnbox: return "LoadDynSlotUnbox.inline";
        case Op_::StoreDynamicSlot: return "StoreDynSlot.inline";
        case Op_::LoadElement: return "LoadElement.inline";
        case Op_::LoadElementAndUnbox: return "LoadElemUnbox.inline";
        case Op_::StoreElement: return "StoreElement.inline";
        case Op_::Call: return "Call";
        case Op_::GuardShape: return "GuardShape";
        case Op_::Unbox: return "Unbox";
        case Op_::Box: return "Box";
        default: return "other";
      }
    };
    std::map<std::string, uint32_t> hist;
    uint32_t total = 0;
    for (MBasicBlock* b : blocks) {
      for (MInstructionIterator it = b->begin(); it != b->end(); it++) {
        hist[cat((*it)->op())]++;
        total++;
      }
    }
    JSScript* s = mir.outerInfo().script();
    const char* fname = "?";
    JS::UniqueChars fnameHolder;
    if (s && s->function() && s->function()->maybePartialDisplayAtom()) {
      JSContext* ncx = js::TlsContext.get();
      fnameHolder = js::AtomToPrintableString(ncx, s->function()->maybePartialDisplayAtom());
      if (fnameHolder) fname = fnameHolder.get();
    }
    fprintf(stderr, "[wb-ophist] %s:%u name=%s total=%u", s ? s->filename() : "?",
            s ? s->lineno() : 0, fname, total);
    for (auto& kv : hist) fprintf(stderr, " %s=%u", kv.first.c_str(), kv.second);
    fprintf(stderr, "\n");
  }

  const uint32_t bidLocal = be.paramCount + uint32_t(be.localTy.size());

  // Locals declaration: each declared local as its own group, plus $bid (i32).
  if (!e.writeVarU32(uint32_t(be.localTy.size()) + 1)) return false;
  for (uint8_t ty : be.localTy) {
    if (!e.writeVarU32(1) || !e.writeFixedU8(ty)) return false;
  }
  if (!e.writeVarU32(1) || !e.writeFixedU8(uint8_t(TypeCode::I32))) return false;

  // Prologue: scratchBase = (i32)param0.
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(0)) return false;
  if (!e.writeOp(MiscOp::I32TruncSatF64S)) return false;
  if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(be.scratchBase)) return false;

  // Prologue: envLocal = gWJCurrentEnv. The caller (entry trampoline, WJH_CALL,
  // or the fast inline call_indirect) sets gWJCurrentEnv to THIS invocation's
  // environment right before transferring control. Saving it to a local now --
  // before any nested call can overwrite the global -- makes MFunctionEnvironment
  // sound for closures (each invocation keeps its own env).
  // Entry GuardGlobalGeneration check, BEFORE any effect or root push: on a
  // generation mismatch return [flag=2.0, undefined] = "did not run" -- the
  // caller executes the function in PBL (no deopt-resume nesting; the storm
  // during bundle-init global definition phases exhausted the V8 stack
  // uncatchably). Mid-function bumps are still covered by the per-site guard.
  // Constructible functions are excluded: the inline-ctor call path doesn't
  // understand the flag-2 "didn't run" protocol (it would propagate a phantom
  // exception); they keep the per-site GGG deopt instead.
  if (be.gggAddr) {
    if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(be.gggAddr)) ||
        !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0) ||
        !e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(be.gggExpected)) ||
        !e.writeOp(Op::I32Ne))
      return false;
    if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
    if (!e.writeOp(Op::F64Const) || !e.writeFixedF64(2.0) ||
        !e.writeOp(Op::I64Const) ||
        !e.writeVarS64(int64_t(JS::UndefinedValue().asRawBits())) ||
        !e.writeOp(Op::Return))
      return false;
    if (!e.writeOp(Op::End)) return false;
  }
  {
    uintptr_t addr = uintptr_t(static_cast<void*>(&gWJCurrentEnv));
    if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(addr)) ||
        !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0) ||
        !e.writeOp(Op::LocalSet) || !e.writeVarU32(be.envLocal))
      return false;
  }
  // Prologue: actualArgcLocal = gWJCallArgc (this invocation's actual arg count).
  // Snapshotted right after env, BEFORE any nested call can overwrite the global --
  // exactly the same invariant as gWJCurrentEnv. The caller (entry trampoline, fast
  // call_indirect, WJH_CALL slow path, construct) sets gWJCallArgc before transfer.
  // Only emitted for functions that read it (ArgumentsLength), so leaves pay nothing.
  if (be.usesArgc) {
    uintptr_t aAddr = uintptr_t(static_cast<void*>(&gWJCallArgc));
    if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(aAddr)) ||
        !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0) ||
        !e.writeOp(Op::LocalSet) || !e.writeVarU32(be.actualArgcLocal))
      return false;
  }
  // GECKO_WJ_ARGC0DBG: trap a zero argc snapshot at entry (kind-252 debug helper
  // prints the dispatch globals) -- diagnosing stale/unstaged gWJCallArgc.
  if (be.usesArgc && WJGETENV("GECKO_WJ_ARGC0DBG")) {
    if (!GetLocal(e, be.actualArgcLocal) || !e.writeOp(Op::I32Eqz)) return false;
    if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
    if (!e.writeOp(Op::F64Const) || !e.writeFixedF64(252.0) ||
        !e.writeOp(Op::F64Const) || !e.writeFixedF64(0.0) ||
        !e.writeOp(Op::Call) || !e.writeVarU32(0) || !e.writeOp(Op::Drop))
      return false;
    if (!e.writeOp(Op::End)) return false;
  }
  // Entry variadic-args guard: a fn that materializes `arguments` (or a rest array)
  // can only stage kWJMaxArgs actuals; if called with MORE, return flag-2 "did not
  // run" so the caller executes it in PBL (which has the full frame). Same reliable
  // mechanism as the GGG check above (no deopt-resume, no valve). Emitted before any
  // GC-root push, so nothing to unwind. Body then always sees actualArgc<=kWJMaxArgs.
  if (be.needsArgcGuard && be.usesArgc) {
    if (!GetLocal(e, be.actualArgcLocal) || !e.writeOp(Op::I32Const) ||
        !e.writeVarS32(int32_t(kWJMaxArgs)) || !e.writeOp(Op::I32GtU))
      return false;
    if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
    if (!e.writeOp(Op::F64Const) || !e.writeFixedF64(2.0) ||
        !e.writeOp(Op::I64Const) ||
        !e.writeVarS64(int64_t(JS::UndefinedValue().asRawBits())) ||
        !e.writeOp(Op::Return))
      return false;
    if (!e.writeOp(Op::End)) return false;
  }
  // Persistently GC-root the env for closures: push boxed(envLocal) onto the
  // GC-root shadow stack at the current gWJRootSP, remember the slot index in
  // envRootIdx, and bump gWJRootSP. The slot is traced on EVERY GC (minor+major)
  // for the function's whole lifetime, so FunctionEnvironment can rematerialize a
  // GC-current env even across non-call GC points (allocations). Popped at exit.
  if (be.useEnvRoot) {
    uintptr_t spAddr = uintptr_t(static_cast<void*>(&gWJRootSP));
    uintptr_t rootsBase = uintptr_t(static_cast<void*>(&gWJCallRoots[0]));
    // envRootIdx = gWJRootSP
    if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(spAddr)) ||
        !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0) ||
        !e.writeOp(Op::LocalSet) || !e.writeVarU32(be.envRootIdx))
      return false;
    // gWJCallRoots[envRootIdx] = box(envLocal):  addr = rootsBase + envRootIdx*8
    if (!GetLocal(e, be.envRootIdx) || !e.writeOp(Op::I32Const) ||
        !e.writeVarS32(8) || !e.writeOp(Op::I32Mul) || !e.writeOp(Op::I32Const) ||
        !e.writeVarS32(int32_t(rootsBase)) || !e.writeOp(Op::I32Add))
      return false;
    if (!GetLocal(e, be.envLocal) || !EmitBoxFromStack(e, MIRType::Object) ||
        !e.writeOp(Op::I64Store) || !e.writeVarU32(3) || !e.writeVarU32(0))
      return false;
    // usesCallee: gWJCallRoots[envRootIdx+1] = gWJCallCallee (the boxed RUNTIME
    // callee, set by every dispatch path before transfer). Traced like the env,
    // so MCallee reads stay GC-current. See usesCallee in the struct.
    if (be.usesCallee) {
      uintptr_t calleeAddr = uintptr_t(static_cast<void*>(&gWJCallCallee));
      if (!GetLocal(e, be.envRootIdx) || !e.writeOp(Op::I32Const) ||
          !e.writeVarS32(8) || !e.writeOp(Op::I32Mul) || !e.writeOp(Op::I32Const) ||
          !e.writeVarS32(int32_t(rootsBase)) || !e.writeOp(Op::I32Add))
        return false;
      if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(calleeAddr)) ||
          !e.writeOp(Op::I64Load) || !e.writeVarU32(3) || !e.writeVarU32(0) ||
          !e.writeOp(Op::I64Store) || !e.writeVarU32(3) || !e.writeVarU32(8))
        return false;
    }
    // needsArgRoots: root the kWJMaxArgs actuals in shadow slots after env/callee
    // (traced+relocated by WJTraceRoots -> GC-current). GetFrameArgument reads these
    // for pure-variadic fns that have no formal Parameter copy locals. Popped with the
    // env (EmitEnvRootPop restores gWJRootSP=envRootIdx, discarding these).
    uint32_t argRootBase = be.usesCallee ? 2 : 1;  // slot offset from envRootIdx
    if (be.needsArgRoots) {
      for (uint32_t i = 0; i < kWJMaxArgs; i++) {
        // gWJCallRoots[envRootIdx + argRootBase + i] = raw param arg i (boxed Value)
        if (!GetLocal(e, be.envRootIdx) || !e.writeOp(Op::I32Const) ||
            !e.writeVarS32(int32_t(argRootBase + i)) || !e.writeOp(Op::I32Add) ||
            !e.writeOp(Op::I32Const) || !e.writeVarS32(8) || !e.writeOp(Op::I32Mul) ||
            !e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(rootsBase)) ||
            !e.writeOp(Op::I32Add))
          return false;
        if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(kWJArg0Param + i) ||
            !e.writeOp(Op::I64Store) || !e.writeVarU32(3) || !e.writeVarU32(0))
          return false;
      }
    }
    // gWJRootSP = envRootIdx + argRootBase (+ kWJMaxArgs if needsArgRoots)
    uint32_t spBump = argRootBase + (be.needsArgRoots ? kWJMaxArgs : 0);
    if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(spAddr)) ||
        !GetLocal(e, be.envRootIdx) || !e.writeOp(Op::I32Const) ||
        !e.writeVarS32(int32_t(spBump)) || !e.writeOp(Op::I32Add) ||
        !e.writeOp(Op::I32Store) ||
        !e.writeVarU32(2) || !e.writeVarU32(0))
      return false;
  }

  // Initialize each Parameter's COPY local from its raw wasm param at the very
  // top of the body -- before any block-entry spill, call, or deopt. GetOp/spill
  // for a Parameter then read this copy local, which WJCollectRoots reloads across
  // every call/GC point, so a minor GC that moves an arg object (earley/nboyer
  // `new sc_Pair`) can't leave a stale raw param behind. Params are always boxed
  // i64 Values, so a plain LocalGet pidx -> LocalSet copy suffices.
  for (MInstructionIterator it = graph.entryBlock()->begin();
       it != graph.entryBlock()->end(); it++) {
    if (it->op() != MDefinition::Opcode::Parameter) continue;
    MParameter* p = it->toParameter();
    int32_t cl = be.local(p);
    if (cl < 0) continue;
    if (p->index() != MParameter::THIS_SLOT &&
        uint32_t(p->index()) >= kWJMaxArgs)
      continue;
    uint32_t pidx = (p->index() == MParameter::THIS_SLOT)
                        ? kWJThisParam
                        : (kWJArg0Param + uint32_t(p->index()));
    if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(pidx)) return false;
    if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(uint32_t(cl))) return false;
    // Record the GC-rooted copy local for this arg index so GetFrameArgument reads
    // the copy (GC-current) instead of the raw param (stale after a minor GC).
    if (p->index() != MParameter::THIS_SLOT &&
        uint32_t(p->index()) < kWJMaxArgs)
      be.argCopyLocal[uint32_t(p->index())] = cl;
  }

  static int cfInfo = WJGETENV("GECKO_WJ_CFINFO") ? 1 : 0;
  size_t cfStartOff = e.currentOffset();
  uint32_t cfLineno = 0;
  if (cfInfo) {
    JSScript* s = mir.outerInfo().script();
    cfLineno = s ? s->lineno() : 0;
  }
  auto cfReport = [&](const char* path) {
    if (cfInfo) {
      fprintf(stderr, "[wb-cf] lineno=%u n=%u path=%s bodybytes=%zu\n", cfLineno,
              n, path, e.currentOffset() - cfStartOff);
    }
  };

  // Single-block fast path: straight-line, no dispatch loop. OOL deopt by default:
  // wrap the body in `block $D` so each guard miss is just `cmp; br $D` and the full
  // deopt spill+resume sequence lives out-of-line after $D (a dispatcher keyed on the
  // site) -- exactly like Ion's `jcc bailout_stub`. Without this, EVERY guard in a
  // leaf function (raytrace's dot/add/etc.) carried a ~13-store inline deopt sequence
  // in the hot path, bloating it and pinning values out of registers. GECKO_WJ_NOOOL /
  // NOSINGLEOOL revert to inline deopt.
  if (n == 1) {
    MBasicBlock* b = blocks[0];
    static int noOolS = WJGETENV("GECKO_WJ_NOOOL") ? 1 : 0;
    static int noSingleOol = WJGETENV("GECKO_WJ_NOSINGLEOOL") ? 1 : 0;
    bool singleOol = !noOolS && !noSingleOol;
    be.oolDeopt = singleOol;
    be.reloopOol = singleOol;       // reuse the reloop OOL br-depth formula
    be.reloopScopeDepth = 0;        // body sits directly inside $D
    const uint8_t kVoidB = 0x40;
    if (singleOol) {
      if (!e.writeOp(Op::Block) || !e.writeFixedU8(kVoidB)) return false;  // $D
    }
    be.inDispatchBody = singleOol;  // in-body guard misses br to $D
    bool bodyOk = EmitBlockBody(e, be, b);
    be.inDispatchBody = false;      // terminator deopts stay inline
    if (!bodyOk) return false;
    MControlInstruction* t = b->lastIns();
    if (t->resumePoint()) be.curRp = t->resumePoint();  // else keep body last RP
    if (t->isReturn()) {
      if (!EmitReturn(e, be, t->getOperand(0))) return false;
    } else if (t->isUnreachable() || t->isThrow()) {
      if (!EmitDeopt(e, be)) return false;
    } else {
      js::wasm::gWJBailReason = "single-bad-term";
      return false;
    }
    if (singleOol) {
      if (!e.writeOp(Op::Unreachable)) return false;  // dead (body returned/deopted)
      if (!e.writeOp(Op::End)) return false;          // end $D
      for (uint32_t s = 0; s < be.deoptSites.size(); s++) {
        if (!GetLocal(e, be.deoptSiteLocal) || !e.writeOp(Op::I32Const) ||
            !e.writeVarS32(int32_t(s)) || !e.writeOp(Op::I32Eq) ||
            !e.writeOp(Op::If) || !e.writeFixedU8(kVoidB))
          return false;
        be.curRp = be.deoptSites[s].first;
        be.curOp = be.deoptSites[s].second;
        if (!EmitDeoptResumeInline(e, be)) {
          if (!js::wasm::gWJBailReason ||
              !strstr(js::wasm::gWJBailReason, "resume"))
            js::wasm::gWJBailReason = "single-deopt-resume";
          return false;
        }
        if (!e.writeOp(Op::End)) return false;
      }
      if (!e.writeOp(Op::Unreachable)) return false;
    }
    cfReport("single");
    return e.writeOp(Op::End);
  }

  // Structured control flow (relooper): real wasm loops/blocks/ifs, so the host
  // VM optimizes the loop natively instead of paying the br_table dispatch loop's
  // per-edge overhead. Big win for TIGHT compute loops (crypto am3, splay tree
  // ops: up to 1.5x); a net loss for call-heavy loops where the call dominates.
  // NESTED loops: historically blamed for wrong results (navier advect) -- that
  // was actually the PBL TableSwitch resume-base bug (fixed 2026-07-10;
  // GECKO_WJ_RELOOPNEST is CORRECT now) -- but an A/B measured NO perf win for
  // nested-loop fns (octane navier/gbemu/box2d/raytrace + realapp/coffee all
  // neutral), so the depth<=1 gate stays for simplicity.
  // Auto-gate: enable the relooper for functions whose max loop depth <= 1 AND
  // which make no calls (a call dominates the loop, so reloop's structure is a net
  // loss -- regressed richards). That's exactly where the natural-wasm-loop win
  // lands (crypto am3, splay tree ops). GECKO_WJ_RELOOP forces on; NORELOOP forces
  // off. If reloop bails mid-emit we fall through to the dispatch loop (started is
  // only set true once WJEmitStructured commits, in which case we must bail).
  // Megamorphic recompile flag MUST be set before the relooper branch: small
  // loopless call-less functions (deltablue output()/input()/isSatisfied) take the
  // reloop path, which returns before the dispatch-loop setup below -- if forceMega
  // were set only there, the relooper would emit the deopting GuardShape on a
  // polymorphic receiver and storm (deltablue 414 was here). Setting it up here
  // makes BOTH paths mega-convert.
  // Megamorphic codegen is now the DEFAULT for every compile (not just post-storm
  // recompiles): shape-guarded property reads/stores become self-guarding by-name
  // EmitPropIC (per-site shape cache + helper fallback, NO deopt on a polymorphic
  // receiver). This is the keystone that made deltablue correct AND fast (BROKEN
  // /timeout -> 109 = ~9x): the valve's PARTIAL mega left some reads as fixed-slot
  // under removed guards (wrong slot -> "Cycle"); full mega is consistent. Measured
  // no regression to the others (crypto 124->174, raytrace 40->52, richards/splay/
  // navier comparable). GECKO_WJ_NOMEGA reverts to fixed-slot+deopt codegen.
  // forceMega DEFAULT-ON (every compile). Honest -O2 measurements show this is a
  // DOUBLE-EDGED keystone: it is REQUIRED for deltablue (selective/valve-only mega
  // makes deltablue storm+HANG -- its runtime-polymorphic constraint methods need
  // by-name reads from first compile), but it HALVES richards (126 vs 242 = -48%,
  // mono code paying mega cache/helper bloat) and CORRUPTS box2d (memory-OOB via
  // the EmitPropIC path). The proper fix is PER-ACCESS mega (mono accesses keep a
  // fast guarded read, only polymorphic ones go mega) -- not landed yet.
  // DEFAULT-OFF (2026-06-27): forceMega routes monomorphic stores through the
  // WJH_SETPROP helper (per-store C++ hop) and globally converts reads, which
  // crushed splay (214->483 off) and richards (191->1165 off). It was only
  // default-ON to mask the GuardShape staleness crash, now FIXED (WJCollectRoots
  // list-order, see [[forcemega-double-edged]]/[[gbemu-deopt-storm]]). Off-by-default
  // wins splay/gbemu/earley (the targets) + richards 10x; raytrace/crypto prefer
  // mega and can opt in. GECKO_WJ_FORCEMEGA = always on; GECKO_WJ_MEGAVALVE =
  // storm-triggered per-function (now crash-free; crypto 1240, deltablue 283).
  be.forceMega = WJGETENV("GECKO_WJ_FORCEMEGA") ||
                 (WJGETENV("GECKO_WJ_MEGAVALVE") && js::wasm::gWJForceMega);
  // GECKO_WJ_REMATCACHE: DISPROVEN (2026-06-27). Caching re-derived Unbox/GuardShape
  // pointers in locals to "avoid rematerialization" REGRESSES splay ~42% (266 vs 470)
  // -- TurboFan (V8, --no-liftoff) ALREADY CSEs the on-stack re-derivation optimally;
  // introducing explicit mutable locals defeats its SSA optimization (phi nodes / pinned
  // values). Correct but net-negative => default-off. Proves the per-op cost is
  // semantically-required ops (boxing/guards), NOT eliminable redundancy. See
  // [[profiling-and-the-per-op-wall]]. Do NOT re-attempt manual remat/CSE caching.
  be.rematCacheOn = WJGETENV("GECKO_WJ_REMATCACHE") ? 1 : 0;
  be.rematCacheLocal.clear();
  be.rematValid.clear();
  static int forceReloop = WJGETENV("GECKO_WJ_RELOOP") ? 1 : 0;
  static int noReloop = WJGETENV("GECKO_WJ_NORELOOP") ? 1 : 0;
  static int reloopNest = WJGETENV("GECKO_WJ_RELOOPNEST") ? 1 : 0;
  // Gate: reloop functions with no nested loops (maxLoopDepth<=1; the relooper
  // mis-handles nested loops -> navier) AND no calls. The `!hasCalls` restriction
  // is kept as the DEFAULT because widening to call-heavy functions is net-mixed:
  // crypto +61% (inline deopt, correct) but richards -27%/navier -12% (inline
  // deopt bloat), and OOL deopt (the documented cure for the call-heavy bloat)
  // currently miscompiles crypto (ERR). Opt-in via GECKO_WJ_RELOOPCALLS to widen
  // the gate to call-having shallow functions (uses inline deopt) -- correct
  // everywhere measured EXCEPT it regresses richards/navier. The pre-validation
  // in WJEmitStructured makes this safe from the deltablue compile-storm.
  // DEFAULT-ON (2026-06-27): widen the relooper to call-having shallow-loop
  // functions. Was opt-in because call-heavy reloop with inline deopt regressed
  // richards/navier and OOL deopt (the cure) miscompiled crypto. Both are now
  // resolved -- auto-OOL above + the WJCollectRoots fix -- so the widened gate is
  // a net suite win (crypto +24%, richards +16%, navier +13%, earley +8%,
  // splay/gbemu neutral, no ERRs). GECKO_WJ_NORELOOPCALLS reverts to !hasCalls.
  static int reloopCalls = WJGETENV("GECKO_WJ_NORELOOPCALLS") ? 0 : 1;
  bool useReloop = forceReloop ||
                   (!noReloop && (reloopNest ||
                                  (maxLoopDepth <= 1 && (!hasCalls || reloopCalls))));
  // try/catch: the relooper mis-handles try-block CFGs (it starts then hard-bails ->
  // no fallback). The dispatch loop handles arbitrary CFGs, so route try/catch fns
  // (hasCatchTrynote, compiled via NOTRYBAIL) there.
  if (be.hasCatchTrynote) useReloop = false;
  if (useReloop) {
    be.reloopHasCalls = hasCalls;  // call-heavy -> OOL deopt inside the relooper
    // Checkpoint so a STARTED-but-failed structuring attempt can roll back and
    // fall through to the dispatch-loop emitter instead of bailing the whole
    // compile to PBL (typescript had 4 fns permanently stranded this way).
    // Rolled back: emitted bytes, declared locals, deopt sites. Leaked
    // per-attempt: keyed prop-IC sites are REUSED by key; plain call sites leak
    // one slot per failed attempt (bounded, one-time). OSR targets may get
    // duplicate entries (OSR is opt-in off). GECKO_WJ_NORELOOPFALLBACK reverts.
    size_t ckBytes = e.currentOffset();
    size_t ckLocals = be.localTy.size();
    size_t ckDeopt = be.deoptSites.size();
    bool started = false;
    if (WJEmitStructured(e, be, blocks, blockIdx, &started)) {
      cfReport("reloop");
      return true;
    }
    if (WJGETENV("GECKO_WJ_CFINFO"))
      fprintf(stderr, "[wb-cf] lineno=%u reloop-%s reason=%s\n", cfLineno,
              started ? "FAILED(fallback to dispatch)" : "NOSTART",
              js::wasm::gWJBailReason ? js::wasm::gWJBailReason : "?");
    if (started) {
      if (WJGETENV("GECKO_WJ_NORELOOPFALLBACK")) {
        cfReport("reloop-bail");
        js::wasm::gWJBailReason = "reloop-bail";
        return false;
      }
      e.wjShrinkTo(ckBytes);
      be.localTy.resize(ckLocals);
      be.deoptSites.resize(ckDeopt);
      be.reloopOol = false;
      be.inDispatchBody = false;
      be.reloopScopeDepth = 0;
      cfReport("reloop-fallback");
    } else {
      cfReport("reloop-nostart");
    }
  }

  // Out-of-line deopt: wrap the dispatch loop in `block $D`. A guard miss in a
  // hot block body sets deoptSiteLocal and `br $D`, jumping to the dispatcher
  // emitted AFTER the loop -- so the hot loop body stays tiny (no inline spill
  // bloat). Default on; GECKO_WJ_NOOOL falls back to inline deopt everywhere.
  static int noOol = WJGETENV("GECKO_WJ_NOOOL") ? 1 : 0;
  be.oolDeopt = !noOol;
  // OSR cheap-resume entry (GECKO_WJ_OSR): if the runtime set gWJOsrActive, restore
  // the frame locals from the deopt spill (gWJResumeVals) and start the dispatch loop
  // at gWJOsrBlock instead of block 0 -- so a deopt that resumes at a loop head can
  // re-enter the JIT here instead of interpreting the whole rest of the fn in PBL.
  // Only single-frame, depth-0 loop-header blocks are OSR targets. Per-target locals-
  // load (the slot->def map differs per loop head). gWJResumeVals layout: f=0 frame,
  // order this/args/locals (matches EmitDeoptResumeInline's spill). Cleared after.
  static int osrEnabled = WJGETENV("GECKO_WJ_OSR") ? 1 : 0;
  if (osrEnabled) {
    uintptr_t activeAddr = uintptr_t(static_cast<void*>(&gWJOsrActive));
    uintptr_t blockAddr = uintptr_t(static_cast<void*>(&gWJOsrBlock));
    if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(activeAddr)) ||
        !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0))
      return false;
    if (!e.writeOp(Op::If) || !e.writeFixedU8(kVoid)) return false;  // if gWJOsrActive
    // OSR targets = any single-frame, depth-0 (stack-empty) block entry. A deopt
    // resuming at such a block's start pc can re-enter the JIT there: the dispatch
    // loop reaches any block via bidLocal, and depth 0 means the locals fully
    // determine the resumed state (no live expression-stack values to rebuild).
    // (NOT just loop headers -- crypto-sha1's core_sha1 deopt resumes at pc=272,
    // a mid-loop stack-empty block, not a LoopHead at pc=124/204.)
    static int osrLoopOnly = WJGETENV("GECKO_WJ_OSRLOOPONLY") ? 1 : 0;
    // OSR-LIVENESS GATE (sound, conservative). OSR jumps into a block, skipping its
    // dominators (esp. a loop PREHEADER). A def with a PERSISTENT wasm local that is
    // defined BEFORE the OSR block and used AT/AFTER it (a LICM-hoisted loop-invariant
    // SSA temp) is un-restored garbage under OSR -- the deopt spill saves only frame
    // slots (this/args/locals), not SSA-temp locals -- feeding a call_indirect ->
    // "table index out of bounds" (coffeescript/typescript). Reject such blocks as OSR
    // targets. Over-rejection is SAFE (fewer OSR sites, never a miscompile); diffcheck
    // guards under-rejection. GECKO_WJ_OSRNOLIVEGATE disables (A/B). Built only when OSR
    // is enabled (no default-build cost). block index approximates program order (reloop
    // numbers dominators lower); a spanning persistent def => the block's dominators are
    // needed => unsafe to jump straight in.
    static int osrNoLiveGate = WJGETENV("GECKO_WJ_OSRNOLIVEGATE") ? 1 : 0;
    std::unordered_map<uint32_t, uint32_t> osrMaxUseBlk;
    std::vector<std::tuple<MDefinition*, uint32_t, uint32_t>> osrPersistDefs;
    if (!osrNoLiveGate) {
      for (uint32_t bi = 0; bi < n; bi++) {
        MBasicBlock* bb = blocks[bi];
        for (MInstructionIterator it = bb->begin(); it != bb->end(); it++)
          for (size_t k = 0; k < (*it)->numOperands(); k++) {
            uint32_t oid = (*it)->getOperand(k)->id();
            if (bi > osrMaxUseBlk[oid]) osrMaxUseBlk[oid] = bi;
          }
        for (MPhiIterator it = bb->phisBegin(); it != bb->phisEnd(); it++)
          for (size_t k = 0; k < (*it)->numOperands(); k++) {
            uint32_t oid = (*it)->getOperand(k)->id();
            if (bi > osrMaxUseBlk[oid]) osrMaxUseBlk[oid] = bi;
          }
      }
      for (uint32_t bi = 0; bi < n; bi++) {
        MBasicBlock* bb = blocks[bi];
        for (MInstructionIterator it = bb->begin(); it != bb->end(); it++)
          if (be.local(*it) >= 0) {
            auto mu = osrMaxUseBlk.find((*it)->id());
            osrPersistDefs.emplace_back(*it, bi, mu == osrMaxUseBlk.end() ? 0 : mu->second);
          }
        for (MPhiIterator it = bb->phisBegin(); it != bb->phisEnd(); it++)
          if (be.local(*it) >= 0) {
            auto mu = osrMaxUseBlk.find((*it)->id());
            osrPersistDefs.emplace_back(*it, bi, mu == osrMaxUseBlk.end() ? 0 : mu->second);
          }
      }
    }
    for (uint32_t bi = 0; bi < n; bi++) {
      MBasicBlock* b = blocks[bi];
      if (osrLoopOnly && !b->isLoopHeader()) continue;
      MResumePoint* rp = b->entryResumePoint();
      if (!rp || rp->caller()) continue;  // single-frame only
      if (!IsResumeAfter(rp->mode())) { /* ResumeAt: pc is the block start */ }
      const CompileInfo& info = rp->block()->info();
      if (uint32_t(rp->numOperands()) != info.firstStackSlot()) continue;  // depth 0
      uint32_t idx = blockIdx[b];
      if (!osrNoLiveGate) {
        // Restored frame-slot defs for this block (this/args/locals rp operands).
        std::set<uint32_t> restored;
        restored.insert(rp->getOperand(info.thisSlot())->id());
        for (uint32_t i = 0; i < info.nargs(); i++)
          restored.insert(rp->getOperand(info.argSlot(i))->id());
        for (uint32_t i = 0; i < info.nlocals(); i++)
          restored.insert(rp->getOperand(info.localSlot(i))->id());
        bool unsafe = false;
        for (auto& pd : osrPersistDefs) {
          uint32_t dblk = std::get<1>(pd), mublk = std::get<2>(pd);
          if (dblk >= idx || mublk < idx) continue;  // not spanning this block
          if (restored.count(std::get<0>(pd)->id())) continue;  // restored frame slot
          unsafe = true;
          break;
        }
        if (unsafe) continue;  // skip: has an un-restored persistent live-in
      }
      WJAddOsrTarget(uint32_t(rp->pc() - info.script()->code()), idx);
      if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(blockAddr)) ||
          !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0) ||
          !e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(idx)) ||
          !e.writeOp(Op::I32Eq))
        return false;
      if (!e.writeOp(Op::If) || !e.writeFixedU8(kVoid)) return false;  // if block match
      uint32_t nargs = info.nargs(), nlocals = info.nlocals(), vi = 0;
      // OSR spill/restore-correspondence dump (GECKO_WJ_OSRMAPDBG): per target
      // block, log each slot's def opcode/type/target-local so a wrong restore
      // (the OSR "table index OOB" crash on complex blocks) can be pinned by diffing
      // against EmitDeoptResumeInline's spill. Compile-time only (no runtime cost).
      static int osrMapDbg = WJGETENV("GECKO_WJ_OSRMAPDBG") ? 1 : 0;
      if (osrMapDbg) {
        uint32_t li = 0;
        auto logSlot = [&](const char* kind, uint32_t si, MDefinition* d) {
          fprintf(stderr, "[wj-osrmap]   %s[%u] vi=%u def=%s ty=%s local=%d\n", kind,
                  si, li++, WJOpName(d->op()), StringFromMIRType(d->type()),
                  be.local(d));
        };
        fprintf(stderr,
                "[wj-osrmap] %s:%u blk=%u loopHdr=%d nargs=%u nlocals=%u\n",
                info.script()->filename() ? info.script()->filename() : "?",
                unsigned(info.script()->lineno()), idx, b->isLoopHeader() ? 1 : 0,
                nargs, nlocals);
        logSlot("this", 0, rp->getOperand(info.thisSlot()));
        for (uint32_t i = 0; i < nargs; i++) logSlot("arg", i, rp->getOperand(info.argSlot(i)));
        for (uint32_t i = 0; i < nlocals; i++) logSlot("loc", i, rp->getOperand(info.localSlot(i)));
      }
      if (!EmitOsrLoadSlot(e, be, rp->getOperand(info.thisSlot()), vi++)) return false;
      for (uint32_t i = 0; i < nargs; i++)
        if (!EmitOsrLoadSlot(e, be, rp->getOperand(info.argSlot(i)), vi++))
          return false;
      for (uint32_t i = 0; i < nlocals; i++)
        if (!EmitOsrLoadSlot(e, be, rp->getOperand(info.localSlot(i)), vi++))
          return false;
      if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(idx)) ||
          !e.writeOp(Op::LocalSet) || !e.writeVarU32(bidLocal))
        return false;  // bidLocal = idx
      if (!e.writeOp(Op::End)) return false;  // end if (block match)
    }
    if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(activeAddr)) ||
        !e.writeOp(Op::I32Const) || !e.writeVarS32(0) || !e.writeOp(Op::I32Store) ||
        !e.writeVarU32(2) || !e.writeVarU32(0))
      return false;  // gWJOsrActive = 0
    if (!e.writeOp(Op::End)) return false;  // end if (gWJOsrActive)
  }

  if (be.oolDeopt) {
    if (!e.writeOp(Op::Block) || !e.writeFixedU8(kVoid)) return false;  // $D
  }

  // Dispatch loop: loop $L { block $b0 {...{ block $b_{n-1} { br_table }}}}.
  if (!e.writeOp(Op::Loop) || !e.writeFixedU8(kVoid)) return false;
  for (uint32_t i = 0; i < n; i++) {
    if (!e.writeOp(Op::Block) || !e.writeFixedU8(kVoid)) return false;
  }
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(bidLocal)) return false;
  if (!e.writeOp(Op::BrTable) || !e.writeVarU32(n)) return false;
  for (uint32_t i = 0; i < n; i++) {
    if (!e.writeVarU32(n - 1 - i)) return false;
  }
  if (!e.writeVarU32(0)) return false;  // default

  for (uint32_t ri = 0; ri < n; ri++) {
    uint32_t bi = n - 1 - ri;
    MBasicBlock* b = blocks[bi];
    if (!e.writeOp(Op::End)) { js::wasm::gWJBailReason = "dispatch-blockclose-enc"; return false; }  // close block $b_{bi}
    uint32_t loopDepthHere = n - 1 - ri;
    be.bodyLoopIdx = loopDepthHere;  // br index of $L from this body base
    be.inDispatchBody = be.oolDeopt;  // in-body guards go out-of-line
    if (!EmitBlockBody(e, be, b)) {
      if (WJGETENV("GECKO_WJ_CFINFO")) {
        JSScript* s = be.info ? be.info->script() : nullptr;
        fprintf(stderr, "[wj-blockbody-bail] %s:%u block bi=%u id=%u reason=%s\n",
                s ? s->filename() : "?", s ? unsigned(s->lineno()) : 0, bi, b->id(),
                js::wasm::gWJBailReason ? js::wasm::gWJBailReason : "?");
      }
      return false;
    }
    be.inDispatchBody = false;  // terminator deopts stay inline (at body base)
    // alwaysBails blocks emitted their own deopt (which exits); no terminator.
    if (b->alwaysBails() || WJBlockForceDeopt(be, b)) continue;
    if (!EmitTerminator(e, be, blockIdx, bidLocal, loopDepthHere, b)) {
      js::wasm::gWJBailReason = "terminator";
      return false;
    }
  }

  if (!e.writeOp(Op::Unreachable)) { js::wasm::gWJBailReason = "dispatch-postloop-enc"; return false; }
  if (!e.writeOp(Op::End)) { js::wasm::gWJBailReason = "dispatch-endloop-enc"; return false; }  // end loop $L

  // Out-of-line deopt dispatcher: `br $D` from a guard lands here (just past the
  // loop). Switch on deoptSiteLocal and run the matching site's full resume.
  if (be.oolDeopt) {
    if (!e.writeOp(Op::End)) return false;  // end $D
    for (uint32_t s = 0; s < be.deoptSites.size(); s++) {
      if (!GetLocal(e, be.deoptSiteLocal) || !e.writeOp(Op::I32Const) ||
          !e.writeVarS32(int32_t(s)) || !e.writeOp(Op::I32Eq) ||
          !e.writeOp(Op::If) || !e.writeFixedU8(kVoid))
        return false;
      be.curRp = be.deoptSites[s].first;
      be.curOp = be.deoptSites[s].second;
      if (!EmitDeoptResumeInline(e, be)) {
        if (WJGETENV("GECKO_WJ_DEOPTRESUMEDBG"))
          fprintf(stderr,
                  "[wj-deopt-resume-fail] site=%u curRp=%p curOp=%u(%s) "
                  "rpPc=%s reason=%s\n",
                  s, (void*)be.curRp, be.curOp,
                  WJOpName(jit::MDefinition::Opcode(be.curOp)),
                  be.curRp ? js::CodeName(JSOp(*be.curRp->pc())) : "?",
                  js::wasm::gWJBailReason ? js::wasm::gWJBailReason : "?");
        if (!js::wasm::gWJBailReason ||
            !strstr(js::wasm::gWJBailReason, "resume"))
          js::wasm::gWJBailReason = "deopt-resume";
        return false;  // ends in `return`
      }
      if (!e.writeOp(Op::End)) return false;
    }
  }

  if (!e.writeOp(Op::Unreachable)) return false;
  cfReport("dispatch");
  return e.writeOp(Op::End);  // end function body
}
