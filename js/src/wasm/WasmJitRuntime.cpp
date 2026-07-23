/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*-
 * vim: set ts=8 sts=2 et sw=2 tw=80:
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

// WasmJitRuntime.cpp -- runtime integration for the re-architected JS->wasm JIT:
// the compile trigger (delays until ICs warm, then drives WasmJitWarp), call
// routing + arg/result marshalling through gWJScratch, the wjhelp/wasmjit_invoke
// trampolines, and GC root tracing of the scratch buffer.

#include "mozilla/ScopeExit.h"
#include "mozilla/TimeStamp.h"  // GECKO_WJ_COMPILESTAT tier-up compile cost

#include "wasm/WasmJit.h"
#include "wasm/WasmJitBackend.h"  // kWJResultSlot / kWJThisSlot / kWJScratchSlots

#include <stdint.h>
#include <stdlib.h>
#include <unordered_map>
#include <utility>
#include <map>
#include <vector>
#include <string>
#include <algorithm>

#include "js/CallAndConstruct.h"  // JS::Call
#include "js/PropertyAndElement.h"  // JS_GetProperty
#include "js/GCAPI.h"
#include "js/RootingAPI.h"
#include "js/Value.h"
#include "vm/JSContext.h"
#include "vm/JSFunction.h"
#include "vm/ArgumentsObject.h"  // ArgumentsObject::createForWasmJit (WJH_NEWARGUMENTS)
#include "builtin/Object.h"      // js::ObjectClassToString (WJH_OBJCLASSTOSTRING)
#include "builtin/MapObject.h"   // js::MapObject::create / js::SetObject::create (WJH_NEWMAP/NEWSET)
#include "jit/JitScript.h"
#include "jit/JitScript-inl.h"  // inline AutoKeepJitScripts ctor/dtor (WJH_RESUME JitScript recreate)
#include "jit/VMFunctions.h"  // CreateThisFromIon
#include "js/experimental/JitInfo.h"  // JSJitGetterOp, JSJitGetterCallArgs (WJH_GETDOMPROP)
#include "vm/JSScript.h"
#include "vm/JSAtomUtils.h"  // js::Atomize
#include "vm/BytecodeUtil.h"  // js::CodeName
#include "vm/NativeObject.h"
#include "vm/PlainObject.h"  // js::PlainObject::createWithShape
#include "vm/EnvironmentObject.h"  // js::CallObject::createWithShape
#include <cstdarg>  // WJStatsJSON (task #57 __wjStats)
#include "vm/Scope.h"  // js::LexicalScope (WJH_NEWLEXENV)
#include "vm/Interpreter.h"  // SetObjectElement, InstanceofOperator, LessThan, ...
#include "vm/EqualityOperations.h"  // LooselyEqual, StrictlyEqual
#include "vm/Stack.h"  // ConstructArgs
#include "vm/ProxyObject.h"
#include "gc/Barrier.h"  // gc::ValuePreWriteBarrier
#include "gc/Tracer.h"   // js::TraceRoot
#include "vm/Shape.h"    // js::Shape
#include "vm/ArrayObject.h"   // js::ArrayObject
#include "builtin/Array.h"    // js::SetLengthProperty, IsArrayFromJit
#include "vm/RegExpObject.h"  // js::CloneRegExpObject (MRegExp)
#include "builtin/String.h"   // js::StringFromCharCode
#include "builtin/RegExp.h"   // js::RegExpHasCaptureGroups
#include "vm/TypedArrayObject.h"  // js::NewTypedArrayWithTemplateAndLength
#include "js/Conversions.h"    // JS::ToInt32
#include "builtin/Number.h"   // js::NumberParseInt
#include "builtin/Math.h"     // js::GetUnaryMathFunctionPtr (MMathFunction)
#include "vm/StringType.h"    // js::ToString, js::EqualStrings
#include "js/friend/DumpFunctions.h"  // js::DumpBacktrace (GECKO_WJ_VALIDATEHELPER)
#include "vm/ObjectOperations.h"  // js::HasProperty, js::HasOwnProperty
#include "vm/Iteration.h"     // js::ValueToIterator, NativeIterator, PropertyIteratorObject
#include "util/Unicode.h"     // js::unicode::ToLowerCase/ToUpperCase (CharCodeConvertCase)
#include "vm/Interpreter-inl.h"  // js::CheckPrivateFieldOperation (CheckPrivateFieldCache)
#include "vm/BoundFunctionObject.h"  // js::BoundFunctionObject::createWithTemplate (NewBoundFunction)

#include "vm/NativeObject-inl.h"
#include "vm/ObjectOperations-inl.h"  // js::GetProperty(obj,receiver,id) (WJH_GETPROPSUPER)
#include "vm/AsyncFunction.h"  // js::AsyncFunctionResolve (WJH_ASYNCRESOLVE)
#include "builtin/Promise.h"  // js::CanSkipAwait / js::ExtractAwaitValue (await-skip)
#include "vm/PlainObject-inl.h"  // createWithShape
#include "vm/Interpreter-inl.h"  // GetElementOperation
#include "vm/JSObject-inl.h"     // GuessArrayGCKind

#if defined(__EMSCRIPTEN__)
#  include <emscripten.h>
#else
#  define EMSCRIPTEN_KEEPALIVE
#endif

using namespace js;
using namespace js::wasm;

extern "C" {
double wasmhost_call(int handle, int index, const double* args, int argc);
}

namespace js {
namespace wasm {
// Front-end + compile orchestrator (WasmJitWarp.cpp).
int WJWarpCompile(JSContext* cx, JSScript* script, uint32_t* nargsOut,
                  uint32_t* nlocalsOut, int* tblSlotOut);
}  // namespace wasm
namespace pbl {
// Resume a partially-executed JIT'd function in the Portable Baseline Interpreter
// at `pcOff`, seeding fixed locals from osrLocals[] and the expr stack from osrStack[].
extern bool WasmJitResumeViaPBL(JSContext* cx, JSScript* script, uint64_t thisBits,
                                const JS::Value* args, uint32_t argc,
                                JSObject* envChain, const uint64_t* osrLocals,
                                uint32_t nLocals, uint32_t pcOff, uint64_t* retBits,
                                const uint64_t* osrStack, uint32_t osrStackDepth,
                                JSObject* enclosingEnv = nullptr,
                                bool keepFrameEnv = false,
                                bool resumeInError = false,
                                JSFunction* runtimeCallee = nullptr);
}  // namespace pbl
}  // namespace js

// --- Resume state (set partly by RunCall, partly by the emitted spill code) ---
// These five are declared in WasmJitBackend.h (js::wasm) and read by the emitted
// code's addresses, so their definitions must live in js::wasm too.
namespace js {
namespace wasm {
uint64_t gWJResumeVals[1024];   // all frames' [this, args.., locals.., stack..]
// gWJResumeVals holds boxed GC pointers spilled at a deopt, read back by WJH_RESUME.
// Between the spill (emitted code) and the read, WJH_RESUME allocates (RootedValueVector
// reserve) which can trigger a minor GC -> the spilled pointers would go stale unless
// traced. So WJTraceRoots traces gWJResumeVals[0..gWJResumeValsCount] while a resume is
// in progress (gWJResumeActive). (The deltablue-under-inlining heisenbug.)
bool gWJResumeActive = false;
uint32_t gWJResumeValsCount = 0;
uint32_t gWJResumeNFrames = 0;             // inline chain length (1 = no inlining)
uint32_t gWJResumePc[kWJMaxResumeFrames] = {0};
uint32_t gWJResumeStackDepth[kWJMaxResumeFrames] = {0};
uint32_t gWJResumeScriptPtr[kWJMaxResumeFrames] = {0};
uint32_t gWJResumeEnvPtr[kWJMaxResumeFrames] = {0};
uint32_t gWJResumeEnclosingEnv[kWJMaxResumeFrames] = {0};
// The RUNTIME callee function for the resumed frame (0 -> canonical fallback).
// Closures sharing a script differ from script->function(); resuming with the
// canonical made JSOp::Callee push a null-env function (acorn-walk NFE).
uint32_t gWJResumeCalleeFn[kWJMaxResumeFrames] = {0};
uint32_t gWJLastDeoptOp = 0;  // MIR opcode of the most recent deopt site (see gggDeopts)
// try/catch: set to 1 by emitted code before a WJH_RESUME that is an in-try-region
// exception deopt (the JS exn is pending on cx). WJH_RESUME passes it to
// WasmJitResumeViaPBL -> gPBLResumeInError -> PBL `goto error` -> HandleException runs
// the catch. Cleared by WJH_RESUME after reading.
uint32_t gWJResumeInError = 0;
uint32_t gWJDeoptByOp[js::wasm::kWJNumOps] = {0};
uint32_t gWJResumeNArgs[kWJMaxResumeFrames] = {0};
uint32_t gWJResumeNLocals[kWJMaxResumeFrames] = {0};
uint32_t gWJResumeValsOff[kWJMaxResumeFrames] = {0};
uint64_t gWJResumeActuals[js::wasm::kWJMaxArgs] = {0};  // see WasmJitBackend.h
uint32_t gWJExitFPLastSite = 0;  // task #60 tracer: 1=VMFrame 2=nativeTramp 3=jitFastCall
uint32_t gWJExitFPSetCount = 0;
// Known-good exitFP for the CURRENT JIT invocation (task #60): set by
// WasmJitRunCall from the activation's exitFP at entry (the PBL fast-call site
// establishes a fresh CallNative exit frame just before, valid for the whole
// call). wjhelp re-installs it on entry so a GC DURING any helper walks a valid
// chain even after an exception unwind rewound packedExitFP. Save/restore
// disciplined around the trampoline call (nesting-safe).
static uint8_t* gWJEntryExitFP = nullptr;
static void* gWJEntryAct = nullptr;
extern "C" uint32_t WJExitFPDbgState(uint32_t* cnt) {
  if (cnt) *cnt = gWJExitFPSetCount;
  return gWJExitFPLastSite;
}
// Record an exitFP install from OUTSIDE the wasm TU (FrameIter unwind sites).
extern "C" void WJExitFPDbgRecord(int siteId, void* v) {
  static const char* e = getenv("GECKO_WJ_EXITFPDBG");
  static uintptr_t target = e ? strtoul(e, nullptr, 0) : 0;
  if (target && uintptr_t(v) == target) {
    gWJExitFPLastSite = uint32_t(siteId);
    gWJExitFPSetCount++;
  }
}
uint32_t gWJResumeActualArgc = 0;
uint64_t gWJCallCallee = 0;     // boxed callee Value (set by emitted code)
uint32_t gWJCallArgc = 0;
uint64_t gWJConstructNewTarget = 0;  // boxed newTarget for constructing calls
uintptr_t gWJMarkBarrierAddr = 0;  // baked zone needs-marking-barrier flag address
uintptr_t gWJWholeCellLastAddr = 0;  // baked &storeBuffer.bufferWholeCell.last_
uint64_t gWJGlobalLexEnvVal = 0;   // boxed global lexical env (for FunctionEnvironment)
uint32_t gWJCurrentEnv = 0;        // current fn's runtime environment (raw ptr)
// OSR cheap-resume (GECKO_WJ_OSR): when a deopt resumes at a loop-head, instead of
// interpreting the rest of the fn in PBL, the runtime sets gWJOsrActive=1 +
// gWJOsrBlock=<dispatch block index of the loop head> and re-calls the JIT'd fn; its
// prologue, seeing gWJOsrActive, loads the frame locals from gWJResumeVals and starts
// the dispatch loop at gWJOsrBlock (instead of block 0). Cleared by the prologue.
uint32_t gWJOsrActive = 0;
uint32_t gWJOsrBlock = 0;
uint32_t gWJOsrDepth = 0;          // nested OSR re-entries on the stack (churn guard)
uint64_t gWJOsrHits = 0;           // count of OSR re-entries (GECKO_WJ_OSRDBG)
// Per-compile sink the backend appends OSR targets to: (resume pcOff -> dispatch
// block index) for each OSR-able loop head. Cleared before each WJWarpCompile,
// copied into the WJEntry on success.
static std::vector<std::pair<uint32_t, uint32_t>> gWJPendingOsrTargets;
extern "C" void WJAddOsrTarget(uint32_t pcOff, uint32_t blockIdx) {
  gWJPendingOsrTargets.emplace_back(pcOff, blockIdx);
}
int gWJExecDepth = 0;              // >0 while emitted wasm-JIT code is on the stack
                                   // (drives inWasmJit()/inJit()/inIon() in the shell)
bool gWJHadAlwaysBails = false;    // last compile emitted an alwaysBails deopt block
uint32_t gWJEmitShapeDeopts = 0;   // last compile: # GuardShape-family deopt sites
uint32_t gWJEmitTotalDeopts = 0;   // last compile: # total deopt sites
bool gWJForceMega = false;         // megamorphic recompile (set per-compile by the valve)
bool gWJForceNumberArith = false;  // de-speculate Int32 arith/elem (set per-compile by the valve)
uint64_t gWJCompileAttempts = 0;
uint64_t gWJCompileOK = 0;
double gWJCompileMs = 0.0;
double gWJCompileOKMs = 0.0;
double gWJHostCompileMs = 0.0;
double gWJHostInstMs = 0.0;
uint64_t gWJEmitBytes = 0;
int64_t gWJTraceVal = 0;  // per-def value tracer staging (GECKO_WJ_VALUETRACE)
double gWJSnapshotMs = 0.0;
double gWJBuildMs = 0.0;
double gWJOptimizeMs = 0.0;
const char* gWJBailReason = "unknown";  // why the last WJEmitBody returned false
uint32_t gWJBailLine = 0;          // source line of the last bailed function
uint32_t gWJBailOpLine = 0;        // source line of the specific bailed op
uint32_t gWJBailOpOff = 0;         // bytecode offset of the specific bailed op
const char* gWJBailOpFile = nullptr;  // (inlinee) script filename of the bailed op
uint32_t gWJNewShapeSlot = 0;      // alloc helper: shape pool index
uint32_t gWJNewAux = 0;            // alloc helper: allocKind or array length
uint32_t gWJNewHeap = 0;           // alloc helper: gc::Heap
uint32_t gWJNewObjScript = 0;      // WJH_NEWOBJECT: JSScript* (raw)
uint32_t gWJNewObjPcOff = 0;       // WJH_NEWOBJECT: bytecode offset
uint32_t gWJLexScope = 0;          // WJH_NEWLEXENV: LexicalScope* (raw)
uint32_t gWJVarScope = 0;          // WJH_NEWVARENV: VarScope* (raw)
uintptr_t gWJNurseryPosAddr = 0;   // address of zone nursery position_ (inline alloc)
uintptr_t gWJObjHeaderWord = 0;    // NurseryCellHeader value for Object cells
uintptr_t gWJStringHeaderWord = 0; // NurseryCellHeader value for String cells (rope-concat inline)
uint32_t gWJHelpObj = 0;        // object ptr for WJH_SETSLOT
uint32_t gWJHelpSlot = 0;
uint64_t gWJHelpVal = 0;        // boxed value for WJH_SETSLOT
uint64_t gWJPreBarBadSkips = 0; // WJH_PREBARRIER: count of skipped invalid/garbage old-value cells (a store handed the barrier a bad ptr -- see discord.com pre-barrier crash)
uint64_t gWJElemHits = 0;       // DEBUG: inline typed-array element-store IC hits
uint32_t gWJCallFn[kWJCallSites * kWJCallWays];     // polymorphic call IC: callee ptrs
int32_t gWJCallTblIdx[kWJCallSites * kWJCallWays];  // polymorphic call IC: table slots
uint32_t gWJCallSiteLine[kWJCallSites] = {0};       // DEBUG: caller script line per site
static uint32_t gWJNextCallSite = 0;
uint32_t WJAllocCallSite() {
  if (gWJNextCallSite >= kWJCallSites) return 0;  // site 0 is a safe sentinel
  return gWJNextCallSite++;
}

// Per-construct-site monomorphic inline cache (GECKO_WJ_CTORINLINE). On a hit the
// backend inline-allocates `this` (cached shape/size/nfixed) + call_indirects the
// ctor directly, eliminating the 8.87M/run WJH_CONSTRUCT boundary crossings (earley
// cons cells). Filled by WJH_CONSTRUCT on the slow path. site 0 = sentinel (never
// cached). Callee/shape are GC ptrs -> traced in WJTraceRoots.
uint32_t gWJCtorCallee[kWJCtorSites] = {0};  // cached ctor JSFunction ptr (0 = empty)
uint32_t gWJCtorShape[kWJCtorSites] = {0};   // cached `this` SharedShape ptr
uint32_t gWJCtorSize[kWJCtorSites] = {0};    // total nursery cell size (header+thing)
uint32_t gWJCtorNfixed[kWJCtorSites] = {0};  // fixed slot count (init to undefined)
int32_t gWJCtorTblIdx[kWJCtorSites] = {0};   // ctor's shared-table index (-1 = none)
uint32_t gWJCtorEnv[kWJCtorSites] = {0};     // ctor's environment ptr
static uint8_t gWJCtorNoFill[kWJCtorSites] = {0};  // 1 = site can never inline-fill
                                                   // (forwarding wrapper) -> stop the
                                                   // per-construct fill-attempt churn
static uint32_t gWJNextCtorSite = 0;
uint32_t WJAllocCtorSite() {
  if (gWJNextCtorSite >= kWJCtorSites) return 0;
  return ++gWJNextCtorSite;  // site 0 reserved as sentinel
}

// Inline property-load IC (see WasmJitBackend.h).
uint32_t gWJPropShape[kWJPropSites * kWJPropWays];
uint32_t gWJPropOff[kWJPropSites * kWJPropWays];
uint32_t gWJPropWayKey[kWJPropSites * kWJPropWays];  // store-IC way key atom (see .h)
// ADD-IC: per-site cached fixed-slot property-ADD transition (oldShape -> newShape
// at a fixed slot). gWJAddOld/NewShape hold POOL ADDRESSES into gWJShapePool (traced
// + GC-current), so the cached newShape is safe to write into an object. 0 = unset.
uint32_t gWJAddOldShape[kWJPropSites] = {0};
uint32_t gWJAddNewShape[kWJPropSites] = {0};
uint32_t gWJAddOff[kWJPropSites] = {0};  // added prop's fixed-slot byte offset (from obj)
uint32_t gWJAddKey[kWJPropSites] = {0};  // added key's atom (see .h)
uint32_t gWJPropHolder[kWJPropSites * kWJPropWays];  // proto-read holder (0 = own)
uint64_t gWJPropKey[kWJPropSites];
uint8_t gWJPropStrict[kWJPropSites];
static uint32_t gWJNextPropSite = 1;  // site 0 is a never-filled sentinel
uint32_t WJAllocPropSite() {
  if (gWJNextPropSite >= kWJPropSites) return 0;
  return gWJNextPropSite++;
}
// Keyed allocation: the SAME read (identified by a stable (script, read-index) key)
// REUSES its site across recompiles instead of consuming a fresh one each time.
// Without this, the deopt-storm valve's repeated mega-recompiles leak sites until
// the pool exhausts -- then a late recompile's read can't get a site, can't convert
// to a by-name EmitPropIC, and (with its shape guard already removed) reads the
// WRONG fixed slot for a polymorphic receiver (deltablue 741 "Cycle"). Reuse bounds
// the pool to the number of distinct read sites in the program. Bonus: the reused
// site keeps its warm shape cache across recompiles.
static std::unordered_map<uint64_t, uint32_t>* gWJPropSiteMap = nullptr;
uint32_t WJAllocPropSiteKeyed(uint64_t key) {
  if (!gWJPropSiteMap) gWJPropSiteMap = new std::unordered_map<uint64_t, uint32_t>();
  auto it = gWJPropSiteMap->find(key);
  if (it != gWJPropSiteMap->end()) return it->second;
  if (gWJNextPropSite >= kWJPropSites) return 0;
  uint32_t s = gWJNextPropSite++;
  (*gWJPropSiteMap)[key] = s;
  return s;
}

// GetName IC site state (see WasmJitBackend.h).
uintptr_t gWJNameHolder[kWJNameSites];
uint32_t gWJNameShape[kWJNameSites];
uint32_t gWJNameOff[kWJNameSites];
uint64_t gWJNameKey[kWJNameSites];
static uint32_t gWJNextNameSite = 1;  // site 0 is a never-allocated sentinel
uint32_t WJAllocNameSite() {
  if (gWJNextNameSite >= kWJNameSites) return 0;
  return gWJNextNameSite++;
}
void WJClearPropIC() {
  memset(gWJPropShape, 0, sizeof(gWJPropShape));
  memset(gWJPropWayKey, 0, sizeof(gWJPropWayKey));
  memset(gWJAddOldShape, 0, sizeof(gWJAddOldShape));
  memset(gWJAddKey, 0, sizeof(gWJAddKey));
}
}  // namespace wasm
}  // namespace js

// Argument/result transfer buffer in the guest heap (a fixed global => a stable
// address the emitted wasm can compute from). Args at [0..nargs), `this` at
// [kWJThisSlot], result at [kWJResultSlot]. GC-traced by WJTraceRoots.
alignas(8) uint64_t gWJScratch[js::wasm::kWJScratchSlots];

uint64_t gWJWasmRuns = 0;
uint64_t gWJFastCalls = 0;
uint64_t gWJSlowCalls = 0;
uint64_t gWJWasmDeopts = 0;
// Set by WJH_RESUME; checked by WasmJitRunCall to tell a deopted entry from one
// that ran fully in JIT (both return flag 0), driving the deopt-storm safety valve.
static bool gWJDidResume = false;

// GC-constant pool: boxed JS::Values (object/string constants) baked into JIT'd
// modules. Traced + relocated in place by WJTraceRoots so the emitted code,
// which loads pool[i] at runtime, always sees the live pointer.
// 2026-07-02: bumped 4096 -> 65536. These pools are shared across ALL functions
// compiled in the process (traced+permanent, can't reset), so a large real app
// (uglify-js etc.) exhausts 4096 and then EVERY subsequent shape/constant-baking
// op (GuardShape, NewCallObject, GuardShapeListToOffset...) bails to PBL. Tracing
// cost is O(count) not O(size), so a bigger array is free until actually used.
static constexpr uint32_t kWJConstPoolSize = 65536;
alignas(8) uint64_t gWJConstPool[kWJConstPoolSize];
static uint32_t gWJConstPoolCount = 0;

uintptr_t js::wasm::WJInternConstant(uint64_t valueBits) {
  // Dedupe (pools are shared across all compiled functions for the process).
  for (uint32_t i = 0; i < gWJConstPoolCount; i++) {
    if (gWJConstPool[i] == valueBits) {
      return uintptr_t(static_cast<void*>(&gWJConstPool[i]));
    }
  }
  if (gWJConstPoolCount >= kWJConstPoolSize) {
    static bool warned = false;
    if (!warned) { warned = true; fprintf(stderr, "[wj-pool] CONST POOL FULL at %u\n", kWJConstPoolSize); }
    return 0;
  }
  uint32_t i = gWJConstPoolCount++;
  static int poolstat = getenv("GECKO_WJ_POOLSTAT") ? 1 : 0;
  if (poolstat && (gWJConstPoolCount % 1024) == 0)
    fprintf(stderr, "[wj-pool] const count=%u\n", gWJConstPoolCount);
  gWJConstPool[i] = valueBits;
  return uintptr_t(static_cast<void*>(&gWJConstPool[i]));
}

// Shape pool: GuardShape bakes the address of a pool slot and loads the CURRENT
// shape pointer from it at runtime. WJTraceRoots traces + relocates each slot, so
// when a compacting GC moves a shape the guard still compares against the live
// pointer instead of a stale one (which would deopt-storm -- crypto's am3 was
// deopting ~475k times because its `this.array` shape moved post-compile).
static constexpr uint32_t kWJShapePoolSize = 65536;  // was 4096 (see const-pool note)
uintptr_t gWJShapePool[kWJShapePoolSize];
static uint32_t gWJShapePoolCount = 0;

uintptr_t js::wasm::WJInternShape(uintptr_t shapeBits) {
  for (uint32_t i = 0; i < gWJShapePoolCount; i++) {
    if (gWJShapePool[i] == shapeBits) {
      return uintptr_t(static_cast<void*>(&gWJShapePool[i]));
    }
  }
  if (gWJShapePoolCount >= kWJShapePoolSize) {
    static bool warned = false;
    if (!warned) { warned = true; fprintf(stderr, "[wj-pool] SHAPE POOL FULL at %u\n", kWJShapePoolSize); }
    return 0;
  }
  uint32_t i = gWJShapePoolCount++;
  static int poolstat = getenv("GECKO_WJ_POOLSTAT") ? 1 : 0;
  if (poolstat && (gWJShapePoolCount % 1024) == 0)
    fprintf(stderr, "[wj-pool] shape count=%u\n", gWJShapePoolCount);
  gWJShapePool[i] = shapeBits;
  return uintptr_t(static_cast<void*>(&gWJShapePool[i]));
}

// Deopt-resume SCRIPT pool: raw JSScript* cells, traced+RELOCATED by WJTraceRoots
// (WJTracePtrRoot<JSScript>). The deopt spill interns info.script() here at compile
// time and the emitted code LOADS this slot at runtime -> spills the GC-CURRENT script
// into gWJResumeScriptPtr, so a compacting GC that moved the script can't leave a stale
// baked pointer (the crypto/pako deopt-resume-under-compaction OOB). Mirrors gWJShapePool.
static constexpr uint32_t kWJScriptPoolSize = 8192;
uintptr_t gWJScriptPool[kWJScriptPoolSize];
uint32_t gWJScriptPoolCount = 0;

uintptr_t js::wasm::WJInternScript(uintptr_t scriptBits) {
  for (uint32_t i = 0; i < gWJScriptPoolCount; i++) {
    if (gWJScriptPool[i] == scriptBits) {
      return uintptr_t(static_cast<void*>(&gWJScriptPool[i]));
    }
  }
  if (gWJScriptPoolCount >= kWJScriptPoolSize) {
    static bool warned = false;
    if (!warned) { warned = true; fprintf(stderr, "[wj-pool] SCRIPT POOL FULL at %u\n", kWJScriptPoolSize); }
    return 0;
  }
  uint32_t i = gWJScriptPoolCount++;
  gWJScriptPool[i] = scriptBits;
  return uintptr_t(static_cast<void*>(&gWJScriptPool[i]));
}

namespace {

struct WJEntry {
  enum class State : uint8_t { Cold, Compiled, Failed };
  State state = State::Cold;
  int handle = -1;
  int tblSlot = -1;  // dense shared-table slot (-1 = not call_indirect-able)
  int directIdx = -1;  // slot in MAIN indirect table for direct PBL->JIT entry
  uint32_t jitRuns = 0;  // entries that ran fully in JIT (no deopt)
  uint32_t deopts = 0;   // entries that deopted to PBL (resume); RESET on recompile
  uint32_t deoptsTotal = 0;  // lifetime deopts, never reset (matches gWJDeoptByOp scale)
  uint32_t recompiles = 0;  // deopt-storm-triggered recompiles so far
  uint32_t gggDeopts = 0;   // GuardGlobalGeneration deopts (global redefined since
                            // compile). Storm control: >=4 -> respec (fresh
                            // generation), >=10 -> Failed (permanent PBL). Bundle
                            // INIT phases bump the generation between every call;
                            // without this the per-entry deopt->resume->PBL->JIT
                            // nesting exhausted the V8 stack UNCATCHABLY (webtooling).
  bool hasAlwaysBails = false;  // compiled fn has a cold-IC alwaysBails deopt block
  bool shapeDeoptDom = false;   // compiled fn's deopt sites are GuardShape-family dominated
  uint32_t nargs = 0;
  uint32_t nlocals = 0;
  uint32_t observes = 0;
  uint32_t nextTry = 0;  // observe count at which to (re)attempt compilation
  uint32_t fails = 0;    // failed attempts so far
  // After a GuardShape deopt-storm, recompiling immediately reproduces the same
  // MONOMORPHIC guard (the fn compiled before PBL saw the receiver's other
  // shapes). Force it to run in PBL for this many more observes first, so the
  // property ICs accumulate the polymorphic shapes -> the recompile reads a
  // polymorphic IC and Warp emits GuardShapeList (handled, no deopt) instead.
  uint32_t recompileFloor = 0;
  // Set after a (non-alwaysBails) deopt storm: the next recompile uses megamorphic
  // property codegen (multi-shape EmitPropIC, no deopt) for GuardShape-guarded
  // reads -- fixes a monomorphic GuardShape storming on a polymorphic receiver.
  bool forceMega = false;
  // Set when the Ion-style count trigger has spent its recompile attempt on a
  // MODERATE-rate (<=50%) storm and it did NOT heal: stop re-triggering and keep the
  // current JIT version (it's net-positive-ish; failing to PBL would cost the JIT<->PBL
  // boundary, measured WORSE than the residual deopts). Only >50% storms fail to PBL.
  bool recompileDone = false;
  // OSR cheap-resume targets: (resume pcOff -> dispatch block index) for each
  // OSR-able loop head, recorded at compile time (GECKO_WJ_OSR). A single-frame
  // deopt landing at one of these pcs re-enters the JIT there instead of PBL.
  std::vector<std::pair<uint32_t, uint32_t>> osrTargets;
  // DEFERRED COMPILE (GECKO_WJ_DEFERCOMPILE): true while this script sits in the
  // pending-compile queue (crossed the warmup threshold but not yet compiled).
  bool queued = false;
  // # of observes seen while queued but not yet drained. If a drain never happens
  // (a compute-bound page that never idles, or a harness with no task boundary),
  // this climbs; past kWJDeferStrandLimit we compile SYNCHRONOUSLY so a hot fn is
  // never stranded in PBL forever. Loads drain at a task boundary well before this.
  uint32_t queuedObserves = 0;
};

// Per-script compile state. Keyed by JSScript*; entries persist for the process.
static std::unordered_map<JSScript*, WJEntry>* gEntries = nullptr;

// DEFERRED-COMPILE queue: scripts that crossed the warmup threshold but whose
// synchronous compile was deferred OFF the current (load) critical path. Drained by
// WasmJitDrainDeferred() at an idle / task boundary (the embed calls it). GC-traced
// in WJTraceRoots so queued scripts stay live + relocated across the defer window.
static std::vector<JSScript*> gWJDeferQueue;
// True while draining: makes WasmJitObserveCall skip the enqueue path and compile
// inline (the drain reuses the same observe->compile logic without re-deferring).
static bool gWJDrainingNow = false;

static WJEntry& EntryFor(JSScript* script) {
  if (!gEntries) gEntries = new std::unordered_map<JSScript*, WJEntry>();
  // Direct-mapped cache: a 1-entry cache thrashed on uBlock's alternating callees ->
  // the operator[] emplace showed ~0.9% in the profile. Element addresses are stable
  // across rehashes, so caching WJEntry* is safe. 8192-way (was 512): typescript's
  // working set is ~2500 scripts (1409 compiled + ~1017 cold), which 5x-oversubscribed
  // a 512-slot cache -> collision thrash on hot alternating callees; 8192 slots holds
  // the set at ~0.3 load (2026-07-11 profile follow-up; 128KB, negligible).
  static const uint32_t kECBits = 13;
  static const uint32_t kECMask = (1u << kECBits) - 1;
  static JSScript* sECScript[1u << kECBits] = {};
  static WJEntry* sECEntry[1u << kECBits] = {};
  uint32_t h = uint32_t(uintptr_t(script) >> 3) & kECMask;
  if (sECScript[h] == script && sECEntry[h]) return *sECEntry[h];
  WJEntry& e = (*gEntries)[script];
  sECScript[h] = script;
  sECEntry[h] = &e;
  return e;
}

// GECKO_WJ_ENTRYDUMP: dump every WJEntry with deopts >= threshold -- script
// filename:line, deopts, jitRuns, deopt-rate%, recompiles, final state. Tells us
// which functions storm-in-JIT (Compiled + high deopts) vs failed-to-PBL.
static void WJDumpEntries() {
  if (!gEntries) return;
  int thr = getenv("GECKO_WJ_ENTRYDUMP") ? atoi(getenv("GECKO_WJ_ENTRYDUMP")) : 0;
  if (thr <= 0) thr = 1000;
  fprintf(stderr, "[wj-entrydump] (deopts>=%d):\n", thr);
  for (auto& kv : *gEntries) {
    JSScript* sc = kv.first;
    WJEntry& e = kv.second;
    if (e.deopts < uint32_t(thr)) continue;
    uint64_t tot = uint64_t(e.deopts) + e.jitRuns;
    int rate = tot ? int((100ull * e.deopts) / tot) : 0;
    const char* st = e.state == WJEntry::State::Compiled ? "COMPILED"
                     : e.state == WJEntry::State::Failed ? "FAILED(PBL)"
                                                         : "Cold";
    fprintf(stderr, "  %s:%u deopts=%u jitRuns=%u rate=%d%% recomp=%u %s\n",
            sc->filename() ? sc->filename() : "?", unsigned(sc->lineno()),
            e.deopts, e.jitRuns, rate, e.recompiles, st);
  }
}

// Warm-up delay: WarpOracle needs PBL-attached CacheIR stubs, which only appear
// after the body has run many times. Compile after this many observations.
static uint32_t WarmupDelay() {
  static uint32_t n = 0;
  if (!n) {
    const char* s = getenv("GECKO_WJWARP_DELAY");
    // Default 800 (was 200): under realistic V8 (Liftoff baseline + tier-up, what
    // a real browser uses) a higher compile threshold nearly halves the COLD-INIT
    // real-site penalty (cold acorn parse 2.3x->1.2x vs PBL) by not compiling the
    // many moderately-called functions of a one-shot parse, while octane stays
    // neutral (richards/navier flat, raytrace/earley/crypto up, deltablue -13% but
    // still ~6.7x). Only compile-TIMING, never correctness. GECKO_WJWARP_DELAY
    // overrides. See memory: liftoff-vs-turbofan-measurement.
    n = s ? uint32_t(atoi(s)) : 800;
    if (!n) n = 800;
  }
  return n;
}

}  // namespace

bool js::wasm::WasmJitInWasm() { return js::wasm::gWJExecDepth > 0; }

// __wjStats() backend: aggregate the ALWAYS-ON JIT counters into a JSON string
// (no env flags, queryable mid-run -- the printf-free introspection of task #57).
// Per-entry deopts/jitRuns are maintained unconditionally (they drive recompiles),
// so this is zero added hot-path cost. Reports entry-state tallies, JIT/deopt
// totals + rate, the top fns by deopt count, and deopt-by-op (named) if the
// GECKO_WJ_DEOPTHIST counters were populated this run.
//
// TRUST THE ALWAYS-ON NUMBERS (deopts/deoptRatePct/topDeoptFns): they are real
// top-level frame resumes. deoptByOp is a separate DEOPTHIST-only histogram of
// guard-deopt-instruction *executions*; it matches the per-entry total 1:1 for a
// non-inlined fn but OVER-counts under inlining (guard sites in inlined callee
// bodies fire the histogram without each being a distinct top-level resume), so
// read it only for the RELATIVE ranking of which guard kinds deopt, never as an
// absolute bail count -- cross-check against deoptRatePct.
// BROWSER-capturable form of __wjStats: the XUL embed has no JS shell global to
// hang a native off, and atexit never runs here (-sEXIT_RUNTIME=0 + the runner's
// process.exit skip C++ dtors), so -- like COMPILESTAT/DEOPTHIST -- we emit the JSON
// PERIODICALLY during execution to stderr with a grep-able prefix. The playwright
// harness captures process stderr and reads the LAST `[wj-statsjson] {...}` line
// once the page has settled (a near-final snapshot). Gated by GECKO_WJ_STATSJSON,
// throttled to every STATSJSON'th JIT run (env value; default 50000) to bound cost
// (WJStatsJSON walks gEntries). gWJStatsEvery==0 means the flag is off.
uint32_t js::wasm::gWJStatsEvery = 0xffffffffu;
static void WJStatsDumpTick() {
  if (js::wasm::gWJStatsEvery == 0xffffffffu) {
    const char* s = getenv("GECKO_WJ_STATSJSON");
    js::wasm::gWJStatsEvery = s ? uint32_t(atoi(s)) : 0;
    if (s && js::wasm::gWJStatsEvery == 0) js::wasm::gWJStatsEvery = 50000;
  }
  if (!js::wasm::gWJStatsEvery) return;
  static uint64_t tick = 0;
  if (++tick % js::wasm::gWJStatsEvery) return;
  static char buf[8192];
  js::wasm::WJStatsJSON(buf, sizeof(buf));
  fprintf(stderr, "[wj-statsjson] %s\n", buf);
}

void js::wasm::WJStatsJSON(char* buf, size_t n) {
  if (!buf || n == 0) return;
  uint32_t compiled = 0, failed = 0, cold = 0;
  uint64_t totJit = 0, totDeopt = 0, totRecomp = 0;
  // Rank + total on LIFETIME deopts (deoptsTotal, never reset) so the summary
  // reconciles with the monotonic gWJDeoptByOp histogram; per-entry `deopts`
  // is reset on each recompile and would undercount here.
  struct Top { JSScript* sc; uint32_t deopts, jitRuns; };
  Top top[8] = {};
  if (gEntries) {
    for (auto& kv : *gEntries) {
      WJEntry& e = kv.second;
      if (e.state == WJEntry::State::Compiled) compiled++;
      else if (e.state == WJEntry::State::Failed) failed++;
      else cold++;
      totJit += e.jitRuns;
      totDeopt += e.deoptsTotal;
      totRecomp += e.recompiles;
      if (e.deoptsTotal > top[7].deopts) {
        top[7] = {kv.first, e.deoptsTotal, e.jitRuns};
        for (int i = 7; i > 0 && top[i].deopts > top[i - 1].deopts; i--) {
          Top t = top[i]; top[i] = top[i - 1]; top[i - 1] = t;
        }
      }
    }
  }
  uint64_t totRuns = totJit + totDeopt;
  int rate = totRuns ? int((100ull * totDeopt) / totRuns) : 0;
  size_t o = 0;
  auto app = [&](const char* fmt, ...) {
    if (o >= n) return;
    va_list ap; va_start(ap, fmt);
    int w = vsnprintf(buf + o, n - o, fmt, ap);
    va_end(ap);
    if (w > 0) o += size_t(w);
  };
  app("{\"compiled\":%u,\"failed\":%u,\"cold\":%u,\"jitRuns\":%llu,"
      "\"deopts\":%llu,\"deoptRatePct\":%d,\"recompiles\":%llu,\"topDeoptFns\":[",
      compiled, failed, cold, (unsigned long long)totJit,
      (unsigned long long)totDeopt, rate, (unsigned long long)totRecomp);
  bool first = true;
  for (int i = 0; i < 8; i++) {
    if (!top[i].sc || !top[i].deopts) continue;
    JSScript* sc = top[i].sc;
    app("%s{\"fn\":\"%s:%u\",\"deopts\":%u,\"jitRuns\":%u}",
        first ? "" : ",", sc->filename() ? sc->filename() : "?",
        unsigned(sc->lineno()), top[i].deopts, top[i].jitRuns);
    first = false;
  }
  app("],\"deoptByOp\":[");
  first = true;
  for (uint32_t op = 0; op < js::wasm::kWJNumOps; op++) {
    if (!gWJDeoptByOp[op]) continue;
    app("%s{\"op\":\"%s\",\"n\":%u}", first ? "" : ",",
        js::wasm::WJMirOpName(op), gWJDeoptByOp[op]);
    first = false;
  }
  app("]}");
  if (o >= n) buf[n - 1] = 0;
}

void js::wasm::WasmJitInvalidateAll(const char* reason) {
  uint32_t n = 0;
  if (gEntries) {
    for (auto& kv : *gEntries) {
      WJEntry& e = kv.second;
      if (e.state != WJEntry::State::Compiled) continue;
      e.state = WJEntry::State::Cold;
      e.handle = -1;
      e.tblSlot = -1;
      e.directIdx = -1;
      e.observes = 0;
      e.nextTry = 0;
      e.osrTargets.clear();
      n++;
    }
  }
  // Stale call-IC entries would dispatch to the abandoned table slots; clearing
  // the callee-ptr side is sufficient (tblIdx is only read on a fn-ptr hit).
  // NOTE: a JIT frame already ON THE STACK when the fuse pops finishes with the
  // old (elided) code, like a non-bailing Ion frame would -- acceptable for the
  // rare mid-flight pop; new calls all re-enter PBL and recompile fresh.
  memset(gWJCallFn, 0, sizeof(uint32_t) * kWJCallSites * kWJCallWays);
  static int dbg = getenv("GECKO_WJ_INVALDBG") ? 1 : 0;
  if (dbg)
    fprintf(stderr, "[wj-invalidate-all] reason=%s flushed=%u\n", reason, n);
}

// Cheap routing pre-check for the PBL call fast path (task #60): 0 = not
// JIT-able (take the PBL fast path), 1 = already compiled (route to the IC,
// no compile possible), 2 = a synchronous COMPILE may run inside
// WasmJitObserveCall -- the caller must establish a covering exit frame first
// (the compile allocates; a GC with no valid exitFP leaves PBL frames
// untraced). Mirrors WasmJitObserveCall's head checks; one hash lookup.
int js::wasm::WasmJitPreCall(JSScript* script) {
  static int sEnabled = -1;
  if (sEnabled < 0) sEnabled = getenv("GECKO_NOWASMJIT") ? 0 : 1;
  if (!sEnabled) return 0;
  static uint32_t sMaxLen2 = 0;
  if (!sMaxLen2) {
    const char* s = getenv("GECKO_WJ_MAXLEN");
    sMaxLen2 = s ? uint32_t(atoi(s)) : 4096;
    if (!sMaxLen2) sMaxLen2 = 4096;
  }
  if (!script->function() || script->isModule() ||
      script->length() > sMaxLen2) {
    return 0;
  }
  WJEntry& e = EntryFor(script);
  if (e.state == WJEntry::State::Compiled) return 1;
  if (e.state == WJEntry::State::Failed) return 0;
  return 2;
}

bool js::wasm::WasmJitObserveCall(JSScript* script) {
  static int sEnabled = -1;
  if (sEnabled < 0) sEnabled = getenv("GECKO_NOWASMJIT") ? 0 : 1;
  if (!sEnabled) return false;
  // Max compilable bytecode length. Big functions (uBlock's parseNetPattern/validateNet/
  // analyze_re ~4.4-6KB) were never attempted at the old hard 4096. Configurable so we can
  // compile them. GECKO_WJ_MAXLEN overrides.
  static uint32_t sMaxLen = 0;
  if (!sMaxLen) {
    const char* s = getenv("GECKO_WJ_MAXLEN");
    sMaxLen = s ? uint32_t(atoi(s)) : 4096;
    if (!sMaxLen) sMaxLen = 4096;
  }
  if (!script->function() || script->isModule() || script->length() > sMaxLen) {
    static int pblWho = getenv("GECKO_WJ_PBLWHO") ? 1 : 0;
    if (pblWho && script->function() && !script->isModule() &&
        script->length() > 4096) {
      static std::unordered_map<JSScript*, uint64_t> big;
      uint64_t& c = big[script];
      if (c == 0)
        fprintf(stderr, "[wb-pblwho] TOO-BIG(len=%u>4096, never compiled) %s:%u\n",
                unsigned(script->length()),
                script->filename() ? script->filename() : "?",
                unsigned(script->lineno()));
      c++;
    }
    return false;
  }

  WJEntry& e = EntryFor(script);
  if (e.state == WJEntry::State::Compiled) return true;
  if (e.state == WJEntry::State::Failed) return false;

  // Size-scaled warmup: the per-function wasm COMPILE cost scales with bytecode
  // length (big parser/framework fns cost 10-25ms to compile; tiny hot loops <1ms).
  // A compile only pays off after enough REMAINING runs to amortize it, and
  // break-even runs scale with the compile cost -> with length. A flat threshold
  // makes a compile-heavy one-shot LOAD (acorn/marked parse, a site's bundle)
  // compile many big fns that never amortize -- measured JIT one-shot 1.85x (acorn)
  // to 5.1x (marked) SLOWER than PBL, while throughput is 1.4-2x FASTER. Scale the
  // threshold by length so big fns need far more evidence of hotness before paying
  // their large compile; small hot loops (octane) still tier up early. Applied to
  // the call-count (nextTry) gate ONLY (see below). GECKO_WJ_SIZEWARMUP=K sets the
  // per-byte term; DEFAULT 0 (OFF/flat) -- measured, it is a genuine LOAD-vs-
  // THROUGHPUT TRADEOFF, not a clean win: K=6 improved a one-shot LOAD ~30% (acorn
  // 2169->1500ms, marked 3041->2196ms) but regressed repeated-execution THROUGHPUT
  // ~20-35% (marked 48->65ms) because medium hot fns tier up later. The real fix is
  // DEFERRED/IDLE compilation (compile off the load critical path), which needs a
  // main-thread-idle hook the embed lacks. Knob kept for load-optimized deployments.
  static uint32_t sizeK = 0xffffffffu;
  if (sizeK == 0xffffffffu) {
    const char* s = getenv("GECKO_WJ_SIZEWARMUP");
    sizeK = s ? uint32_t(atoi(s)) : 0;
  }
  uint32_t lenScale = sizeK * script->length();
  if (e.nextTry == 0) e.nextTry = WarmupDelay() + lenScale;
  // Trigger on EITHER call count (our observes) OR the script's loop-aware
  // warmUpCount (PBL bumps it on every LoopHead). The latter catches the hot
  // driver functions that are entered rarely but loop thousands of times
  // internally (e.g. richards' schedule) -- by their 2nd call the accumulated
  // loop warmup is huge, so we compile them and let Warp inline their dispatch.
  ++e.observes;
  // Post-storm: stay in PBL until the IC warms polymorphic (overrides the warm-
  // path bypass below, which would otherwise recompile early + monomorphic).
  if (e.recompileFloor && e.observes < e.recompileFloor) return false;
  uint32_t warm =
      script->hasJitScript() ? script->jitScript()->warmUpCount() : 0;
  static uint32_t kLoopWarm = 0;
  if (!kLoopWarm) {
    const char* s = getenv("GECKO_WJWARP_LOOPWARM");
    kLoopWarm = s ? uint32_t(atoi(s)) : 2000;
    if (!kLoopWarm) kLoopWarm = 2000;
  }
  static int obsDbg = getenv("GECKO_WJ_OBSDBG") ? atoi(getenv("GECKO_WJ_OBSDBG")) : -1;
  bool obsMatch = obsDbg >= 0 && uint32_t(obsDbg) == unsigned(script->lineno());
  if (obsMatch) {
    static uint64_t oc = 0;
    if ((++oc % 5000) == 0 || oc < 5)
      fprintf(stderr, "[wj-obsdbg] :%u state=%d observes=%u nextTry=%u warm=%u kLoop=%u fails=%u len=%u\n",
              unsigned(script->lineno()), int(e.state), e.observes, e.nextTry, warm,
              kLoopWarm, e.fails, unsigned(script->length()));
  }
  // Scale ONLY the call-count gate by size, NOT the loop-warm gate: a genuinely
  // hot function (high warmUpCount from internal LoopHeads) should still tier up
  // promptly regardless of size (that is where the JIT win is + preserves octane/
  // throughput). Deferring only applies to functions that are merely CALLED a lot
  // without looping hot -- the parser-dispatch pattern a one-shot load hits.
  // NOTE (2026-07-11): a SMALL-FN call-count gate (GECKO_WJ_SMALLLEN/SMALLMULT) was
  // TRIED here to keep gbemu's net-negative tiny dispatched handlers in PBL, and
  // REVERTED: it CATASTROPHICALLY regresses benches with hot small NON-LOOPING
  // methods (richards 2333->102, deltablue 531->91, raytrace 1860->383) because the
  // loop-warm bypass below does NOT protect call-hot-but-loopless small fns, which
  // ARE net-positive to compile there. Size+call-count CANNOT distinguish gbemu's
  // net-negative tiny handlers (net-negative because reached via MEGAMORPHIC fn-ptr
  // dispatch) from richards' net-positive tiny methods. See gbemu memory.
  if (e.observes < e.nextTry && warm < kLoopWarm) return false;

  // NOCOMPILE-lineno-range bisection (GECKO_WJ_NOCOMPILERANGE=lo,hi): refuse to
  // compile any function whose script lineno is in [lo,hi] -> it runs in PURE PBL
  // (never compiled, NO deopt-resume involved -- unlike FORCEDEOPT which routes
  // through the resume path and can corrupt). Binary-search lo/hi to find which
  // function's JIT compilation is the miscompile, cleanly.
  static int nclo = -2, nchi = -2;
  if (nclo == -2) {
    nclo = nchi = -1;
    if (const char* r = getenv("GECKO_WJ_NOCOMPILERANGE")) {
      nclo = atoi(r); const char* c = strchr(r, ',');
      nchi = c ? atoi(c + 1) : nclo;
    }
  }
  if (nclo >= 0) {
    uint32_t ln = unsigned(script->lineno());
    if (ln >= uint32_t(nclo) && ln <= uint32_t(nchi)) { e.state = WJEntry::State::Failed; return false; }
  }

  // DEFERRED COMPILE: the function crossed the warmup threshold, but compiling it
  // SYNCHRONOUSLY here blocks the currently-executing task (a page LOAD compiles
  // many briefly-run fns => the compile is pure critical-path cost that never
  // amortizes; measured node one-shot 1.85-5.1x slower than PBL). Instead, enqueue
  // and stay in PBL; WasmJitDrainDeferred() (called by the embed at an idle / task
  // boundary) compiles queued fns OFF the critical path. A one-shot LOAD then runs
  // entirely in PBL (fast); repeated execution drains between tasks so hot fns still
  // get JIT'd (throughput preserved). Only defer the FIRST compile decision (fails==0,
  // not draining) -- retries after a bail and drain-time compiles go inline.
  // Enabled by GECKO_WJ_DEFERCOMPILE (any value). The BROWSER embed (embed-xul
  // main()) setenv's it on by DEFAULT since it wires an idle-loop drain
  // (WasmJitDrainDeferred); the node embed leaves it OFF (its single-JS::Evaluate
  // benchmark harness has no intra-loop task boundary to drain at, so deferring
  // would strand hot fns in PBL and crater octane/jetstream). GECKO_WJ_NODEFER
  // force-disables (for A/B). Requires the embed to call WasmJitDrainDeferred() at
  // an idle/task boundary, else deferred fns never compile.
  static int deferCompile =
      (getenv("GECKO_WJ_DEFERCOMPILE") && !getenv("GECKO_WJ_NODEFER")) ? 1 : 0;
  // STRAND LIMIT: if a hot fn is observed this many times while queued but no drain
  // has compiled it, fall through to a SYNCHRONOUS compile instead of staying in PBL
  // forever. This is the safety net for (a) a compute-bound page that never yields to
  // the idle-drain, and (b) a harness (node octane) with no intra-loop task boundary
  // -- both otherwise strand every hot fn in PBL (octane cratered 18x). A real LOAD
  // drains at its first task boundary long before this, so it keeps the defer benefit.
  // GECKO_WJ_DEFERSTRAND overrides the limit (0 = never fall through, old behavior).
  static uint32_t strandLimit =
      getenv("GECKO_WJ_DEFERSTRAND") ? uint32_t(atoi(getenv("GECKO_WJ_DEFERSTRAND")))
                                     : 4096;
  if (deferCompile && !gWJDrainingNow && e.fails == 0 &&
      e.state == WJEntry::State::Cold) {
    // Enqueue on first qualification; on EVERY later observe stay in PBL (return
    // false) until the drain compiles it -- otherwise the 2nd observe would fall
    // through to the inline compile and defeat the deferral.
    if (!e.queued) {
      e.queued = true;
      e.queuedObserves = 0;
      gWJDeferQueue.push_back(script);
    }
    // Stay in PBL until either the drain compiles it OR we hit the strand limit, at
    // which point fall through to compile synchronously (a hot fn stranded because
    // no drain ever fired).
    if (strandLimit == 0 || ++e.queuedObserves < strandLimit) {
      return false;
    }
    // fall through: synchronous compile of a stranded hot fn (queued flag stays set;
    // the queued slot in gWJDeferQueue becomes a no-op when drained since state
    // advances past Cold below).
  }

  JSContext* cx = js::TlsContext.get();
  if (!cx) return false;
  uint32_t nargs = 0, nlocals = 0;
  // Preferred shared-table slot: reuse the prior slot on a deopt-storm recompile
  // (callers' call ICs cache funPtr->slot; funPtr is unchanged, so the new module
  // must take over the same slot). -1 (first compile) allocates a fresh one.
  int tblSlot = e.tblSlot;
  js::wasm::gWJForceMega = e.forceMega;  // megamorphic recompile (post-storm)
  gWJPendingOsrTargets.clear();  // backend appends OSR targets during this compile
  // COMPILE-COST instrumentation (GECKO_WJ_COMPILESTAT): synchronous main-thread
  // tier-up compile is on the load critical path -- a real-site load compiles
  // MANY briefly-run functions and NEVER amortizes the compile (unlike octane's
  // hot loops). Measure count + wall-ms + attempts-that-bail to test whether the
  // compile itself is the "JIT slower to LOAD real sites" cost.
  static int compileStat =
      getenv("GECKO_WJ_COMPILESTAT") ? atoi(getenv("GECKO_WJ_COMPILESTAT")) : 0;
  mozilla::TimeStamp wjCompT0;
  if (compileStat) wjCompT0 = mozilla::TimeStamp::Now();
  int handle = js::wasm::WJWarpCompile(cx, script, &nargs, &nlocals, &tblSlot);
  js::wasm::gWJForceMega = false;
  if (compileStat) {
    double ms = (mozilla::TimeStamp::Now() - wjCompT0).ToMilliseconds();
    // Per-compile (bytecode-length, ms) log to test emit super-linearity:
    // GECKO_WJ_COMPILESTAT=2 prints every compile's script length + wall-ms.
    if (compileStat >= 2 && handle >= 0) {
      fprintf(stderr, "[wj-percompile] len=%u ms=%.2f\n",
              unsigned(script->length()), ms);
    }
    js::wasm::gWJCompileAttempts++;
    js::wasm::gWJCompileMs += ms;
    if (handle >= 0) {
      js::wasm::gWJCompileOK++;
      js::wasm::gWJCompileOKMs += ms;
    }
    if ((js::wasm::gWJCompileAttempts % 25) == 0) {
      double emitMs = js::wasm::gWJCompileMs - js::wasm::gWJSnapshotMs -
                      js::wasm::gWJBuildMs - js::wasm::gWJOptimizeMs -
                      js::wasm::gWJHostCompileMs - js::wasm::gWJHostInstMs;
      fprintf(stderr,
              "[wj-compilestat] attempts=%llu ok=%llu totalMs=%.1f | "
              "snapshot=%.1f build=%.1f optimize=%.1f emit=%.1f "
              "hostCompile=%.1f hostInst=%.1f | bytes=%lluK (avg-ok=%.2fms)\n",
              (unsigned long long)js::wasm::gWJCompileAttempts,
              (unsigned long long)js::wasm::gWJCompileOK,
              js::wasm::gWJCompileMs, js::wasm::gWJSnapshotMs,
              js::wasm::gWJBuildMs, js::wasm::gWJOptimizeMs, emitMs,
              js::wasm::gWJHostCompileMs, js::wasm::gWJHostInstMs,
              (unsigned long long)(js::wasm::gWJEmitBytes / 1024),
              js::wasm::gWJCompileOK
                  ? js::wasm::gWJCompileOKMs / double(js::wasm::gWJCompileOK)
                  : 0.0);
    }
  }
  if (e.forceMega && getenv("GECKO_WJ_MEGADBG")) {
    fprintf(stderr, "[wj-mega-compile] %s:%u handle=%d\n",
            script->filename() ? script->filename() : "?",
            unsigned(script->lineno()), handle);
  }
  if (handle < 0) {
    // Recompile-when-warm: a bail is often a cold IC (Warp emits an unconditional
    // bailout for an op whose baseline IC hasn't specialized yet). Retry later --
    // as the bench runs, callee/op ICs warm up and the bail disappears. Cap retries
    // so a genuinely-unsupported function eventually gives up (stays in PBL).
    const char* reason =
        js::wasm::gWJBailReason ? js::wasm::gWJBailReason : "?";
    // LOG-BAIL (non-fatal coverage audit): print each compile bail so a bench that
    // silently degrades to ~1x (hot fns in PBL) can be diagnosed. Post-process with
    // sort|uniq -c (retries repeat the same line). Off by default.
    if (getenv("GECKO_WJ_LOGBAIL")) {
      fprintf(stderr,
              "[WJ-BAIL] fn=%s:%u reason=%s op@=%s:%u(off=%u)\n",
              script->filename() ? script->filename() : "?",
              unsigned(script->lineno()), reason,
              js::wasm::gWJBailOpFile ? js::wasm::gWJBailOpFile : "?",
              js::wasm::gWJBailOpLine, js::wasm::gWJBailOpOff);
    }
    // FAIL-ON-BAIL: a compile bail means the function runs in PBL forever -- the
    // exact "running in PBL not JIT" signal we must never miss. With FAILONBAIL set,
    // print it loudly and ABORT so any bench with an unJIT-able function insta-fails
    // instead of silently degrading to ~1x. (Off by default so normal runs tolerate
    // the rare residual bail; turn on to audit coverage.)
    if (getenv("GECKO_WJ_FAILONBAIL")) {
      fprintf(stderr,
              "\n[WJ-COMPILE-FAIL] %s:%u reason=%s -- function will run in PBL, "
              "not JIT. Aborting (GECKO_WJ_FAILONBAIL).\n",
              script->filename() ? script->filename() : "?",
              unsigned(script->lineno()), reason);
      fflush(stderr);
      MOZ_CRASH("WJ compile bail with GECKO_WJ_FAILONBAIL");
    }
    // Permanent-bail short-circuit: a bail whose cause is STRUCTURAL (the function's
    // bytecode / arg count / control-flow can't change) produces the identical bail
    // on every retry -- each retry burns a full snapshot+build+optimize(+emit) on the
    // load critical path for nothing. Measured on a real workload: 80 failed compile
    // attempts were only 12 unique functions (~6.7 wasted retries each), and EVERY
    // reason was permanent (reloop-bail / too-many-args / host-compile-reject), zero
    // transient cold-IC. So cap retries per-reason: structural = give up at once;
    // host-compile-reject (our codegen emitted invalid wasm, ~always deterministic) =
    // cap low; everything else (possibly a cold IC that warms up) keeps the full 8.
    // GECKO_WJ_NOPERMBAIL restores the flat 8-retry behavior.
    static int noPermBail = getenv("GECKO_WJ_NOPERMBAIL") ? 1 : 0;
    uint32_t cap = 8;
    if (!noPermBail && reason) {
      if (strcmp(reason, "reloop-bail") == 0 ||
          strcmp(reason, "too-many-args") == 0 ||
          strcmp(reason, "trycatch") == 0) {
        cap = 1;
      } else if (strcmp(reason, "host-compile-reject") == 0) {
        cap = 2;
      }
    }
    if (++e.fails >= cap) {
      e.state = WJEntry::State::Failed;
    } else {
      e.nextTry = e.observes + (WarmupDelay() << e.fails);
    }
    return false;
  }
  if (getenv("GECKO_WJ_PBLWHO"))
    fprintf(stderr, "[wb-compiled] %s:%u len=%u directIdx=%d\n",
            script->filename() ? script->filename() : "?",
            unsigned(script->lineno()), unsigned(script->length()), e.directIdx);
  e.state = WJEntry::State::Compiled;
  e.handle = handle;
  e.tblSlot = tblSlot;
  e.hasAlwaysBails = js::wasm::gWJHadAlwaysBails;
  e.shapeDeoptDom = js::wasm::gWJEmitShapeDeopts > 0;  // ANY shape-family deopt site
  e.osrTargets = gWJPendingOsrTargets;  // OSR loop-head pc->block map for this fn
  if (getenv("GECKO_WJ_COMPILECNT")) {
    static uint64_t cc = 0;
    fprintf(stderr, "[wj-compile] #%llu %s:%u hasAB=%d recomp=%u\n",
            (unsigned long long)(++cc),
            script->filename() ? script->filename() : "?",
            unsigned(script->lineno()), e.hasAlwaysBails, e.recompiles);
  }
  e.nargs = nargs;
  e.nlocals = nlocals;
  // Register the trampoline in the MAIN indirect table for direct (no-JS-hop)
  // PBL->JIT entry via a C function pointer. -1 if registration failed.
  if (!getenv("GECKO_WJ_NODIRECT")) {
    e.directIdx = int(wasmhost_call(handle, -1, nullptr, 0));
    if (e.directIdx <= 0) e.directIdx = -1;
  }
  if (getenv("GECKO_WJ_PBLWHO"))
    fprintf(stderr, "[wb-directidx] %s:%u directIdx=%d\n",
            script->filename() ? script->filename() : "?",
            unsigned(script->lineno()), e.directIdx);
  return true;
}

// Drain the deferred-compile queue: compile every script that crossed the warmup
// threshold while GECKO_WJ_DEFERCOMPILE deferred it. The embed calls this at an
// idle / task boundary (browser event loop between tasks; node between top-level
// runs) so the compiles happen OFF the load critical path. Reuses WasmJitObserveCall
// with gWJDrainingNow set, so the queued script takes the (already-qualified) inline
// compile path instead of re-deferring. Bounded per-call by GECKO_WJ_DRAINBUDGET
// (0 = all) so a huge backlog doesn't stall one idle slot.
void js::wasm::WasmJitDrainDeferred() {
  if (gWJDeferQueue.empty() || gWJDrainingNow) return;
  static uint32_t budget = 0xffffffffu;
  if (budget == 0xffffffffu) {
    const char* s = getenv("GECKO_WJ_DRAINBUDGET");
    budget = s ? uint32_t(atoi(s)) : 0;  // 0 => drain the whole queue each call
  }
  gWJDrainingNow = true;
  uint32_t done = 0;
  size_t i = 0;
  for (; i < gWJDeferQueue.size(); i++) {
    if (budget && done >= budget) break;
    JSScript* s = gWJDeferQueue[i];
    WJEntry& e = EntryFor(s);
    e.queued = false;
    if (e.state == WJEntry::State::Cold) {
      WasmJitObserveCall(s);  // qualified already -> compiles inline now
      done++;
    }
  }
  // Remove the drained prefix; keep any remainder (budget-limited) for next drain.
  if (i >= gWJDeferQueue.size()) {
    gWJDeferQueue.clear();
  } else {
    gWJDeferQueue.erase(gWJDeferQueue.begin(), gWJDeferQueue.begin() + i);
  }
  gWJDrainingNow = false;
  if (getenv("GECKO_WJ_DEFERDBG"))
    fprintf(stderr, "[wj-drain] compiled=%u remaining=%zu\n", done,
            gWJDeferQueue.size());
}

// Set during the differential verifier's interpreter re-run so nested calls
// decline JIT entry (run in PBL) -- otherwise the "interp" re-run would re-JIT
// and we'd compare JIT-vs-JIT (never catching a JIT bug).
static bool gWJVerifyReentry = false;

// The deopted call COMPLETED via the resume (side effects + result done): report
// "ran" with the result so the PBL caller doesn't re-run it (valveReturn semantics).
static int valveReturnEarly(uint64_t* retBits) {
  *retBits = gWJScratch[js::wasm::kWJResultSlot];
  return 1;
}

int js::wasm::WasmJitRunCall(JSScript* script, uint64_t thisBits,
                             const JS::Value* argv, uint32_t argc,
                             JSObject* envChain, uint64_t* retBits) {
  if (gWJVerifyReentry) return 0;  // verifier re-run: force PBL
  if (!gEntries) return 0;
  // Last-lookup cache: hot functions are entered repeatedly with the same script,
  // so skip the unordered_map find (per-PBL->JIT-entry cost on entry-heavy benches
  // like splay). Map element addresses are stable across rehashes, so caching the
  // WJEntry* is safe; state is re-read fresh each call.
  // Direct-mapped cache over the unordered_map: uBlock alternates between many callees
  // per call site, so a 1-entry cache thrashed -> the hash find showed ~0.9% in the
  // profile. Map element addresses are stable across rehashes, so caching the WJEntry*
  // is safe; state is re-read fresh each call. 8192-way (was 512): sized to the
  // ~2500-script working set (typescript) so the RunCall dispatch stops thrashing this
  // cache (2026-07-11 profile follow-up; 128KB, negligible).
  static const uint32_t kLCBits = 13;
  static const uint32_t kLCMask = (1u << kLCBits) - 1;
  static JSScript* sLScript[1u << kLCBits] = {};
  static WJEntry* sLEntry[1u << kLCBits] = {};
  uint32_t lch = uint32_t(uintptr_t(script) >> 3) & kLCMask;
  WJEntry* ep;
  if (sLScript[lch] == script) {
    ep = sLEntry[lch];
  } else {
    auto it = gEntries->find(script);
    if (it == gEntries->end()) {
      static int pblWho0 = getenv("GECKO_WJ_PBLWHO") ? 1 : 0;
      if (pblWho0) {
        static std::unordered_map<JSScript*, int> seen;
        if (seen.emplace(script, 1).second)
          fprintf(stderr, "[wb-pblwho] NOT-IN-MAP(never-compiled) %s:%u\n",
                  script->filename() ? script->filename() : "?",
                  unsigned(script->lineno()));
      }
      return 0;
    }
    ep = &it->second;
    sLScript[lch] = script;
    sLEntry[lch] = ep;
  }
  WJEntry& e = *ep;
  static int pblWho = getenv("GECKO_WJ_PBLWHO") ? 1 : 0;
  if (pblWho && e.state == WJEntry::State::Compiled) {
    // Count JIT'd functions entered FROM PBL (the 2.25M PBL->JIT transitions). The
    // top callees reveal what the hot PBL caller is invoking -> identifies the
    // uncompiled hot caller to target.
    static std::unordered_map<JSScript*, uint64_t> cc;
    static uint64_t t = 0;
    cc[script]++;
    if ((++t % 1000000) == 0) {
      fprintf(stderr, "[wb-pblcallee] top PBL->JIT callees @%llu:\n",
              (unsigned long long)t);
      std::vector<std::pair<JSScript*, uint64_t>> v(cc.begin(), cc.end());
      std::sort(v.begin(), v.end(),
                [](auto& a, auto& b) { return a.second > b.second; });
      for (size_t i = 0; i < v.size() && i < 6; i++)
        fprintf(stderr, "    %s:%u x%llu\n",
                v[i].first->filename() ? v[i].first->filename() : "?",
                unsigned(v[i].first->lineno()), (unsigned long long)v[i].second);
    }
  }
  if (e.state != WJEntry::State::Compiled) {
    if (pblWho) {
      static std::unordered_map<JSScript*, uint64_t> cnt;
      static uint64_t tot = 0;
      cnt[script]++;
      if ((++tot % 500000) == 0) {
        fprintf(stderr, "[wb-pblwho] state!=Compiled top scripts:\n");
        std::vector<std::pair<JSScript*, uint64_t>> v(cnt.begin(), cnt.end());
        std::sort(v.begin(), v.end(),
                  [](auto& a, auto& b) { return a.second > b.second; });
        for (size_t i = 0; i < v.size() && i < 8; i++)
          fprintf(stderr, "    %s:%u state=%d  count=%llu\n",
                  v[i].first->filename() ? v[i].first->filename() : "?",
                  unsigned(v[i].first->lineno()), int(gEntries->count(v[i].first)
                      ? (*gEntries)[v[i].first].state : WJEntry::State()),
                  (unsigned long long)v[i].second);
      }
    }
    return 0;
  }
  if (argc < e.nargs) {
    if (pblWho) {
      static uint64_t uf = 0;
      if ((++uf % 200000) == 0)
        fprintf(stderr, "[wb-pblwho] argc<nargs underflow %s:%u (argc=%u nargs=%u) x%llu\n",
                script->filename() ? script->filename() : "?",
                unsigned(script->lineno()), argc, e.nargs, (unsigned long long)uf);
    }
    return 0;  // underflow: let the interpreter pad
  }

  // Stage ALL actual args (not just the formal nargs) so the callee's GetFrameArgument
  // can read args beyond its formal count (e.g. self-hosted IteratorReduce reads
  // GetArgument(1)=initialValue while its only formal is `reducer`). Staging only
  // e.nargs left those slots stale -> reduce-with-init double-counted during warmup
  // (before the caller is JIT'd and uses the all-8-args fast path). Pad the rest with
  // undefined. Capped at kWJMaxArgs (GetFrameArgument bails for i>=kWJMaxArgs).
  uint32_t nstage = argc < uint32_t(js::wasm::kWJMaxArgs) ? argc
                                                          : uint32_t(js::wasm::kWJMaxArgs);
  for (uint32_t i = 0; i < nstage; i++) {
    gWJScratch[i] = argv[i].asRawBits();
  }
  for (uint32_t i = nstage; i < uint32_t(js::wasm::kWJMaxArgs); i++) {
    gWJScratch[i] = JS::UndefinedValue().asRawBits();
  }
  gWJScratch[js::wasm::kWJThisSlot] = thisBits;

  // Mutation verifier (GECKO_WJ_VERIFYMUT): snapshot this/arg object slots BEFORE
  // the wasm runs, so we can later compare the wasm's mutations against a PBL
  // re-run (catches wrong SIDE EFFECTS the return-value verifier can't see).
  static int verifyMut = getenv("GECKO_WJ_VERIFYMUT") ? 1 : 0;
  struct MutSnap {
    uint32_t rootIdx;  // index into mutRoots (the GC-rooted object handle)
    uint32_t n;
    uint64_t before[16];
    uint64_t wasmAfter[16];
    // Dense element snapshot (crypto bignum limbs live here, NOT in slots).
    uint32_t denseBeforeLen = 0;
    uint32_t denseWasmAfterLen = 0;
    std::vector<uint64_t> denseBefore;
    std::vector<uint64_t> denseWasmAfter;
  };
  std::vector<MutSnap> msnaps;
  JSContext* mutCx = verifyMut ? js::TlsContext.get() : nullptr;
  // Persisted, GC-rooted handles to the snapshot objects so they survive the
  // wasm call's GC (raw JSObject* would go stale -> false garbage mutations).
  JS::RootedValueVector mutRoots(mutCx ? mutCx : js::TlsContext.get());
  if (verifyMut && mutCx) {
    auto snap = [&](uint64_t bits) {
      JS::Value v = JS::Value::fromRawBits(bits);
      if (v.isObject() && v.toObject().is<js::NativeObject>()) {
        js::NativeObject& o = v.toObject().as<js::NativeObject>();
        // Dedup by object identity: an in-place call (e.g. x.drShiftTo(n,x),
        // this===r) would otherwise snapshot the SAME object twice, and the
        // sequential restore-before-rerun corrupts the 2nd snapshot's wasmAfter
        // (it reads the already-restored "before" state) -> false divergence.
        for (uint32_t m = 0; m < mutRoots.length(); m++) {
          if (mutRoots[m].isObject() && &mutRoots[m].toObject() == &v.toObject())
            return;
        }
        MutSnap sp;
        sp.rootIdx = uint32_t(mutRoots.length());
        (void)mutRoots.append(v);
        sp.n = std::min<uint32_t>(o.slotSpan(), 16);
        for (uint32_t s = 0; s < sp.n; s++) sp.before[s] = o.getSlot(s).asRawBits();
        uint32_t dn = std::min<uint32_t>(o.getDenseInitializedLength(), 256);
        sp.denseBeforeLen = dn;
        for (uint32_t d = 0; d < dn; d++)
          sp.denseBefore.push_back(o.getDenseElement(d).asRawBits());
        msnaps.push_back(std::move(sp));
      }
    };
    snap(thisBits);
    for (uint32_t i = 0; i < argc; i++) snap(argv[i].asRawBits());
    // One level deep: also snapshot objects reachable through this/args' slots
    // (e.g. this.position, this.color) so nested-object mutations are caught.
    uint32_t topLevel = uint32_t(msnaps.size());
    for (uint32_t m = 0; m < topLevel; m++) {
      for (uint32_t s = 0; s < msnaps[m].n; s++) snap(msnaps[m].before[s]);
    }
  }

  // Resume context is now self-contained: the emitted deopt code sets
  // gWJResumeScriptPtr/EnvPtr/NArgs/NLocals itself, so no setup is needed here.
  // The function's runtime environment (for MFunctionEnvironment): stash it so the
  // JIT'd code reads the correct closure env at entry (no GC before it reads it).
  gWJCurrentEnv = uint32_t(uintptr_t(static_cast<void*>(envChain)));
  // GECKO_WJ_MAGICPROBE diag: dump env + its raw slot bits at each JIT entry of
  // a target-line script (env-corruption hunting).
  static int envProbe = getenv("GECKO_WJ_ENVPROBE") ? atoi(getenv("GECKO_WJ_ENVPROBE")) : 0;
  if (envProbe && script && script->lineno() == uint32_t(envProbe)) {
    uint64_t s32 = envChain ? *reinterpret_cast<uint64_t*>(
        reinterpret_cast<char*>(envChain) + 32) : 0;
    fprintf(stderr, "[wj-envprobe] entry %s:%u env=%p slot32=%016llx argcnt=%u\n",
            script->filename() ? script->filename() : "?", script->lineno(),
            (void*)envChain, (unsigned long long)s32, argc);
  }
  // Actual arg count for this invocation: the JIT'd code snapshots it at entry for
  // ArgumentsLength. Set here (the C++ entry trampoline) just like gWJCurrentEnv, so a
  // top-level / PBL->JIT entry communicates the real argc (the JIT->JIT paths set it
  // themselves before transferring control).
  gWJCallArgc = argc;
  if (js::wasm::kWJEHABI) {
    // EHABI trampoline reads the boxed callee from gWJScratch[kWJCalleeSlot] (ABI slot 1).
    // The param is currently unused (Callee/HomeObject still bake the canonical fn); stage
    // the canonical function so it's well-defined. Wiring Callee/HomeObject to this param +
    // threading the real runtime callee here is the Option-B-consumer follow-up.
    JSFunction* f = script->function();
    gWJScratch[js::wasm::kWJCalleeSlot] =
        f ? JS::ObjectValue(*f).asRawBits() : JS::UndefinedValue().asRawBits();
  }
  // CACHED: this is the per-call JIT entry (millions of calls on gbemu); a bare
  // getenv here cost ~3% (getenv+strncmp) in the profile. See [[wjhelp-getenv-tax]].
  static int envDbg = getenv("GECKO_WJ_ENVDBG") ? 1 : 0;
  if (envDbg) {
    fprintf(stderr, "[wb-envdbg] ENTRY env=%u %s:%u\n", gWJCurrentEnv,
            script ? script->filename() : "?", script ? script->lineno() : 0);
  }
  double ptr = double(uintptr_t(static_cast<void*>(gWJScratch)));
  // Direct entry: call the trampoline via a C function pointer (call_indirect
  // through the MAIN indirect table) -- no wasm->JS->wasm hop. Falls back to the
  // JS shim if registration failed.
  double flag;
  gWJDidResume = false;
  gWJExecDepth++;
  // Record the entry-time exitFP + activation for wjhelp's re-install (task #60);
  // nesting-safe via save/restore around the call.
  uint8_t* wjPrevEntryExitFP = gWJEntryExitFP;
  void* wjPrevEntryAct = gWJEntryAct;
  {
    JSContext* ecx = js::TlsContext.get();
    js::Activation* eact = ecx ? ecx->activation() : nullptr;
    if (eact && eact->isJit()) {
      gWJEntryExitFP = eact->asJit()->jsExitFP();
      gWJEntryAct = (void*)eact;
    } else {
      gWJEntryExitFP = nullptr;
      gWJEntryAct = nullptr;
    }
  }
  if (e.directIdx >= 0) {
    typedef double (*WJTrampFn)(double);
    WJTrampFn fp = reinterpret_cast<WJTrampFn>(uintptr_t(e.directIdx));
    flag = fp(ptr);
  } else {
    flag = wasmhost_call(e.handle, 0, &ptr, 1);
  }
  gWJEntryExitFP = wjPrevEntryExitFP;
  gWJEntryAct = wjPrevEntryAct;
  gWJExecDepth--;
  // Convention: 0.0 = result ready in gWJScratch (normal completion OR sound
  // resume); 1.0 = an exception is pending (a call/resume threw) -> propagate;
  // 2.0 = the function DID NOT RUN (entry GGG mismatch / depth valve) ->
  // return 0 so the caller executes it in PBL (no resume nesting).
  if (flag == 2.0) {
    static int depthDbg2 = getenv("GECKO_WJ_DEPTHDBG") ? 1 : 0;
    if (depthDbg2) {
      static uint64_t flag2s = 0;
      if ((++flag2s % 2000) == 1)
        fprintf(stderr, "[wj-depth2] flag2s=%llu jitDepth=%d\n",
                (unsigned long long)flag2s, gWJJitDepth);
    }
    e.gggDeopts++;
    if (e.gggDeopts == 8) {
      // Back off, then recompile once things stabilize (fresh generation).
      e.state = WJEntry::State::Cold;
      e.nextTry = e.observes + 2000;
      e.gggDeopts = 0;
    }
    return 0;
  }
  if (flag != 0.0) {
    gWJWasmDeopts++;
    // GECKO_WJ_DEPTHDBG: a "threw" flag with NO pending exception is a contract
    // violation somewhere below (helper returned 1.0 without throwing) and
    // surfaces as an undebuggable silent failure -- name the site.
    static int flagDbg = getenv("GECKO_WJ_DEPTHDBG") ? 1 : 0;
    if (flagDbg) {
      JSContext* dcx = js::TlsContext.get();
      if (dcx && !JS_IsExceptionPending(dcx)) {
        fprintf(stderr,
                "[wj-flag1-noexc] flag=%f script=%s:%u lastDeoptOp=%u\n", flag,
                script->filename() ? script->filename() : "?",
                unsigned(script->lineno()), gWJLastDeoptOp);
      }
    }
    return 2;  // propagate pending exception
  }
  // Deopt-storm handling: a function deopting on (almost) every entry compiled
  // with stale/monomorphic IC info. RECOMPILE it (Ion-style) -- by now PBL has
  // populated its ICs with the shapes that caused the deopts, so Warp should emit
  // a polymorphic/megamorphic load instead of the failing monomorphic GuardShape,
  // keeping it in JIT (the goal: almost never engage PBL). Cap recompiles; only
  // after repeated failure fall back to PBL (last resort).
  if (gWJDidResume) {
    e.deopts++;
    e.deoptsTotal++;
    // GuardGlobalGeneration storm control (see gggDeopts in WJEntry).
    if (gWJLastDeoptOp == js::wasm::gWJOpGuardGlobalGeneration) {
      e.gggDeopts++;
      if (e.gggDeopts >= 10) {
        e.state = WJEntry::State::Failed;
        return valveReturnEarly(retBits);
      }
      if (e.gggDeopts == 4 && e.recompiles == 0) {
        e.recompiles++;
        e.state = WJEntry::State::Cold;   // recompile picks the fresh generation
        e.nextTry = e.observes + 1;
        return valveReturnEarly(retBits);
      }
    }
    static int valveN = getenv("GECKO_WJ_VALVEN") ? atoi(getenv("GECKO_WJ_VALVEN")) : 300;
    static int noDeoptValve = getenv("GECKO_WJ_NODEOPTVALVE") ? 1 : 0;
    // Ion-style trigger. Ion (frequentBailoutThreshold, default 10) recompiles after an
    // ABSOLUTE COUNT of fixable bailouts, NOT a deopt RATE -- so a moderate-rate storm
    // still gets invalidated+recompiled with fresh types. Our OLD gate fired only at
    // >50% deopt (deopts > jitRuns+1), so cdjs's ~49%-deopt type-storms (Int32 unbox of
    // double -- now that the IC enriches, see WJ_DBLWARM) NEVER recompiled and stormed
    // forever. Add a COUNT trigger (deopts >= countN) with a ~>=20% rate floor (so
    // low-rate net-positive fns are untouched). Count-triggered fns get a FRESH
    // (type-respec) recompile that reads the enriched ICs and re-types Double; the >50%
    // rate path keeps the forceMega (shape) recompile. GECKO_WJ_NORECOMPILEN -> rate-only.
    static int countN = getenv("GECKO_WJ_NORECOMPILEN") ? 0
                        : (getenv("GECKO_WJ_RECOMPILEN")
                               ? atoi(getenv("GECKO_WJ_RECOMPILEN"))
                               : 1500);
    bool rateGate = e.deopts > (e.jitRuns + 1);
    // Rate floor for the count trigger. Ion uses an ABSOLUTE bailout count (no rate
    // floor); our default 25% floor protects low-rate net-positive fns from churn,
    // but it also blocks recompiling a low-rate-but-EXPENSIVE-resume deopter (e.g.
    // crypto-sha1 safe_add: ~2838 deopts on a param that's sometimes undefined, but
    // <<25% rate -> never re-typed). GECKO_WJ_RECOMPILE_RATE tunes the floor %.
    static int rateFloor =
        getenv("GECKO_WJ_RECOMPILE_RATE") ? atoi(getenv("GECKO_WJ_RECOMPILE_RATE")) : 25;
    bool countGate = countN > 0 && e.deopts >= uint32_t(countN) &&
                     uint64_t(e.deopts) * 100 >= uint64_t(e.jitRuns) * uint64_t(rateFloor);
    // The count-trigger respec recompile of a SHAPE-deopt-dominated fn re-bakes a stale
    // GC const (acorn tokenizer mis-parse, task #27). Block it for such fns -- the
    // count-trigger is for numeric/type re-typing (crypto/cdjs), where the rate gate
    // (forceMega) handles shape storms anyway. DEFAULT-ON; GECKO_WJ_NODEOPTCAT reverts.
    // Validated: fixes acorn (real-app tokenizer mis-parse), suite perf-neutral.
    static int deoptCat = getenv("GECKO_WJ_NODEOPTCAT") ? 0 : 1;
    if (deoptCat && countGate && !rateGate && e.shapeDeoptDom) countGate = false;
    // CRITICAL (2026-06-29): the function ALREADY RAN this call (flag==0 -> its
    // result is in gWJScratch[kWJResultSlot], and any side effects -- incl. those
    // completed by the deopt-RESUME -- are done). The valve paths below change e.state
    // for the NEXT call, but must NOT make THIS call return 0: a 0 return tells the PBL
    // caller "I didn't run it" so PBL RE-RUNS the function from pc=0, DUPLICATING the
    // side effects (hash-map _createHashedEntry ran twice -> 2 orphan entries -> result
    // 210 short; minimal repro /tmp/seh_repro.js SEH=50002). So return the completed
    // result (return 1) even when the valve fires.
    auto valveReturn = [&]() -> int {
      constexpr int noVR = 0;  // valve-result fix is permanent
      if (noVR) return 0;
      *retBits = gWJScratch[js::wasm::kWJResultSlot];
      return 1;
    };
    if (e.deopts >= uint32_t(valveN) && (rateGate || countGate) &&
        !noDeoptValve && !e.recompileDone) {
      // A storming fn with an alwaysBails block deopts from that COLD-IC block;
      // recompiling reproduces it identically (verified: deltablue 741's recompiled
      // MIR is byte-identical), so recompiling is pointless AND the recompile churn
      // corrupts state (deltablue wrong-value/GC-crash). Go straight to PBL. A
      // storming fn WITHOUT one deopts from a stale monomorphic guard -> recompile
      // to specialize it polymorphic (crypto 3.1x). (Cold branches that NEVER fire,
      // e.g. navier lin_solve's a===0, never reach here: deopts stays 0.)
      if (getenv("GECKO_WJ_VALVEDBG")) {
        fprintf(stderr, "[wj-valve] %s:%u hasAB=%d shapeDom=%d rateG=%d cntG=%d deopts=%u jitRuns=%u recomp=%u len=%u -> %s\n",
                script->filename() ? script->filename() : "?",
                unsigned(script->lineno()), e.hasAlwaysBails, e.shapeDeoptDom,
                rateGate, countGate, e.deopts, e.jitRuns,
                e.recompiles, unsigned(script->length()),
                e.hasAlwaysBails ? "FAIL" : "recompile");
      }
      if (e.hasAlwaysBails) {
        e.state = WJEntry::State::Failed;
        return valveReturn();
      }
      // A TINY function that storms via polymorphic guards (hasAB=0): mega-recompile
      // makes it deopt-free but adds per-access IC overhead, and for a tiny body the
      // JIT call/GC-root/boxing overhead exceeds the body's work -> PBL is faster
      // (deltablue's constraint accessors: output/input/isSatisfied, ~30-80 bytecodes,
      // called polymorphically from the planner). Larger storming bodies (crypto
      // bignum loops) still recompile-mega (the loop work dominates the overhead, so
      // staying in JIT wins). Tunable via GECKO_WJ_TINYBAIL (default 120; 0 disables).
      {
        static int tinyBail = getenv("GECKO_WJ_TINYBAIL")
                                  ? atoi(getenv("GECKO_WJ_TINYBAIL"))
                                  : 0;
        if (tinyBail > 0 && e.recompiles == 0 &&
            script->length() < uint32_t(tinyBail)) {
          e.state = WJEntry::State::Failed;
          return valveReturn();
        }
      }
      // Self-hosted builtins that storm: route to PBL instead of force-mega
      // recompiling. The forceMega recompile of a shape-deopt-dominated self-hosted
      // builtin (e.g. ObjectOrReflectDefineProperty, called polymorphically across
      // many object shapes) MISCOMPILES -> wrong result (ubo: defineProperty stops
      // setting hntrieContainer once IsCallable lets it compile). Self-hosted builtins
      // are NOT the crypto bignum loops (those are user JS), so PBL-routing them does
      // not regress crypto; they run correctly + fast enough in PBL. This un-blocks
      // enabling IsCallable. GECKO_WJ_SHPBL gates it (test); off = old forceMega.
      {
        // DEFAULT-ON 2026-06-30 (GECKO_WJ_NOSHPBL reverts): the forceMega recompile of a
        // shape-deopt-dominated self-hosted builtin miscompiles (ObjectOrReflectDefineProperty
        // -> ubo hntrieContainer undefined once IsCallable lets it compile). Routing self-hosted
        // storming fns to PBL is sound (PBL is the reference; can only affect perf) and does NOT
        // regress crypto (its bignum loops are user JS, still force-mega). Un-blocks IsCallable.
        static int shPbl = getenv("GECKO_WJ_NOSHPBL") ? 0 : 1;
        if (shPbl && script->function() &&
            script->function()->isSelfHostedOrIntrinsic()) {
          e.state = WJEntry::State::Failed;
          return valveReturn();
        }
      }
      // Mega-recompile budget. The FIRST storm triggers a megamorphic recompile
      // (GuardShape-guarded reads -> multi-shape EmitPropIC, no deopt on polymorphic
      // receivers). crypto's storming bignum loops become deopt-free this way and
      // stay fast in JIT. But if a function STILL reaches the storm threshold AFTER
      // being mega-recompiled, its deopts are from guards mega CAN'T convert
      // (BoxNonStrictThis / fallible Unbox / call-guards on a genuinely polymorphic
      // receiver -- deltablue's constraint accessors). More mega passes are
      // identical and pointless; the function is deopt-bound and SLOWER than PBL
      // (deltablue regressed to a timeout here -- ~30% of millions of calls deopt,
      // below the 50% re-trigger but enough to crawl). Fall back to PBL (the valve's
      // documented last resort). GECKO_WJ_MEGARETRIES tunes the budget (default 1).
      static int megaRetries = getenv("GECKO_WJ_MEGARETRIES")
                                   ? atoi(getenv("GECKO_WJ_MEGARETRIES"))
                                   : 1;
      if (int(e.recompiles) < megaRetries) {
        e.recompiles++;
        // Shape-storm (>50% rate) -> megamorphic recompile (multi-shape GuardShapeList).
        // Type-storm (count-triggered, moderate rate) -> FRESH recompile (no mega): just
        // re-read the now-enriched type ICs so Warp re-types the arith/compare Double.
        e.forceMega = rateGate;
        e.state = WJEntry::State::Cold;  // re-observe + recompile with fresh ICs
        e.handle = -1;
        e.directIdx = -1;
        // KEEP e.tblSlot: the recompile reuses it so callers' cached call ICs
        // (funPtr->slot, funPtr unchanged) keep resolving to the new module.
        // Resetting it to -1 allocated a NEW slot, leaving cached ICs pointing at
        // the stale old module -> deltablue wrong results.
        e.deopts = 0;
        e.jitRuns = 0;
        e.observes = 0;
        e.nextTry = 0;
        return valveReturn();
      }
      // Recompile didn't heal. A >50% storm is clearly net-negative -> fall to PBL. A
      // moderate-rate (count-triggered) storm STAYS in JIT (recompileDone stops further
      // triggering): failing it to PBL costs the JIT<->PBL boundary, measured WORSE than
      // the residual deopts (cdjs: compile-bail-to-PBL = 12.5s vs 11.3s deopting).
      if (rateGate) {
        e.state = WJEntry::State::Failed;
      } else {
        e.recompileDone = true;
      }
      return valveReturn();
    }
  } else {
    e.jitRuns++;
  }
  gWJWasmRuns++;
  WJStatsDumpTick();  // GECKO_WJ_STATSJSON: periodic stderr snapshot (browser-capturable)
  if (((gWJFastCalls+gWJSlowCalls) % 20000)==0 && (gWJFastCalls+gWJSlowCalls)>0 && (getenv("GECKO_WJWARP_DUMP")||getenv("GECKO_DEBUG_JIT"))) fprintf(stderr, "[wb-calls] fast=%llu slow=%llu\n", (unsigned long long)gWJFastCalls,(unsigned long long)gWJSlowCalls);
  if (((gWJWasmRuns + gWJWasmDeopts) % 5000) == 0 &&
      (getenv("GECKO_WJWARP_DUMP") || getenv("GECKO_DEBUG_JIT"))) {
    fprintf(stderr, "[wb-stats] wasm runs=%llu deopts=%llu\n",
            (unsigned long long)gWJWasmRuns, (unsigned long long)gWJWasmDeopts);
  }
  *retBits = gWJScratch[js::wasm::kWJResultSlot];

  if (verifyMut && !msnaps.empty() && mutCx) {
    JSContext* cx = mutCx;
    if (!cx->isExceptionPending()) {
      // Save the wasm's mutations (via the GC-rooted, updated object handles),
      // then restore originals so the PBL re-run sees the same starting state.
      for (auto& sp : msnaps) {
        js::NativeObject& o = mutRoots[sp.rootIdx].toObject().as<js::NativeObject>();
        for (uint32_t s = 0; s < sp.n; s++) sp.wasmAfter[s] = o.getSlot(s).asRawBits();
        for (uint32_t s = 0; s < sp.n; s++)
          o.setSlot(s, JS::Value::fromRawBits(sp.before[s]));
        // Dense elements: capture the wasm's post-state, then restore the
        // originals so the PBL re-run starts from the same array contents.
        uint32_t curLen = std::min<uint32_t>(o.getDenseInitializedLength(), 256);
        sp.denseWasmAfterLen = curLen;
        for (uint32_t d = 0; d < curLen; d++)
          sp.denseWasmAfter.push_back(o.getDenseElement(d).asRawBits());
        if (o.getDenseInitializedLength() >= sp.denseBeforeLen) {
          o.setDenseInitializedLength(sp.denseBeforeLen);
          for (uint32_t d = 0; d < sp.denseBeforeLen; d++)
            o.setDenseElement(d, JS::Value::fromRawBits(sp.denseBefore[d]));
        }
      }
      RootedValue fval(cx, JS::ObjectValue(*script->function()));
      RootedValue thisv(cx, JS::Value::fromRawBits(thisBits));
      JS::RootedValueVector av(cx);
      bool okv = av.reserve(argc);
      for (uint32_t i = 0; okv && i < argc; i++) av.infallibleAppend(argv[i]);
      RootedValue rv(cx);
      gWJVerifyReentry = true;
      bool okcall = okv && JS::Call(cx, thisv, fval, JS::HandleValueArray(av), &rv);
      gWJVerifyReentry = false;
      if (okcall) {
        for (auto& sp : msnaps) {
          js::NativeObject& o =
              mutRoots[sp.rootIdx].toObject().as<js::NativeObject>();
          for (uint32_t s = 0; s < sp.n; s++) {
            if (o.getSlot(s).asRawBits() != sp.wasmAfter[s]) {
              static int mn = 0;
              if (mn++ < 40) {
                JS::Value wv = JS::Value::fromRawBits(sp.wasmAfter[s]);
                JS::Value pv = o.getSlot(s);
                fprintf(stderr,
                        "[wb-MUT] %s:%u slot%u wasm=%g/%s pbl=%g/%s\n",
                        script->filename(), uint32_t(script->lineno()), s,
                        wv.isNumber() ? wv.toNumber() : -1,
                        wv.isObject() ? "obj" : wv.isUndefined() ? "undef" : "prim",
                        pv.isNumber() ? pv.toNumber() : -1,
                        pv.isObject() ? "obj" : pv.isUndefined() ? "undef" : "prim");
              }
            }
          }
          // Dense element divergence (the real montReduce signal: bignum limbs).
          uint32_t pblLen = std::min<uint32_t>(o.getDenseInitializedLength(), 256);
          if (pblLen != sp.denseWasmAfterLen) {
            static int dl = 0;
            if (dl++ < 40)
              fprintf(stderr, "[wb-MUTDENSE] %s:%u LEN wasm=%u pbl=%u\n",
                      script->filename(), uint32_t(script->lineno()),
                      sp.denseWasmAfterLen, pblLen);
          }
          uint32_t cmpLen = std::min(pblLen, sp.denseWasmAfterLen);
          for (uint32_t d = 0; d < cmpLen; d++) {
            if (o.getDenseElement(d).asRawBits() != sp.denseWasmAfter[d]) {
              static int dm = 0;
              if (dm++ < 40) {
                JS::Value wv = JS::Value::fromRawBits(sp.denseWasmAfter[d]);
                JS::Value pv = o.getDenseElement(d);
                fprintf(stderr,
                        "[wb-MUTDENSE] %s:%u elem%u wasm=%g/%s pbl=%g/%s\n",
                        script->filename(), uint32_t(script->lineno()), d,
                        wv.isNumber() ? wv.toNumber() : -1,
                        wv.isObject() ? "obj" : wv.isUndefined() ? "undef" : "prim",
                        pv.isNumber() ? pv.toNumber() : -1,
                        pv.isObject() ? "obj" : pv.isUndefined() ? "undef" : "prim");
              }
            }
          }
        }
      }
    }
  }

  // Differential verifier (GECKO_WJ_VERIFY): re-run the call in the interpreter
  // and compare. Only sound for pure functions (re-runs side effects), so a
  // debugging aid, not a correctness mechanism.
  static int verify = getenv("GECKO_WJ_VERIFY") ? 1 : 0;
  if (verify) {
    uint64_t wasmRes = *retBits;
    JSContext* cx = js::TlsContext.get();
    if (cx && !cx->isExceptionPending()) {
      RootedValue fval(cx, JS::ObjectValue(*script->function()));
      RootedValue thisv(cx, JS::Value::fromRawBits(thisBits));
      JS::RootedValueVector av(cx);
      bool okv = av.reserve(argc);
      for (uint32_t i = 0; okv && i < argc; i++) av.infallibleAppend(argv[i]);
      RootedValue rv(cx);
      // ROOT the wasm result across the re-run: JS::Call GCs, which would move a
      // GC-thing result and leave `wasmRes` (raw bits) a stale pointer -> false
      // "garbage class" mismatch. The Rooted value is updated by the GC.
      RootedValue wasmRooted(cx, JS::Value::fromRawBits(wasmRes));
      gWJVerifyReentry = true;  // force the re-run to use PBL (not re-JIT)
      bool okcall = okv && JS::Call(cx, thisv, fval, JS::HandleValueArray(av), &rv);
      gWJVerifyReentry = false;
      if (okcall) {
        JS::Value wasmV = wasmRooted;
        bool gcResult = wasmV.isGCThing() || rv.isGCThing();
        // Strong wrongness signal: JIT returned a primitive/undefined where the
        // interpreter produced an object (or different class), OR both are native
        // objects with differing fixed-slot contents. Catches "returns undefined
        // / wrong-field Vector" -- the raytrace failure mode.
        bool kindMismatch =
            (wasmV.isObject() != rv.isObject()) ||
            (wasmV.isObject() && rv.isObject() &&
             wasmV.toObject().getClass() != rv.toObject().getClass());
        static int slotdbg = getenv("GECKO_WJ_SLOTDBG") ? 1 : 0;
        if (!kindMismatch && wasmV.isObject() && rv.isObject() &&
            wasmV.toObject().is<js::NativeObject>() &&
            rv.toObject().is<js::NativeObject>()) {
          js::NativeObject& wo = wasmV.toObject().as<js::NativeObject>();
          js::NativeObject& io = rv.toObject().as<js::NativeObject>();
          uint32_t n = std::min(wo.slotSpan(), io.slotSpan());
          for (uint32_t s = 0; s < n && s < 8; s++) {
            if (wo.getSlot(s).asRawBits() != io.getSlot(s).asRawBits()) {
              kindMismatch = true;
              if (slotdbg) {
                JS::Value wv = wo.getSlot(s), iv = io.getSlot(s);
                auto ty = [](JS::Value v) {
                  return v.isInt32() ? "i32" : v.isDouble() ? "dbl"
                         : v.isObject() ? "obj" : v.isUndefined() ? "undef"
                         : v.isBoolean() ? "bool" : "other";
                };
                fprintf(stderr,
                        "[wb-SLOT] %s:%u slot%u wasm=%g/%s interp=%g/%s\n",
                        script->filename(), uint32_t(script->lineno()), s,
                        wv.isNumber() ? wv.toNumber() : -999, ty(wv),
                        iv.isNumber() ? iv.toNumber() : -999, ty(iv));
              }
              break;
            }
          }
        }
        if ((!gcResult && rv.asRawBits() != wasmRes) || kindMismatch) {
          // GECKO_WJ_VERIFYSKIP=<lineno>: suppress a known-benign mismatch line so
          // it doesn't flood the cap (e.g. chai isProxyEnabled:41229 undef-vs-false).
          static int vskip = getenv("GECKO_WJ_VERIFYSKIP") ? atoi(getenv("GECKO_WJ_VERIFYSKIP")) : -1;
          static int vn = 0;
          if (uint32_t(script->lineno()) != uint32_t(vskip) && vn++ < 4000)
            fprintf(stderr,
                    "[wb-VERIFY] MISMATCH %s:%u srcStart=%u argc=%u wasm=%s/%.17g interp=%s/%.17g\n",
                    script->filename() ? script->filename() : "?",
                    uint32_t(script->lineno()), uint32_t(script->sourceStart()), argc,
                    wasmV.isObject() ? wasmV.toObject().getClass()->name
                                     : (wasmV.isUndefined() ? "undef"
                                        : wasmV.isNumber() ? "num" : "prim"),
                    wasmV.isNumber() ? wasmV.toNumber() : -1.0,
                    rv.isObject() ? rv.toObject().getClass()->name
                                  : (rv.isUndefined() ? "undef"
                                     : rv.isNumber() ? "num" : "prim"),
                    rv.isNumber() ? rv.toNumber() : -1.0);
        }
      } else if (cx->isExceptionPending()) {
        cx->clearPendingException();
      }
    }
  }
  return 1;
}

// Recognize the Prototype.js `Class.create` forwarding wrapper:
//   function() { this.initialize.apply(this, arguments); }
// raytrace builds every object through this single shared script, so each
// `new Vector/Color/Ray/...` runs: VM construct -> wrapper in PBL -> arguments
// object -> fun_apply native -> initialize in PBL (~15-22% of raytrace, profiled).
// Detection is SOUND by construction: the only operations allowed touch `this`
// or `arguments`, the only property names are "initialize" and "apply", and there
// is exactly one call -- so the script can ONLY be `this.initialize.apply(this,
// arguments)` (no other operand is reachable). Any other op rejects (-> no opt).
// Result cached per-script (constructs are hot). All wrappers share one script.
static std::unordered_map<JSScript*, char>* gWJFwdCache = nullptr;

// Per-class (keyed by wrapper fn ptr) construct cache: a forwarding `new X` repeats
// the SAME resolution every time (WJIsForwardingWrapper + GetProperty("initialize")
// + gEntries lookup ~5% of raytrace). Cache the resolved initialize (compiled
// handle + nargs + env) so repeats skip all lookups and call directly. Key (wrapper
// fn) and env (module scope) are tenured/stable; a moved/evicted key just misses ->
// re-resolve (sound). [GC sweep TODO: trace gWJCC_env or re-validate.]
static const int kWJCtorCacheN = 32;
static uint32_t gWJCC_key[kWJCtorCacheN] = {0};  // wrapper fn ptr (0 = empty)
static int gWJCC_handle[kWJCtorCacheN] = {0};    // initialize compiled handle
static uint32_t gWJCC_nargs[kWJCtorCacheN] = {0};
static uint32_t gWJCC_env[kWJCtorCacheN] = {0};  // initialize environment ptr
static js::SharedShape* gWJCC_shape[kWJCtorCacheN] = {0};  // `this` shape (0=use CreateThisFromIon)
static uint8_t gWJCC_allocKind[kWJCtorCacheN] = {0};       // `this` gc::AllocKind
static bool WJIsForwardingWrapper(JSContext* cx, JSScript* s) {
  if (!s->function() || s->length() > 96) return false;
  if (!gWJFwdCache) gWJFwdCache = new std::unordered_map<JSScript*, char>();
  auto cached = gWJFwdCache->find(s);
  if (cached != gWJFwdCache->end()) return cached->second != 0;
  bool sawInit = false, sawApply = false, sawArgs = false;
  int calls = 0;
  bool ok = true;
  static int dump = getenv("GECKO_WJ_DUMPCTOR") ? 1 : 0;
  jsbytecode* end = s->codeEnd();
  for (jsbytecode* pc = s->code(); pc < end;
       pc += js::GetBytecodeLength(pc)) {
    JSOp op = JSOp(*pc);
    if (dump) fprintf(stderr, "[wj-ctorop] %s:%u %s\n",
                      s->filename() ? s->filename() : "?",
                      unsigned(s->lineno()), js::CodeName(op));
    switch (op) {
      case JSOp::FunctionThis:
      case JSOp::ImplicitThis:
      case JSOp::Callee:
      case JSOp::NewTarget:
      case JSOp::CheckThis:
      case JSOp::CheckThisReinit:
      case JSOp::Dup:
      case JSOp::Dup2:
      case JSOp::DupAt:
      case JSOp::Swap:
      case JSOp::Pick:
      case JSOp::Unpick:
      case JSOp::Pop:
      case JSOp::PopN:
      case JSOp::Undefined:
      case JSOp::Void:
      case JSOp::GetLocal:
      case JSOp::SetLocal:
      case JSOp::InitLexical:
      case JSOp::GetArg:
      case JSOp::SetArg:
      case JSOp::GetAliasedVar:
      case JSOp::SetAliasedVar:
      case JSOp::InitAliasedLexical:
      case JSOp::RetRval:
      case JSOp::Return:
      case JSOp::GetRval:
      case JSOp::SetRval:
      case JSOp::JumpTarget:
      case JSOp::Nop:
      case JSOp::Lineno:
      case JSOp::DebugCheckSelfHosted:
        break;
      case JSOp::Arguments:
        sawArgs = true;
        break;
      case JSOp::GetProp: {
        js::PropertyName* nm = s->getName(pc);
        if (nm && js::StringEqualsAscii(nm, "initialize")) sawInit = true;
        else if (nm && js::StringEqualsAscii(nm, "apply")) sawApply = true;
        else ok = false;  // any other property name -> not the wrapper
        break;
      }
      case JSOp::Call:
      case JSOp::CallContent:
      case JSOp::CallIgnoresRv:
        calls++;
        break;
      default:
        ok = false;  // anything else (stores, branches, other calls) -> reject
        break;
    }
  }
  bool match = ok && sawInit && sawApply && sawArgs && calls == 1;
  (*gWJFwdCache)[s] = match ? 1 : 0;
  return match;
}

// wasm -> C++ trampoline imported by JIT'd modules ("m"."help"). Returns 0.0 on
// success (result, if any, in gWJScratch[kWJResultSlot]) or 1.0 if it threw.
// ---- JIT execution tracer (a real debugger, not printfs): WJH_TRACE(value) appends
// to a ring buffer; dumped (oldest->newest) at process exit. The backend emits these
// at block entries of GECKO_WJ_TRACEFN=<lineno>, so we get the exact block-execution
// PATH of a JIT'd function with the bug live -- non-perturbing of JS warmup (wasm-level).
static const uint32_t kWJTraceN = 1u << 18;
static uint32_t* gWJTraceBuf = nullptr;
static uint64_t gWJTraceCount = 0;
static void WJDumpTraceAtExit() {
  if (!gWJTraceBuf || !gWJTraceCount) return;
  uint64_t n = gWJTraceCount < kWJTraceN ? gWJTraceCount : kWJTraceN;
  uint64_t start = gWJTraceCount < kWJTraceN ? 0 : (gWJTraceCount & (kWJTraceN - 1));
  fprintf(stderr, "[wj-trace] %llu entries (showing last %llu, oldest->newest):\n",
          (unsigned long long)gWJTraceCount, (unsigned long long)n);
  for (uint64_t i = 0; i < n; i++) {
    if (i % 24 == 0) fprintf(stderr, "[wj-trace] ");
    fprintf(stderr, "%u ", gWJTraceBuf[(start + i) & (kWJTraceN - 1)]);
    if (i % 24 == 23) fprintf(stderr, "\n");
  }
  fprintf(stderr, "\n");
}

static void WJDumpVTraceAtExit();  // fwd decl (defined below, near wjhelpImpl)

// JS-callable (shell builtin `wjTraceDump()`): dump the JIT execution-trace ring now
// (e.g. from a bench's catch block, right at the failure).
extern "C" EMSCRIPTEN_KEEPALIVE void WJTraceDumpNow() {
  WJDumpTraceAtExit();
  WJDumpVTraceAtExit();  // per-def value tracer (GECKO_WJ_VALUETRACE)
  // GECKO_WJ_DUMPADDR=a,b,...: dump the i64 at each (const-pool slot) address, decoded.
  if (const char* da = getenv("GECKO_WJ_DUMPADDR")) {
    char buf[256]; snprintf(buf, sizeof buf, "%s", da);
    for (char* tok = strtok(buf, ","); tok; tok = strtok(nullptr, ",")) {
      uintptr_t a = strtoull(tok, nullptr, 0);
      uint64_t v = *reinterpret_cast<uint64_t*>(a);
      uint32_t tag = uint32_t(v >> 32);
      fprintf(stderr, "[wj-dumpaddr] @%lu = %016llx tag=%08x ptr=%08x\n",
              (unsigned long)a, (unsigned long long)v, tag, uint32_t(v));
    }
  }
}

// Per-def VALUE tracer (GECKO_WJ_VALUETRACE=<lineno>, helper kind 251). The backend
// stores each def's value bits to gWJTraceVal then calls wjhelp(251, defId); we record
// (defId, value) oldest->newest and dump at exit. This is the "real debugger" for
// miscompiles: read the trace to find the FIRST def whose value diverges from expected
// (or diff two runs: feature on/off). Complements the block-PATH tracer (kind 250).
// gWJTraceVal is defined with the other gWJ* globals above (js::wasm namespace).
static const uint32_t kWJVTraceN = 1u << 20;
static uint32_t* gWJVTDef = nullptr;
static int64_t* gWJVTVal = nullptr;
static uint64_t gWJVTCount = 0;
static void WJDumpVTraceAtExit() {
  if (!gWJVTDef || !gWJVTCount) return;
  uint64_t cap = getenv("GECKO_WJ_VTRACEN")
                     ? uint64_t(atoi(getenv("GECKO_WJ_VTRACEN")))
                     : 400;
  uint64_t n = gWJVTCount < kWJVTraceN ? gWJVTCount : kWJVTraceN;
  if (n > cap) n = cap;  // show the last `cap` entries by default (small repros)
  uint64_t start = (gWJVTCount - n) & (kWJVTraceN - 1);
  fprintf(stderr, "[wj-vtrace] %llu def-values (showing last %llu):\n",
          (unsigned long long)gWJVTCount, (unsigned long long)n);
  for (uint64_t i = 0; i < n; i++) {
    uint64_t j = (start + i) & (kWJVTraceN - 1);
    int64_t v = gWJVTVal[j];
    fprintf(stderr, "[wj-vtrace] def%u = %lld (0x%llx)\n", gWJVTDef[j],
            (long long)v, (unsigned long long)v);
  }
}

static double wjhelpImpl(double kindF, double siteF);
// Crash forensics: the last helper kind/site entered, readable AFTER a wasm trap
// kills the instance (the embed process survives; __exec calls WJDumpCrashState).
// One i32 store per helper call -- negligible next to the call itself.
static volatile uint32_t gWJLastHelpKind = 0;
static volatile uint32_t gWJLastHelpSite = 0;
extern "C" void WJDumpWalkRing();  // JitFrames.cpp (task #60)
extern "C" EMSCRIPTEN_KEEPALIVE void WJDumpCrashState() {
  WJDumpWalkRing();
  fprintf(stderr,
          "[wj-crashstate] lastHelpKind=%u lastHelpSite=%u callArgc=%u "
          "execDepth=%d rootSP=%u resumeActive=%d\n",
          gWJLastHelpKind, gWJLastHelpSite, gWJCallArgc, gWJExecDepth,
          gWJRootSP, int(gWJResumeActive));
  // Mid-resume crash (resumeActive=1): the gWJResume* globals still describe the
  // crashing resume -- print each frame's identity (fn, pc, JSOp, layout) so the
  // deopt site is named without a re-run. Guard the script pointer (may be
  // garbage if the spill itself was corrupt).
  if (gWJResumeActive && gWJResumeNFrames &&
      gWJResumeNFrames <= js::wasm::kWJMaxResumeFrames) {
    for (uint32_t f = 0; f < gWJResumeNFrames; f++) {
      JSScript* ds =
          reinterpret_cast<JSScript*>(uintptr_t(gWJResumeScriptPtr[f]));
      const char* fn = "?";
      unsigned ln = 0;
      const char* opname = "?";
      if (ds && js::gc::IsCellPointerValid(ds)) {
        fn = ds->filename() ? ds->filename() : "?";
        ln = unsigned(ds->lineno());
        if (gWJResumePc[f] < ds->length())
          opname = js::CodeName(JSOp(*(ds->code() + gWJResumePc[f])));
      }
      fprintf(stderr,
              "[wj-crashstate] resume frame%u/%u %s:%u pc=%u op=%s depth=%u "
              "nargs=%u nlocals=%u off=%u\n",
              f, gWJResumeNFrames, fn, ln, gWJResumePc[f], opname,
              gWJResumeStackDepth[f], gWJResumeNArgs[f], gWJResumeNLocals[f],
              gWJResumeValsOff[f]);
    }
    fprintf(stderr,
            "[wj-crashstate] lastDeoptOp=%s(%u) valsCount=%u actualArgc=%u\n",
            js::wasm::WJMirOpName(gWJLastDeoptOp), gWJLastDeoptOp,
            gWJResumeValsCount, gWJResumeActualArgc);
    // Decode each spilled Value: raw bits, cell validity, and for strings the
    // header (flags/length/rope-ness + child validity). Distinguishes "slot
    // holds a stale/garbage pointer" from "valid rope whose CHILDREN are
    // stale" -- the latter means the rope was BUILT from stale operands.
    uint32_t nv = gWJResumeValsCount <= 32 ? gWJResumeValsCount : 32;
    for (uint32_t i = 0; i < nv; i++) {
      JS::Value v = JS::Value::fromRawBits(gWJResumeVals[i]);
      if (!v.isGCThing()) continue;
      js::gc::Cell* cell = v.toGCThing();
      bool valid = js::gc::IsCellPointerValid(cell);
      fprintf(stderr, "[wj-crashstate] vals[%u]=%016llx kind=%s cellValid=%d",
              i, (unsigned long long)gWJResumeVals[i],
              v.isString() ? "str" : v.isObject() ? "obj" : "gc", int(valid));
      if (valid && v.isString()) {
        JSString* s = v.toString();
        fprintf(stderr, " len=%zu rope=%d", s->length(), int(!s->isLinear()));
        if (!s->isLinear()) {
          JSRope* r = &s->asRope();
          JSString* lc = r->leftChild();
          JSString* rc = r->rightChild();
          fprintf(stderr, " lc=%p(v=%d) rc=%p(v=%d)", (void*)lc,
                  int(js::gc::IsCellPointerValid(lc)), (void*)rc,
                  int(js::gc::IsCellPointerValid(rc)));
        }
      }
      fprintf(stderr, "\n");
    }
  }
  // Prop-IC crash: name the property + the staged receiver bits (identifies the
  // source access; scratch[0] is the object the JIT staged for the miss path).
  if (gWJLastHelpKind == js::wasm::WJH_PROPIC ||
      gWJLastHelpKind == js::wasm::WJH_SETPROPIC) {
    uint32_t site = gWJLastHelpSite;
    JS::PropertyKey id = JS::PropertyKey::fromRawBits(uintptr_t(gWJPropKey[site]));
    if (id.isAtom()) {
      JSAtom* a = id.toAtom();
      char buf[64] = {0};
      size_t n = 0;
      if (a->hasLatin1Chars()) {
        JS::AutoCheckCannotGC nogc;
        const JS::Latin1Char* ch = a->latin1Chars(nogc);
        for (; n < a->length() && n < 63; n++) buf[n] = char(ch[n]);
      }
      fprintf(stderr, "[wj-crashstate] propic key='%s' scratch0=%016llx\n", buf,
              (unsigned long long)gWJScratch[0]);
    } else {
      fprintf(stderr, "[wj-crashstate] propic key=<non-atom> scratch0=%016llx\n",
              (unsigned long long)gWJScratch[0]);
    }
  }
}
extern "C" EMSCRIPTEN_KEEPALIVE double wjhelp(double kindF, double siteF) {
  gWJLastHelpKind = uint32_t(kindF);
  gWJLastHelpSite = uint32_t(siteF);
  // Save + restore the activation's jsExitFP around EVERY helper: a helper that
  // runs JS (WJH_CALL/RESUME/getters) executes PBL whose VMFrame dtors pop their
  // exit frames WITHOUT restoring jsExitFP -- benign inside PBL (the next VMFrame
  // re-sets it) but fatal once control returns to wasm-JIT code and a later
  // helper GCs: TraceJitFrames then walks the stale exitFP into reused stack
  // holding boxed values (the GCZeal deep-chain crash, task #60). Restoring here
  // puts back the OUTER entry's still-valid exit frame.
  JSContext* wjcx = js::TlsContext.get();
  js::Activation* wjact = wjcx ? wjcx->activation() : nullptr;
  uint8_t* wjSavedExitFP =
      (wjact && wjact->isJit()) ? wjact->asJit()->jsExitFP() : nullptr;
  if (wjact && wjact->isJit() && gWJEntryExitFP &&
      (void*)wjact == gWJEntryAct) {
    wjact->asJit()->setJSExitFP(gWJEntryExitFP);
  }
  double r = wjhelpImpl(kindF, siteF);
  if (wjact && wjact->isJit() && wjSavedExitFP) {
    wjact->asJit()->setJSExitFP(wjSavedExitFP);
  }
  // Contract check (GECKO_WJ_DEPTHDBG): 1.0 means "threw" -- returning it with
  // NO pending exception surfaces as an undebuggable silent failure.
  static int helpDbg = getenv("GECKO_WJ_DEPTHDBG") ? 1 : 0;
  if (helpDbg && r == 1.0) {
    JSContext* dcx = js::TlsContext.get();
    if (dcx && !JS_IsExceptionPending(dcx)) {
      fprintf(stderr, "[wj-help-noexc] kind=%d site=%d\n", int(kindF),
              int(siteF));
    }
  }
  return r;
}
static double wjhelpImpl(double kindF, double siteF) {
  int kind = int(kindF);
  // DEBUG (GECKO_WJ_VALIDATEHELPER): validate GC-tagged Values staged in the
  // scratch slots (and the call callee) on EVERY helper entry. Catches a stale
  // pointer at the moment JIT'd code hands it to the runtime -- with the full
  // JS backtrace -- instead of crashing far downstream (task #64: terser rope
  // corruption observed 3 frames away from its source). Ropes also get their
  // children checked (a rope BUILT from a stale operand has garbage kids).
  static int validateHelper = getenv("GECKO_WJ_VALIDATEHELPER") ? 1 : 0;
  if (validateHelper) {
    auto bad = [&](uint64_t bits, const char* what, int slot) {
      JS::Value v = JS::Value::fromRawBits(bits);
      if (!v.isGCThing()) return false;
      js::gc::Cell* cell = v.toGCThing();
      if (!js::gc::IsCellPointerValid(cell)) return true;
      if (v.isString() && !v.toString()->isLinear()) {
        JSRope* r = &v.toString()->asRope();
        if (!js::gc::IsCellPointerValid(r->leftChild()) ||
            !js::gc::IsCellPointerValid(r->rightChild()))
          return true;
      }
      return false;
    };
    // Only kinds whose scratch slots are freshly staged (leftover slots from
    // earlier calls are stale BY DESIGN -- checking them would false-positive).
    int nCheck = 0;
    switch (kind) {
      case js::wasm::WJH_CALL: nCheck = int(gWJCallArgc); break;
      case js::wasm::WJH_BINARYARITH:
      case js::wasm::WJH_COMPARE:
      case js::wasm::WJH_SETPROPIC:
      case js::wasm::WJH_CHARCODEAT: nCheck = 2; break;
      case js::wasm::WJH_PROPIC:
      case js::wasm::WJH_LINEARIZE:
      case js::wasm::WJH_TOSTRING: nCheck = 1; break;
      default: nCheck = 0; break;
    }
    bool hit = false;
    for (int i = 0; i < nCheck && i < 62; i++)
      if (bad(gWJScratch[i], "scratch", i)) {
        fprintf(stderr, "[wj-validate] STALE scratch[%d]=%016llx kind=%d site=%d\n",
                i, (unsigned long long)gWJScratch[i], kind, int(siteF));
        hit = true;
      }
    if (kind == js::wasm::WJH_CALL && bad(gWJCallCallee, "callee", -1)) {
      fprintf(stderr, "[wj-validate] STALE callee=%016llx\n",
              (unsigned long long)gWJCallCallee);
      hit = true;
    }
    if (hit) {
      if (kind == js::wasm::WJH_CALL)
        fprintf(stderr, "[wj-validate] caller line=%u (gWJCallSiteLine[site])\n",
                gWJCallSiteLine[uint32_t(siteF) % js::wasm::kWJCallSites]);
      JSContext* vcx = js::TlsContext.get();
      if (vcx) js::DumpBacktrace(vcx);
      abort();
    }
  }
  if (kind == 252) {  // WJH_CHECKVAL (GECKO_WJ_VALSTORE): validate a to-be-stored
    // Value in gWJHelpVal; siteF = the storing fn's script line. Abort on stale.
    JS::Value v = JS::Value::fromRawBits(gWJHelpVal);
    bool badv = false;
    if (v.isGCThing()) {
      js::gc::Cell* c = v.toGCThing();
      badv = !js::gc::IsCellPointerValid(c);
      if (!badv && v.isString() && !v.toString()->isLinear()) {
        JSRope* r = &v.toString()->asRope();
        badv = !js::gc::IsCellPointerValid(r->leftChild()) ||
               !js::gc::IsCellPointerValid(r->rightChild());
      }
    }
    if (badv) {
      fprintf(stderr, "[wj-valstore] STALE stored value=%016llx line=%d defId=%u\n",
              (unsigned long long)gWJHelpVal, int(siteF), gWJHelpSlot);
      abort();
    }
    return 0.0;
  }
  if (kind == 250) {  // WJH_TRACE: record siteF in the ring buffer. NB: kind 250 is a
    // dedicated sentinel -- it MUST NOT collide with any WJHelpKind enum value.
    // (Was 40, which COLLIDED with WJH_ISARRAY=40: every Array.isArray hit this
    // branch, never computed, and returned a STALE gWJScratch result -> wrong
    // Array.isArray, breaking ubo/webtooling. 2026-07-02 fix.)
    if (!gWJTraceBuf) { gWJTraceBuf = new uint32_t[kWJTraceN](); atexit(WJDumpTraceAtExit); }
    gWJTraceBuf[gWJTraceCount++ & (kWJTraceN - 1)] = uint32_t(int64_t(siteF));
    return 0.0;
  }
  if (kind == 252) {  // ARGC0DBG trap: a usesArgc fn entered with a ZERO argc snapshot.
    static uint64_t n = 0;
    if (n++ < 12) {
      JS::Value cal = JS::Value::fromRawBits(gWJCallCallee);
      JSFunction* cf = cal.isObject() && cal.toObject().is<JSFunction>()
                           ? &cal.toObject().as<JSFunction>() : nullptr;
      JSScript* cs = cf && cf->hasBaseScript() && cf->baseScript()->hasBytecode()
                         ? static_cast<JSScript*>(cf->baseScript()) : nullptr;
      fprintf(stderr,
              "[wj-argc0] gWJCallArgc=%u callee=%s:%u execDepth=%d env=%u\n",
              gWJCallArgc,
              cs && cs->filename() ? cs->filename() : "?",
              cs ? unsigned(cs->lineno()) : 0, gWJExecDepth, gWJCurrentEnv);
    }
    return 0.0;
  }
  if (kind == 251) {  // WJH_VTRACE: record (defId=siteF, gWJTraceVal). See value tracer.
    if (!gWJVTDef) {
      gWJVTDef = new uint32_t[kWJVTraceN]();
      gWJVTVal = new int64_t[kWJVTraceN]();
      atexit(WJDumpVTraceAtExit);
    }
    uint64_t idx = gWJVTCount++ & (kWJVTraceN - 1);
    gWJVTDef[idx] = uint32_t(int64_t(siteF));
    gWJVTVal[idx] = gWJTraceVal;
    return 0.0;
  }
  JSContext* cx = js::TlsContext.get();
  if (!cx) return 1.0;

  static int helpHist = getenv("GECKO_WJ_HELPHIST") ? 1 : 0;
  if (helpHist) {
    static uint64_t hc[40] = {0};
    static uint64_t tot = 0;
    if (kind >= 0 && kind < 40) hc[kind]++;
    if ((++tot % 200000) == 0) {
      static const char* nm[37] = {"?",        "RESUME",  "CALL",
                                   "SETSLOT",  "GETPROP", "SETPROP",
                                   "GETELEM",  "INSTOF",  "ARRPUSH",
                                   "ARRPOP",   "CRTHIS",  "CONSTRUCT",
                                   "POSTBAR",  "PREBAR",  "PROPIC",  "SETPROPIC",
                                   "NEWPLAIN", "NEWARROBJ","NEWARR", "BINARITH",
                                   "UNARITH",  "GETNAME", "TOPROPKEY","CHARCODEAT",
                                   "FROMCC",   "TOSTRING","COMPARE", "NEWOBJECT",
                                   "BINDNAME", "GROWSLOTS","TOINT32","INSTOFPROTO",
                                   "LAMBDA",   "TYPEOFIS","NEWCALLOBJ","CTORALLOC",
                                   "DBGPTR"};
      fprintf(stderr, "[wb-helphist] %llu calls:", (unsigned long long)tot);
      for (int k = 0; k < 37; k++)
        if (hc[k]) fprintf(stderr, " %s=%llu", nm[k], (unsigned long long)hc[k]);
      fprintf(stderr, "\n");
    }
  }

  if (kind == js::wasm::WJH_RESUME) {
    // Depth watermark (GECKO_WJ_DEPTHDBG): confirm/deny resume-nesting stack
    // exhaustion (uncatchable silent exits).
    static int depthDbg = getenv("GECKO_WJ_DEPTHDBG") ? 1 : 0;
    if (depthDbg) {
      static int maxDepth = 0;
      static uint64_t resumes = 0;
      resumes++;
      if (gWJExecDepth > maxDepth) {
        maxDepth = gWJExecDepth;
        if (maxDepth % 50 == 0)
          fprintf(stderr, "[wj-depth] execDepth=%d resumes=%llu lastOp=%u\n",
                  maxDepth, (unsigned long long)resumes, gWJLastDeoptOp);
      }
      if ((resumes % 20000) == 0)
        fprintf(stderr, "[wj-depth] resumes=%llu depth=%d\n",
                (unsigned long long)resumes, gWJExecDepth);
    }
    gWJDidResume = true;  // this JIT entry deopted (safety-valve accounting)
    // NEVER put a bare getenv() in this path -- WJH_RESUME fires on EVERY deopt
    // (cdjs: 3.3M/iter), and getenv walks the environ array each call. Cache all the
    // debug-flag reads in statics (computed once). This was ~5 bare getenvs/resume =
    // ~16M getenv/iter for a deopt-heavy bench -- pure overhead. See [[wjhelp-getenv-tax]].
    static int dbgJit = getenv("GECKO_DEBUG_JIT") ? 1 : 0;
    static int siteHist = getenv("GECKO_WJ_SITEHIST") ? 1 : 0;
    static int entryDump = getenv("GECKO_WJ_ENTRYDUMP") ? 1 : 0;
    static int deoptHist = getenv("GECKO_WJ_DEOPTHIST") ? 1 : 0;
    static int deoptHistN = getenv("GECKO_WJ_DEOPTHISTN") ? 1 : 0;
    static int stormLine = getenv("GECKO_WJ_STORMLINE") ? 1 : 0;
    static int stormFirst = getenv("GECKO_WJ_STORMFIRST") ? 1 : 0;
    if(dbgJit){static uint64_t c=0; if((++c%5000)==0) fprintf(stderr,"[wb-resume-count] %llu\n",(unsigned long long)c);}
    if(siteHist){static uint64_t c=0; uint64_t thr=getenv("GECKO_WJ_SITEHISTN")?atoi(getenv("GECKO_WJ_SITEHISTN")):200000; if((++c%thr)==0) js::wasm::WJDumpDeoptSiteHist();}
    if(entryDump){static uint64_t c=0; if((++c%1000000)==0) WJDumpEntries();}
    if (deoptHist) {
      static uint64_t dc = 0;
      uint64_t dmod = deoptHistN ? 10 : 200;
      if ((++dc % dmod) == 0) {
        fprintf(stderr, "[wb-deopthist] after %llu resumes:\n",
                (unsigned long long)dc);
        for (uint32_t o = 0; o < js::wasm::kWJNumOps; o++)
          if (gWJDeoptByOp[o])
            fprintf(stderr, "[wb-deopthist]   %s(op#%u) = %u\n",
                    js::wasm::WJMirOpName(o), o, gWJDeoptByOp[o]);
      }
    }
    if (stormLine) {
      static uint64_t sc = 0;
      ++sc;
      uint64_t mod = stormFirst ? 1 : 2000;
      uint64_t cap = stormFirst ? 40 : ~0ull;
      if (sc <= cap && (sc % mod) == 0 && gWJResumeNFrames) {
        uint32_t lf = gWJResumeNFrames - 1;
        JSScript* ds =
            reinterpret_cast<JSScript*>(uintptr_t(gWJResumeScriptPtr[lf]));
        if (ds) {
          uint32_t pcoff = gWJResumePc[lf];
          const char* opname = "?";
          if (pcoff < ds->length()) {
            JSOp jop = JSOp(*(ds->code() + pcoff));
            opname = js::CodeName(jop);
          }
          fprintf(stderr, "[wj-storm] %s:%u pc=%u op=%s (resumes=%llu)\n",
                  ds->filename() ? ds->filename() : "?", unsigned(ds->lineno()),
                  pcoff, opname, (unsigned long long)sc);
        }
      }
    }
    // Multi-frame inline bailout: run frames innermost (0) -> outermost. Each
    // frame's return is threaded into the next outer frame's call-result stack
    // slot (the top of its expr stack at the resume-after-call point).
    uint32_t nframes = gWJResumeNFrames;
    if (nframes == 0 || nframes > js::wasm::kWJMaxResumeFrames) {
      if (getenv("GECKO_WJ_DEPTHDBG"))
        fprintf(stderr, "[wj-resume-noexc] bad nframes=%u\n", nframes);
      return 1.0;
    }
    // try/catch: capture + clear the in-error flag now so it can't leak to a later
    // (normal) deopt. Used for the innermost frame below.
    const uint32_t resumeErr = gWJResumeInError;
    gWJResumeInError = 0;
    // Beyond-formal actuals graft (see the backend spill): the PHYSICAL (outermost)
    // frame was spilled with only its formals in the arg region; if the function
    // entered with MORE actuals (arguments-using variadic), widen that region with
    // the staged actuals so the PBL frame sees the true argc (arguments.length /
    // arguments[i] beyond formals). Pure bit-moves inside the traced buffer, done
    // BEFORE any allocation. Consumes gWJResumeActualArgc.
    {
      uint32_t aa = gWJResumeActualArgc;
      gWJResumeActualArgc = 0;
      uint32_t lf = nframes - 1;
      uint32_t na = gWJResumeNArgs[lf];
      static int graftDbg = getenv("GECKO_WJ_GRAFTDBG") ? 1 : 0;
      if (graftDbg) {
        static uint64_t gn = 0;
        if (gn++ < 60)
          fprintf(stderr,
                  "[wj-graft] aa=%u na=%u lf=%u off=%u nlocals=%u depth=%u pc=%u\n",
                  aa, na, lf, gWJResumeValsOff[lf], gWJResumeNLocals[lf],
                  gWJResumeStackDepth[lf], gWJResumePc[lf]);
      }
      if (aa > na && aa <= js::wasm::kWJMaxArgs) {
        uint32_t shift = aa - na;
        uint32_t base = gWJResumeValsOff[lf] + 1 + na;
        uint32_t end = gWJResumeValsOff[lf] + 1 + na + gWJResumeNLocals[lf] +
                       gWJResumeStackDepth[lf];
        if (end + shift <= 1024) {
          memmove(&gWJResumeVals[base + shift], &gWJResumeVals[base],
                  size_t(end - base) * 8);
          for (uint32_t i = na; i < aa; i++) {
            gWJResumeVals[gWJResumeValsOff[lf] + 1 + i] = gWJResumeActuals[i];
          }
          gWJResumeNArgs[lf] = aa;
        }
      }
    }
    // Mark gWJResumeVals as a live GC-root region for the whole resume: the spilled
    // boxed pointers must survive the allocations (RootedValueVector reserve, PBL
    // frame setup, per-frame GC) between the deopt spill and the read of each frame's
    // slots. Count = highest slot used = last frame's off + (1+nargs+nlocals+depth).
    {
      uint32_t lf = nframes - 1;
      gWJResumeValsCount = gWJResumeValsOff[lf] + 1 + gWJResumeNArgs[lf] +
                           gWJResumeNLocals[lf] + gWJResumeStackDepth[lf];
      if (gWJResumeValsCount > 1024) gWJResumeValsCount = 1024;
      gWJResumeActive = true;
    }
    auto resumeGuard = mozilla::MakeScopeExit([] { gWJResumeActive = false; });
    // OSR cheap-resume (GECKO_WJ_OSR): a single-frame deopt landing AT a compiled
    // loop head re-enters the JIT there instead of interpreting the rest of the fn
    // in PBL. The OSR prologue (WasmJitBackend) restores this/args/locals from the
    // gWJResumeVals spill (frame 0, off 0 -- the only OSR-able layout) and jumps the
    // dispatch loop to the loop-head block. Churn guard: gWJOsrDepth bounds nested
    // re-entries so a head that immediately re-deopts falls to PBL instead of looping.
    {
      static int osrEnabled = getenv("GECKO_WJ_OSR") ? 1 : 0;
      static uint32_t osrMaxDepth =
          getenv("GECKO_WJ_OSRDEPTH") ? uint32_t(atoi(getenv("GECKO_WJ_OSRDEPTH"))) : 1;
      static int osrDbg = getenv("GECKO_WJ_OSRDBG") ? 1 : 0;
      if (osrDbg && nframes == 1) {
        static uint64_t cc = 0;
        if (++cc <= 30) {
          JSScript* ds =
              reinterpret_cast<JSScript*>(uintptr_t(gWJResumeScriptPtr[0]));
          if (ds) {
            WJEntry& de = EntryFor(ds);
            fprintf(stderr,
                    "[wj-osr-chk] %s:%u pc=%u off0=%u depth=%u state=%d dir=%d "
                    "ntgt=%zu",
                    ds->filename() ? ds->filename() : "?", unsigned(ds->lineno()),
                    gWJResumePc[0], gWJResumeValsOff[0], gWJOsrDepth,
                    int(de.state), de.directIdx, de.osrTargets.size());
            for (size_t i = 0; i < de.osrTargets.size() && i < 6; i++)
              fprintf(stderr, " t%zu=%u", i, de.osrTargets[i].first);
            fprintf(stderr, "\n");
          }
        }
      }
      if (osrEnabled && nframes == 1 && gWJResumeValsOff[0] == 0 &&
          gWJOsrDepth < osrMaxDepth) {
        JSScript* dscript =
            reinterpret_cast<JSScript*>(uintptr_t(gWJResumeScriptPtr[0]));
        if (dscript) {
          WJEntry& de = EntryFor(dscript);
          // Skip OSR when the actuals graft WIDENED the arg region
          // (gWJResumeNArgs[0] > compile-time nargs): the prologue restore
          // reads [this, nargs args, locals] at the compile-time layout, so a
          // widened frame would shift every local read.
          if (de.state == WJEntry::State::Compiled && de.directIdx >= 0 &&
              !de.osrTargets.empty() && gWJResumeNArgs[0] == de.nargs) {
            uint32_t pc = gWJResumePc[0];
            int blk = -1;
            for (auto& t : de.osrTargets)
              if (t.first == pc) { blk = int(t.second); break; }
            if (blk >= 0) {
              JSObject* envObj =
                  reinterpret_cast<JSObject*>(uintptr_t(gWJResumeEnvPtr[0]));
              if (!envObj && dscript->function())
                envObj = dscript->function()->environment();
              gWJCurrentEnv = uint32_t(uintptr_t(static_cast<void*>(envObj)));
              gWJOsrActive = 1;
              gWJOsrBlock = uint32_t(blk);
              double ptr = double(uintptr_t(static_cast<void*>(gWJScratch)));
              typedef double (*WJTrampFn)(double);
              WJTrampFn fp = reinterpret_cast<WJTrampFn>(uintptr_t(de.directIdx));
              gWJOsrDepth++;
              gWJExecDepth++;
              if (osrDbg) gWJOsrHits++;
              if (osrDbg)
                fprintf(stderr,
                        "[wj-osr-fire] %s:%u directIdx=%d blk=%d nargs=%u nlocals=%u "
                        "execDepth=%d\n",
                        dscript->filename() ? dscript->filename() : "?",
                        unsigned(dscript->lineno()), de.directIdx, blk, de.nargs,
                        de.nlocals, gWJExecDepth);
              double flag = fp(ptr);
              gWJExecDepth--;
              gWJOsrDepth--;
              gWJOsrActive = 0;  // belt-and-suspenders (prologue clears it too)
              if (osrDbg) {
                static uint64_t oc = 0;
                if ((++oc % 50000) == 0)
                  fprintf(stderr, "[wj-osr] hits=%llu (last %s:%u pc=%u blk=%d)\n",
                          (unsigned long long)gWJOsrHits,
                          dscript->filename() ? dscript->filename() : "?",
                          unsigned(dscript->lineno()), pc, blk);
              }
              // fp wrote the fn's return into gWJScratch[kWJResultSlot] on normal
              // completion (flag 0.0) or signalled a pending exception (1.0).
              return flag != 0.0 ? 1.0 : 0.0;
            }
          }
        }
      }
    }
    static int rdbg =
        (getenv("GECKO_WJWARP_DUMP") || getenv("GECKO_WJ_RESUMEDBG")) ? 1 : 0;
    static int rn = 0;
    uint64_t rbits = JS::UndefinedValue().asRawBits();
    bool haveInner = false;
    for (uint32_t f = 0; f < nframes; f++) {
      RootedScript script(
          cx, reinterpret_cast<JSScript*>(uintptr_t(gWJResumeScriptPtr[f])));
      if (script && !script->hasJitScript()) {
        // A GC (esp. gczeal) can DISCARD this script's JitScript between the deopt
        // spill and this resume. PBL needs the JitScript (it interprets the Baseline
        // ICs) to run the rest of the frame, so a missing one used to fail the resume
        // with a silent "threw"/NO-pending-exception (the deterministic gczeal-7,1
        // int/pow deopt crash: WJH_RESUME kind=1 -> [wj-resume-noexc] hasJit=0).
        // Recreate it (valid script; JitScript is derivable from bytecode, fresh cold
        // ICs are correct - PBL just re-warms them) instead of aborting. Mirrors the
        // WasmJitWarp compile-entry ensureHasJitScript path.
        if (cx->zone()->ensureJitZoneExists(cx)) {
          AutoRealm ar(cx, script);
          js::jit::AutoKeepJitScripts keep(cx);
          (void)script->ensureHasJitScript(cx, keep);
        }
      }
      if (!script || !script->hasJitScript()) {
        if (getenv("GECKO_WJ_DEPTHDBG"))
          fprintf(stderr, "[wj-resume-noexc] f=%u script=%p hasJit=%d\n", f,
                  (void*)script, script ? script->hasJitScript() : -1);
        return 1.0;
      }
      JSObject* envObj = reinterpret_cast<JSObject*>(uintptr_t(gWJResumeEnvPtr[f]));
      static int forceFuncEnv = getenv("GECKO_WJ_FORCEFUNCENV") ? 1 : 0;
      if (forceFuncEnv && script->function()) envObj = script->function()->environment();
      if (!envObj && script->function()) envObj = script->function()->environment();
      RootedObject env(cx, envObj);
      // Correct enclosing env for the frame's PBL prologue (see header). 0 ->
      // PBL falls back to the canonical func->environment().
      RootedObject enclosingEnv(
          cx, reinterpret_cast<JSObject*>(uintptr_t(gWJResumeEnclosingEnv[f])));
      // keepFrameEnv: reuse the spilled gWJResumeEnvPtr (the RP environmentChain
      // operand) as the resumed frame's env, skipping the Init path that rebuilds
      // ONLY the function-level env. The spilled env is the TRUE current env chain at
      // the deopt pc -- including any PUSHED LEXICAL/BLOCK envs (NewLexicalEnvironment
      // Object) AND the function's own CallObject. The Init path discards pushed
      // lexical envs, so a deopt INSIDE a `let`/`const` block then resumes with the
      // wrong scope -> aliased-var reads return garbage ("undefined is not a function"
      // in ubo's CSS generator: a closure read its sibling `let` from the function env
      // instead of the block env). Now that the spill is EmitObjPtr(RP env operand)
      // (reliable), keep it whenever present.
      constexpr int noKeepEnv = 0;  // keep-frame-env resume fix is permanent
      bool keepFrameEnv = !noKeepEnv && (gWJResumeEnvPtr[f] != 0);
      uint32_t nargs = gWJResumeNArgs[f];
      uint32_t nlocals = gWJResumeNLocals[f];
      uint32_t depth = gWJResumeStackDepth[f];
      uint32_t off = gWJResumeValsOff[f];
      uint32_t total = 1 + nargs + nlocals + depth;
      // Return-threading: an inlined CALLER frame (f>0) was reconstructed to resume
      // AFTER its call (pc advanced, stack = values-below-the-call-inputs + a result
      // slot). Thread the inner frame's return (rbits from the previous iteration)
      // into that result slot (the top of this frame's expr stack) so PBL continues
      // post-call WITHOUT re-executing the (possibly side-effecting) callee. This is
      // the real-bailout behavior; the old "re-execute the call" path double-ran
      // side-effecting inlined callees (deltablue incrementalAdd corruption).
      if (f > 0 && total >= 1) gWJResumeVals[off + total - 1] = rbits;
      (void)haveInner;
      // gWJResumeVals[off..] is ALREADY a rooted GC region (gWJResumeActive, count set
      // above) and a JS::Value is bit-identical to its uint64_t slot, so view it
      // DIRECTLY instead of copying every value into a per-resume RootedValueVector
      // (a heap alloc + a copy of `total` boxed values on EVERY deopt -- cdjs storms
      // 3.3M deopts/iter). PBL copies args/locals/stack onto its own frame at setup, so
      // later mutations of gWJResumeVals don't alias the running frame.
      const JS::Value* vals =
          reinterpret_cast<const JS::Value*>(&gWJResumeVals[off]);
      static int rchk = getenv("GECKO_WJ_RESUMECHK") ? 1 : 0;
      if (rchk) {
        for (uint32_t i = 0; i < total; i++) {
          JS::Value v = vals[i];
          bool ok = v.isObject() || v.isString() || v.isSymbol() ||
                    v.isBigInt() || v.isInt32() || v.isDouble() || v.isBoolean() ||
                    v.isNull() || v.isUndefined() || v.isMagic();
          if (!ok || (v.isGCThing() && (uintptr_t(v.toGCThing()) < 0x1000 ||
                                        (uintptr_t(v.toGCThing()) & 7)))) {
            const char* kind = i == 0 ? "this"
                               : i < 1 + nargs ? "arg"
                               : i < 1 + nargs + nlocals ? "local" : "stack";
            fprintf(stderr,
                    "[wj-badval] %s:%u f=%u slot=%u(%s) bits=%016llx gcthing=%d\n",
                    script->filename() ? script->filename() : "?",
                    unsigned(script->lineno()), f, i, kind,
                    (unsigned long long)v.asRawBits(), v.isGCThing());
          }
        }
      }
      uint64_t thisBits = vals[0].asRawBits();
      // RESUMEPROBE=<lineno>: at this fn's deopt-resume, log this.* (by property NAME,
      // engine-level/non-perturbing) BEFORE PBL runs the rest -> spot the state the JIT
      // prologue left wrong. Compare the sequence across invocations to find the anomaly.
      static int resumeProbe = getenv("GECKO_WJ_RESUMEPROBE") ? atoi(getenv("GECKO_WJ_RESUMEPROBE")) : -1;
      if (resumeProbe > 0 && uint32_t(script->lineno()) == uint32_t(resumeProbe)) {
        JS::Value tv = JS::Value::fromRawBits(thisBits);
        static uint64_t pc = 0;
        if (tv.isObject()) {
          JS::RootedObject to(cx, &tv.toObject());
          JS::RootedValue p(cx), s(cx), e(cx), lte(cx), ty(cx);
          JS_GetProperty(cx, to, "pos", &p); JS_GetProperty(cx, to, "value", &s);
          JS_GetProperty(cx, to, "exprAllowed", &e); JS_GetProperty(cx, to, "type", &ty);
          const char* tl = "?"; void* typtr = nullptr; int binop = -999;
          if (ty.isObject()) { JS::RootedObject tyo(cx, &ty.toObject()); typtr = (void*)&ty.toObject(); JS::RootedValue lab(cx), bo(cx);
            if (JS_GetProperty(cx, tyo, "label", &lab) && lab.isString()) {
              JS::RootedString ls(cx, lab.toString());
              JS::UniqueChars c = JS_EncodeStringToUTF8(cx, ls); if (c) { static char buf[32]; snprintf(buf,sizeof buf,"%s",c.get()); tl=buf; } }
            if (JS_GetProperty(cx, tyo, "binop", &bo)) binop = bo.isInt32()?bo.toInt32():(bo.isNullOrUndefined()?-1:-2); }
          int posv = p.isInt32()?p.toInt32():-1;
          if (posv >= 9685 && posv <= 9720)  // failing region only
            fprintf(stderr, "[wj-resumeprobe] :%u pos=%d type=%s typtr=%p binop=%d exprAllowed=%d val=%s\n",
                  unsigned(script->lineno()), posv, tl, typtr, binop,
                  e.isBoolean()?e.toBoolean():-1, s.isString()?"str":(s.isUndefined()?"undef":"?"));
        }
      }
      const JS::Value* args = vals + 1;
      const uint64_t* locals =
          reinterpret_cast<const uint64_t*>(vals + 1 + nargs);
      const uint64_t* stack =
          reinterpret_cast<const uint64_t*>(vals + 1 + nargs + nlocals);
      if (rdbg && rn < 4000000 &&
          (!getenv("GECKO_WJ_RESUMEDBGLINE") ||
           uint32_t(script->lineno()) ==
               uint32_t(atoi(getenv("GECKO_WJ_RESUMEDBGLINE"))))) {
        rn++;
        fprintf(stderr,
                "[wb-resume] frame %u/%u %s:%u pc=%u(%s) nargs=%u nlocals=%u depth=%u this=%s",
                f, nframes, script->filename() ? script->filename() : "?",
                uint32_t(script->lineno()), gWJResumePc[f],
                js::CodeName(JSOp(*(script->code() + gWJResumePc[f]))),
                nargs, nlocals, depth,
                vals[0].isObject() ? (vals[0].toObject().is<JSFunction>() ? "fn" : "obj")
                                   : (vals[0].isUndefined() ? "undef" : "prim"));
        for (uint32_t i = 0; i < nargs && i < 6; i++) {
          JS::Value av = vals[1 + i];
          if (av.isInt32()) fprintf(stderr, " a%u=i%d", i, av.toInt32());
          else if (av.isDouble()) fprintf(stderr, " a%u=d%g", i, av.toDouble());
          else fprintf(stderr, " a%u=%s", i, av.isObject() ? "obj" : "prim");
        }
        for (uint32_t i = 0; i < nlocals && i < 6; i++) {
          JS::Value lv = vals[1 + nargs + i];
          if (lv.isInt32()) fprintf(stderr, " L%u=i%d", i, lv.toInt32());
          else if (lv.isDouble()) fprintf(stderr, " L%u=d%g", i, lv.toDouble());
          else fprintf(stderr, " L%u=%s", i,
                       lv.isUndefined() ? "undef" : lv.isObject() ? "obj" : "prim");
        }
        for (uint32_t i = 0; i < depth && i < 6; i++) {
          JS::Value sv = vals[1 + nargs + nlocals + i];
          if (sv.isInt32()) fprintf(stderr, " S%u=i%d", i, sv.toInt32());
          else if (sv.isDouble()) fprintf(stderr, " S%u=d%g", i, sv.toDouble());
          else if (sv.isBoolean()) fprintf(stderr, " S%u=b%d", i, sv.toBoolean());
          else if (sv.isString()) {
            JSString* s = sv.toString();
            size_t len = s->length();
            char16_t c0 = 0;
            JS::RootedString rs(cx, s);
            if (len >= 1) JS_GetStringCharAt(cx, rs, 0, &c0);
            fprintf(stderr, " S%u=str[len=%zu c0=%d '%c']", i, len, int(c0),
                    (c0 >= 32 && c0 < 127) ? char(c0) : '?');
          }
          else fprintf(stderr, " S%u=%s", i,
                       sv.isUndefined() ? "undef" : sv.isNull() ? "null" : sv.isObject()
                       ? (sv.toObject().is<JSFunction>() ? "fn" : "obj") : "prim");
        }
        // GECKO_WJ_RESUMEDISAS: walk the bytecode from the resume pc so the exact
        // resumed op sequence is visible (no disassembler native in the embed).
        static int rdisas = getenv("GECKO_WJ_RESUMEDISAS") ? 1 : 0;
        if (rdisas) {
          jsbytecode* p = script->code() + gWJResumePc[f];
          jsbytecode* endp = script->codeEnd();
          fprintf(stderr, "\n  [ops]");
          for (int k = 0; k < 25 && p < endp; k++) {
            JSOp jop = JSOp(*p);
            fprintf(stderr, " %u:%s", unsigned(p - script->code()),
                    js::CodeName(jop));
            p += js::GetBytecodeLength(p);
          }
          for (uint32_t i = 0; i < nlocals && i < 8; i++) {
            JS::Value lv = vals[1 + nargs + i];
            fprintf(stderr, " Lbits%u=%016llx", i,
                    (unsigned long long)lv.asRawBits());
          }
        }
        // montReduce x_array sanity: S0=x_array(obj), S1=write index. Flag if the
        // index is out of the dense range or the array length looks corrupt.
        if (getenv("GECKO_WJ_XARRDBG") && depth >= 2) {
          JS::Value xa = vals[1 + nargs + nlocals + 0];
          JS::Value iv = vals[1 + nargs + nlocals + 1];
          if (xa.isObject() && xa.toObject().is<js::NativeObject>()) {
            js::NativeObject* no = &xa.toObject().as<js::NativeObject>();
            uint32_t il = no->getDenseInitializedLength();
            fprintf(stderr, "  [xarr] initLen=%u idx=%s denseCap~%u\n", il,
                    iv.isInt32() ? std::to_string(iv.toInt32()).c_str()
                                 : (iv.isDouble() ? "dbl" : "?"),
                    no->getDenseCapacity());
          }
        }
        fprintf(stderr, "\n");
      }
      // GECKO_WJ_RESARRDBG: inspect the dense elements of local L0 (the result
      // array in the reloop repros) at resume ENTRY, and again on the resume's
      // return value below -- splits "array corrupted before the resume" from
      // "corrupted by the resume".
      static int resArrDbg = getenv("GECKO_WJ_RESARRDBG") ? 1 : 0;
      if (resArrDbg && nlocals > 0) {
        JS::Value l0 = vals[1 + nargs];
        if (l0.isObject() && l0.toObject().is<js::NativeObject>()) {
          js::NativeObject* no = &l0.toObject().as<js::NativeObject>();
          uint32_t il = no->getDenseInitializedLength();
          int selfIdx = -1;
          for (uint32_t i2 = 0; i2 < il; i2++) {
            const JS::Value& ev = no->getDenseElement(i2);
            if (ev.isObject() && &ev.toObject() == &l0.toObject()) { selfIdx = int(i2); break; }
          }
          uint64_t lastBits = il ? no->getDenseElement(il - 1).asRawBits() : 0;
          fprintf(stderr, "[wj-resarr] PRE il=%u last=%016llx selfIdx=%d\n", il,
                  (unsigned long long)lastBits, selfIdx);
        }
      }
      // try/catch: error-mode resume for the innermost (throwing) frame -> PBL runs the
      // catch via HandleException. gWJResumeInError was set by emitted code before this
      // WJH_RESUME for an in-try exception deopt.
      bool inErr = (resumeErr != 0) && (f == 0);
      JSFunction* runtimeCallee =
          reinterpret_cast<JSFunction*>(uintptr_t(gWJResumeCalleeFn[f]));
      if (getenv("GECKO_WJ_RESUMECALLEEDBG")) {
        JSScript* sp = reinterpret_cast<JSScript*>(uintptr_t(gWJResumeScriptPtr[f]));
        bool scrValid = sp && js::gc::IsCellPointerValid(reinterpret_cast<js::gc::Cell*>(sp));
        JSFunction* rc = runtimeCallee;
        bool rcValid = rc && js::gc::IsCellPointerValid(reinterpret_cast<js::gc::Cell*>(rc));
        void* rcScript = nullptr;
        bool rcIsFun = false, rcInterp = false;
        if (rcValid) {
          rcIsFun = rc->is<JSFunction>();
          if (rcIsFun) {
            rcInterp = rc->isInterpreted();
            if (rcInterp && rc->hasBaseScript()) rcScript = (void*)rc->baseScript();
          }
        }
        bool scriptMatch = rcScript && rcScript == (void*)script.get();
        if (!scriptMatch && rcInterp && rc->hasBaseScript()) {
          // identify BOTH scripts by filename:line using the SAFE BaseScript accessors
          // (no asJSScript() -- crashes on lazy). resume vs rooted-callee: caller/callee?
          js::BaseScript* rbs = rc->baseScript();
          fprintf(stderr,
                  "[wj-rc-MISMATCH] f=%u nframes=%u resume=%s:%u callee=%s:%u "
                  "resumeNargs=%u calleeNargs=%u pc=%u rootSP=%u\n",
                  f, gWJResumeNFrames,
                  script->filename() ? script->filename() : "?",
                  unsigned(script->lineno()),
                  rbs->filename() ? rbs->filename() : "?", unsigned(rbs->lineno()),
                  unsigned(script->function() ? script->function()->nargs() : 999),
                  unsigned(rc->nargs()), gWJResumePc[f], gWJRootSP);
        }
        fprintf(stderr,
                "[wj-resumecallee] f=%u calleeFn=%08x scriptPtr=%08x scrValid=%d "
                "paramScript=%p paramScrValid=%d rcValid=%d rcIsFun=%d rcInterp=%d "
                "rcScript=%p SCRIPT_MATCH=%d rootSP=%u resumeActive=%d\n",
                f, (unsigned)gWJResumeCalleeFn[f], (unsigned)gWJResumeScriptPtr[f],
                int(scrValid), (void*)script.get(),
                int(script.get() && js::gc::IsCellPointerValid(
                                        reinterpret_cast<js::gc::Cell*>(script.get()))),
                int(rcValid), int(rcIsFun), int(rcInterp), rcScript,
                int(scriptMatch), gWJRootSP, int(gWJResumeActive));
      }
      if (!js::pbl::WasmJitResumeViaPBL(cx, script, thisBits, args, nargs, env,
                                        locals, nlocals, gWJResumePc[f], &rbits,
                                        stack, depth, enclosingEnv, keepFrameEnv,
                                        inErr, runtimeCallee)) {
        if (getenv("GECKO_WJ_DEPTHDBG")) {
          JSContext* dcx = js::TlsContext.get();
          fprintf(stderr,
                  "[wj-resume-false] f=%u %s:%u pc=%u callee=%p pending=%d\n", f,
                  script->filename() ? script->filename() : "?",
                  unsigned(script->lineno()), gWJResumePc[f],
                  (void*)runtimeCallee,
                  dcx ? int(JS_IsExceptionPending(dcx)) : -1);
        }
        return 1.0;  // resumed execution threw (uncaught) -> propagate
      }
      if (resArrDbg) {
        JS::Value rv = JS::Value::fromRawBits(rbits);
        if (rv.isObject() && rv.toObject().is<js::NativeObject>()) {
          js::NativeObject* no = &rv.toObject().as<js::NativeObject>();
          uint32_t il = no->getDenseInitializedLength();
          int selfIdx = -1;
          for (uint32_t i2 = 0; i2 < il; i2++) {
            const JS::Value& ev = no->getDenseElement(i2);
            if (ev.isObject() && &ev.toObject() == &rv.toObject()) { selfIdx = int(i2); break; }
          }
          uint64_t lastBits = il ? no->getDenseElement(il - 1).asRawBits() : 0;
          fprintf(stderr, "[wj-resarr] POST il=%u last=%016llx selfIdx=%d\n", il,
                  (unsigned long long)lastBits, selfIdx);
        }
      }
      static int db414 = getenv("GECKO_WJ_DB414") ? atoi(getenv("GECKO_WJ_DB414")) : -1;
      if (db414 >= 0 && uint32_t(script->lineno()) == uint32_t(db414)) {
        JS::Value tv = JS::Value::fromRawBits(thisBits);
        if (tv.isObject()) {
          JS::RootedObject to(cx, &tv.toObject());
          JS::RootedValue dir(cx), v1(cx), v2(cx);
          JS_GetProperty(cx, to, "direction", &dir);
          JS_GetProperty(cx, to, "v1", &v1);
          JS_GetProperty(cx, to, "v2", &v2);
          JS::Value rv = JS::Value::fromRawBits(rbits);
          void* rp = rv.isObject() ? (void*)&rv.toObject() : nullptr;
          void* p1 = v1.isObject() ? (void*)&v1.toObject() : nullptr;
          void* p2 = v2.isObject() ? (void*)&v2.toObject() : nullptr;
          static uint64_t dbc = 0;
          uint64_t* raw = reinterpret_cast<uint64_t*>(&tv.toObject());
          uint64_t s32 = raw[32 / 8];   // body's then-arm load: i64.load offset=32
          uint64_t s40 = raw[40 / 8];   // this.direction load: i64.load offset=40
          if ((++dbc) <= 30)
            fprintf(stderr, "[db414] this=%p dir=%d ret=%p v1bits=%llx v2bits=%llx "
                    "off32=%llx off40=%llx (off32==v2:%d off40==dir:%d)\n",
                    (void*)&tv.toObject(), dir.isInt32() ? dir.toInt32() : -999, rp,
                    (unsigned long long)v1.asRawBits(),
                    (unsigned long long)v2.asRawBits(),
                    (unsigned long long)s32, (unsigned long long)s40,
                    s32 == v2.asRawBits(), s40 == dir.asRawBits());
        }
      }
      haveInner = true;
    }
    gWJScratch[js::wasm::kWJResultSlot] = rbits;
    return 0.0;
  }

  if (kind == js::wasm::WJH_CALL) {
    RootedValue callee(cx, JS::Value::fromRawBits(gWJCallCallee));
    uint32_t argc = gWJCallArgc;

    // GECKO_WJ_MAGICPROBE=1: report a MAGIC-tagged Value leaking out of JIT'd
    // code as a call argument or `this` (e.g. an ELEMENTS_HOLE flowing into
    // set.add -> the webtooling MOZ_RELEASE_ASSERT(whyMagic()==why) trap).
    static int magicProbe = getenv("GECKO_WJ_MAGICPROBE") ? 1 : 0;
    if (magicProbe) {
      for (uint32_t a = 0; a <= argc && a <= js::wasm::kWJThisSlot; a++) {
        uint32_t slot = (a == argc) ? js::wasm::kWJThisSlot : a;
        JS::Value v = JS::Value::fromRawBits(gWJScratch[slot]);
        if (v.isMagic()) {
          const char* calleeName = "?";
          JS::UniqueChars nameBytes;
          if (callee.isObject() && callee.toObject().is<JSFunction>()) {
            JSFunction* f = &callee.toObject().as<JSFunction>();
            if (f->maybePartialDisplayAtom()) {
              nameBytes = JS_EncodeStringToUTF8(
                  cx, JS::RootedString(cx, f->maybePartialDisplayAtom()));
              if (nameBytes) calleeName = nameBytes.get();
            }
          }
          fprintf(stderr,
                  "[wj-magicprobe] MAGIC arg leaked: callee=%s %s=%u/%u why=%d\n",
                  calleeName, a == argc ? "this" : "arg", a, argc,
                  int(v.whyMagic()));
        }
      }
    }

    // Fast path: if the callee is itself a compiled WJ function, run its wasm
    // directly (its args + `this` are already staged in gWJScratch by the
    // caller's marshalling), skipping JS::Call's generic dispatch machinery.
    // GECKO_WJ_FORCESLOW (debug) skips this so callees re-enter via JS::Call ->
    // WasmJitRunCall, where the differential verifier can check them.
    static int forceSlow = getenv("GECKO_WJ_FORCESLOW") ? 1 : 0;
    if (!forceSlow && gEntries && callee.isObject() &&
        callee.toObject().is<JSFunction>()) {
      JSFunction* fun = &callee.toObject().as<JSFunction>();
      if (fun->isInterpreted() && fun->hasBytecode()) {
        JSScript* cs = fun->nonLazyScript();
        // Drive compilation of JIT-only callees: a function called ONLY from
        // already-compiled code never reaches the interpreter/PBL observe hook,
        // so without this it stays COLD forever and every call falls to the slow
        // JS::Call path (richards: 160k slow calls). Observing here warms it to
        // the threshold, compiles it, and lets subsequent calls take the direct
        // wasm path + fill the inline call IC.
        js::wasm::WasmJitObserveCall(cs);
        auto it = gEntries->find(cs);
        // argc < nargs is OK: the callee's register ABI reads nargs arg slots, so we
        // pad the missing ones with undefined below (exactly what the backend's inline
        // fast-call path does, 6302). The old `argc >= nargs` guard forced these
        // underflow calls (uBlock compileToFilter/getNodeFlags-class, ~52K/iter) to the
        // full JS::Call AND -- since the IC fill is inside this block -- never cached
        // them, so EVERY call slow-pathed. nargs <= kWJMaxArgs always holds for a
        // register-ABI-compiled callee.
        if (it != gEntries->end() &&
            it->second.state == WJEntry::State::Compiled &&
            it->second.nargs <= js::wasm::kWJMaxArgs &&
            it->second.handle >= 0) {  // invalid handle (post-valve reset / failed
                                       // recompile) -> fall to slow JS::Call, never
                                       // call_indirect a -1 handle (wasm "null function
                                       // or signature mismatch" trap). Matches the ctor
                                       // fast paths (handle>=0 guarded).
          WJEntry& ce = it->second;
          // Fill the caller's call-site IC so the next call goes direct
          // (call_indirect) with no helper hop. site is wjhelp's 2nd arg.
          uint32_t site = uint32_t(siteF);
          // Fill a free way of the polymorphic IC (or refresh the matching one).
          // Only if the callee has a real table slot; else stays on the slow path.
          static int noICFill = getenv("GECKO_WJ_NOICFILL") ? 1 : 0;
          if (noICFill) { /* force all calls through this C++ path (debug) */ }
          else if (site < kWJCallSites && ce.tblSlot >= 0) {
            uint32_t funPtr = uint32_t(uintptr_t(static_cast<void*>(fun)));
            uint32_t base = site * kWJCallWays;
            uint32_t w = 0;
            for (; w < kWJCallWays; w++) {
              if (gWJCallFn[base + w] == 0 || gWJCallFn[base + w] == funPtr) break;
            }
            if (w == kWJCallWays) w = 0;  // all ways full: evict way 0
            gWJCallFn[base + w] = funPtr;
            gWJCallTblIdx[base + w] = ce.tblSlot;
          }
          // Resume context is self-contained (emitted code sets it), so just run.
          gWJCurrentEnv = uint32_t(uintptr_t(static_cast<void*>(fun->environment())));
          static int dbgEnvCall = getenv("GECKO_WJ_ENVDBG") ? 1 : 0;
          if (dbgEnvCall) {
            js::BaseScript* dsc = fun->hasBaseScript() ? fun->baseScript() : nullptr;
            fprintf(stderr, "[wb-envdbg] WJH_CALL fun=%p env=%u %s:%u\n",
                    (void*)fun, gWJCurrentEnv, dsc ? dsc->filename() : "?",
                    dsc ? dsc->lineno() : 0);
          }
          // Underflow call: pad the missing arg slots with undefined (the callee's
          // register ABI reads nargs slots; the caller only stored argc).
          for (uint32_t a = argc; a < ce.nargs; a++)
            gWJScratch[a] = JS::UndefinedValue().asRawBits();
          // GECKO_WJ_ARGDBG: on the WJH_CALL direct-dispatch path, log argc vs
          // nargs + each staged arg's tag for a target callee line -- to catch a
          // stale trailing arg (the baseClone `stack`-arg empty-clone bug).
          static int argDbg = getenv("GECKO_WJ_ARGDBG") ? 1 : 0;
          if (argDbg && fun->hasBaseScript() &&
              fun->baseScript()->lineno() == uint32_t(getenv("GECKO_WJ_ARGDBG")
                  ? atoi(getenv("GECKO_WJ_ARGDBG")) : 0)) {
            static uint64_t nlog = 0;
            nlog++;
            uint32_t realEnv = uint32_t(uintptr_t(static_cast<void*>(fun->environment())));
            bool envMismatch = (gWJCurrentEnv != realEnv);
            if (nlog <= 12 || envMismatch) {
              fprintf(stderr,
                      "[wj-argdbg] #%llu callee@%u argc=%u nargs=%u env=%u realEnv=%u%s hdl=%d tbl=%d rootSP=%u depth=%d tags=",
                      (unsigned long long)nlog, fun->baseScript()->lineno(), argc,
                      ce.nargs, gWJCurrentEnv, realEnv,
                      envMismatch ? " ENVMISMATCH" : "", ce.handle, ce.tblSlot,
                      gWJRootSP, gWJExecDepth);
              for (uint32_t a = 0; a < ce.nargs && a < 8; a++) {
                JS::Value v = JS::Value::fromRawBits(gWJScratch[a]);
                fprintf(stderr, "%s ",
                        v.isUndefined() ? "undef"
                        : v.isObject() ? "obj"
                        : v.isInt32() ? "i32"
                        : v.isString() ? "str"
                        : v.isNull() ? "null"
                        : v.isMagic() ? "MAGIC" : "other");
              }
              fprintf(stderr, "\n");
            }
          }
          if (js::wasm::kWJEHABI)  // EHABI: stage the boxed callee for the trampoline.
            gWJScratch[js::wasm::kWJCalleeSlot] = JS::ObjectValue(*fun).asRawBits();
          double ptr = double(uintptr_t(static_cast<void*>(gWJScratch)));
          double flag = wasmhost_call(ce.handle, 0, &ptr, 1);
          gWJFastCalls++;
          if (flag == 2.0) {
            // Entry GuardGlobalGeneration mismatch: the callee did NOT run.
            // Mark it cold-with-backoff (recompile bakes the fresh generation)
            // and fall through to the slow JS::Call path, which executes it in
            // the interpreter/PBL -- no resume nesting.
            ce.gggDeopts++;
            ce.state = WJEntry::State::Cold;
            ce.nextTry = ce.observes + (ce.gggDeopts >= 4 ? 2000 : 50);
          } else {
            return flag;  // 0 = result in gWJScratch[result]; 1 = threw
          }
        }
        // FELL THROUGH to JS::Call though callee is interpreted-with-bytecode:
        // diagnose WHY (state/handle/tblSlot/nargs-vs-argc). [CTFDBG]
        static int ctfDbg = getenv("GECKO_WJ_CTFDBG") ? 1 : 0;
        if (ctfDbg) {
          static std::map<std::string, uint64_t> why;
          static uint64_t tw = 0;
          int st = (it == gEntries->end()) ? -2 : int(it->second.state);
          int hd = (it == gEntries->end()) ? -99 : it->second.handle;
          int ts = (it == gEntries->end()) ? -99 : it->second.tblSlot;
          uint32_t nn = (it == gEntries->end()) ? 0 : it->second.nargs;
          char k[96];
          snprintf(k, sizeof k, "%s:%u st=%d hd%s ts%s argc%s",
                   cs->filename() ? "f" : "?", unsigned(cs->lineno()), st,
                   hd >= 0 ? ">=0" : "<0", ts >= 0 ? ">=0" : "<0",
                   argc >= nn ? ">=n" : "<n");
          why[k]++;
          if ((++tw % 100000) == 0) {
            std::vector<std::pair<uint64_t, std::string>> v;
            for (auto& kv : why) v.push_back({kv.second, kv.first});
            std::sort(v.rbegin(), v.rend());
            fprintf(stderr, "[wb-ctfdbg] %llu interp-callee->JS::Call, top:",
                    (unsigned long long)tw);
            for (size_t i = 0; i < v.size() && i < 12; i++)
              fprintf(stderr, " [%s]=%llu", v[i].second.c_str(),
                      (unsigned long long)v[i].first);
            fprintf(stderr, "\n");
          }
        }
      }
    }

    // gWJScratch[0..kWJThisSlot] is GC-traced by WJTraceRoots, so hand JS::Call
    // non-owning Handles straight into it -- no per-call RootedValueVector copy or
    // thisv rooting (the per-call Rooted overhead was ~1.3% across ~1M slow calls).
    // JS::Call copies the args into the callee's frame before running it, and the
    // callee (native/PBL on this JS::Call fallback path) doesn't clobber gWJScratch.
    JS::HandleValue thisv = JS::HandleValue::fromMarkedLocation(
        reinterpret_cast<const JS::Value*>(&gWJScratch[js::wasm::kWJThisSlot]));
    JS::HandleValueArray argv = JS::HandleValueArray::fromMarkedLocation(
        argc, reinterpret_cast<const JS::Value*>(&gWJScratch[0]));
    RootedValue rval(cx);
    gWJSlowCalls++;
    // getenv is a hot-path tax here (~1.2M slow calls in ubo); cache all debug gates
    // in statics ([[wjhelp-getenv-tax]]).
    static int callProf = getenv("GECKO_WJ_CALLPROF") ? 1 : 0;
    static int callTrace = getenv("GECKO_WJ_CALLTRACE") ? atoi(getenv("GECKO_WJ_CALLTRACE")) : -1;
    static int calleeDbg = getenv("GECKO_WJ_CALLEEDBG") ? 1 : 0;
    if (callProf && (gWJSlowCalls % 500000) == 0)
      fprintf(stderr, "[wb-slowcall] %llu\n", (unsigned long long)gWJSlowCalls);
    if (callTrace >= 0 && uint32_t(siteF) == uint32_t(callTrace)) {
      static int n = 0;
      if (n++ < 60) {
        const char* cls = callee.isObject() ? callee.toObject().getClass()->name
                                            : (callee.isUndefined() ? "undef" : "prim");
        const char* tcls = thisv.isObject() ? thisv.toObject().getClass()->name : "prim";
        bool isFn = callee.isObject() && callee.toObject().is<JSFunction>();
        fprintf(stderr, "[calltrace] site=%u #%d calleeBits=%#llx cls=%s isFn=%d thisCls=%s thisBits=%#llx\n",
                uint32_t(siteF), n, (unsigned long long)gWJCallCallee, cls, isFn, tcls,
                (unsigned long long)gWJScratch[js::wasm::kWJThisSlot]);
      }
    }
    if (calleeDbg) {
      bool ok = callee.isObject() && callee.toObject().is<JSFunction>();
      if (!ok) {
        const char* cls = callee.isObject() ? callee.toObject().getClass()->name : "?";
        const char* tcls = thisv.isObject() ? thisv.toObject().getClass()->name : "prim";
        fprintf(stderr, "[calleedbg] BAD callee bits=%#llx calleeClass=%s argc=%u "
                "callerLine=%u thisClass=%s site=%u\n",
                (unsigned long long)gWJCallCallee, cls, argc,
                gWJCallSiteLine[uint32_t(siteF) % js::wasm::kWJCallSites], tcls,
                uint32_t(siteF));
      } else {
        JSFunction* f = &callee.toObject().as<JSFunction>();
        // sane fn? interpreted-with-script or native. flag a function whose
        // script/native looks corrupt.
        bool sane = (f->isNativeFun()) || (f->isInterpreted() && f->hasBaseScript());
        if (!sane)
          fprintf(stderr, "[calleedbg] INSANE fn=%p flags=%#x argc=%u\n",
                  (void*)f, f->flags().toRaw(), argc);
      }
    }
    static int callHist = getenv("GECKO_WJ_CALLHIST") ? 1 : 0;
    if (callHist) {
      static std::map<std::string, uint64_t> hist;
      static uint64_t tot = 0;
      char key[80];
      if (callee.isObject() && callee.toObject().is<JSFunction>()) {
        JSFunction* f = &callee.toObject().as<JSFunction>();
        JSAtom* a = f->maybePartialDisplayAtom();
        char nm[48] = "?";
        if (a && a->length() > 0) {
          JS::AutoCheckCannotGC nogc;
          size_t n = std::min<size_t>(a->length(), 47);
          if (a->hasLatin1Chars()) {
            const JS::Latin1Char* c = a->latin1Chars(nogc);
            for (size_t i = 0; i < n; i++) nm[i] = char(c[i]);
            nm[n] = 0;
          } else {
            const char16_t* c = a->twoByteChars(nogc);
            for (size_t i = 0; i < n; i++) nm[i] = char(c[i]);
            nm[n] = 0;
          }
        }
        snprintf(key, sizeof key, "%s:%s", f->isNativeFun() ? "N" : "J", nm);
      } else {
        snprintf(key, sizeof key, "non-fn");
      }
      hist[key]++;
      if ((++tot % 200000) == 0) {
        std::vector<std::pair<uint64_t, std::string>> v;
        for (auto& kv : hist) v.push_back({kv.second, kv.first});
        std::sort(v.rbegin(), v.rend());
        fprintf(stderr, "[wb-callhist] %llu slow calls, top:", (unsigned long long)tot);
        for (size_t i = 0; i < v.size() && i < 18; i++)
          fprintf(stderr, " %s=%llu", v[i].second.c_str(), (unsigned long long)v[i].first);
        fprintf(stderr, "\n");
      }
    }
    if (!JS::Call(cx, thisv, callee, argv, &rval)) {
      return 1.0;  // callee threw -> propagate
    }
    gWJScratch[js::wasm::kWJResultSlot] = rval.asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_DBGPTR) {
    // DEBUG: log the value the valNursery post-barrier check is about to chunk-load.
    // gWJHelpObj = payload (low32), gWJHelpVal = full boxed bits. If the next line is
    // the last before a trap, that payload is the OOB culprit.
    JS::Value v = JS::Value::fromRawBits(gWJHelpVal);
    fprintf(stderr, "[wj-vgdbg] payload=%#x bits=%#llx isObj=%d isDouble=%d isGCThing=%d isNull=%d\n",
            gWJHelpObj, (unsigned long long)gWJHelpVal, v.isObject(), v.isDouble(),
            v.isGCThing(), v.isNull());
    fflush(stderr);
    return 0.0;
  }

  if (kind == js::wasm::WJH_CHECKCELL) {
    // DEBUG validator: caller detected a loaded value matching the SWEPT nursery
    // poison pattern (0x2B*) -- i.e. an inline access read a FREED (collected) cell's
    // memory: the reuse-staleness GC bug, caught at the exact read. gWJHelpObj holds
    // the read's PC-ish site; the value bits are reported by the caller's site arg.
    //
    // GECKO_WJ_CHECKCELL_LOG=1 makes this AGGREGATE instead of crash-on-first: it
    // histograms stale-read sites (per-site count) and returns, so a full run (or the
    // non-deterministic #19 under-rooting) yields a RANKED list of stale-read sites
    // instead of dying at the first. Dump via __wjStats()'s staleReads / on quit. The
    // default (flag unset) preserves the crash-on-first behavior for pinpoint use.
    static int aggLog = getenv("GECKO_WJ_CHECKCELL_LOG") ? 1 : 0;
    if (aggLog) {
      static const uint32_t kSlots = 4096;
      static uint64_t hits[kSlots] = {0};
      static uint64_t total = 0;
      uint32_t slot = uint32_t(int(siteF)) & (kSlots - 1);
      uint64_t c = ++hits[slot];
      if (c == 1 || (++total % 500) == 1) {
        fprintf(stderr, "[wj-checkcell-log] STALE read site=%d count=%llu total=%llu\n",
                int(siteF), (unsigned long long)c, (unsigned long long)total);
        fflush(stderr);
      }
      return 0.0;  // aggregate: do NOT crash (caller continues; may crash downstream)
    }
    fprintf(stderr, "[wj-poison] STALE READ of freed cell -- site=%d objloc=%u (GC reuse-staleness)\n",
            int(siteF), gWJHelpObj);
    fflush(stderr);
    MOZ_CRASH("WJ stale read of poisoned (freed) cell");
    return 0.0;
  }

  if (kind == js::wasm::WJH_POSTBARRIER) {
    JSObject* obj = reinterpret_cast<JSObject*>(uintptr_t(gWJHelpObj));
    // PostWriteBarrier (putWholeCellDontCheckLast) assumes a TENURED container;
    // buffering a nursery cell corrupts the store buffer. Ion inlines this guard.
    // Validate the container pointer BEFORE obj->isTenured() dereferences it: a
    // stale/garbage store-container ptr (same class as the [[prebarrier-garbage-crash]]
    // discord fix) would otherwise crash/corrupt here. IsCellPointerValid checks the
    // chunk/arena metadata without touching object contents (safe on garbage), and
    // never false-negatives a real cell -> sound.
    if (obj && js::gc::IsCellPointerValid(obj) && obj->isTenured()) {
      js::jit::PostWriteBarrier(cx->runtime(), obj);
    } else if (obj && !js::gc::IsCellPointerValid(obj)) {
      gWJPreBarBadSkips++;  // shared diagnostic: a store handed a barrier a bad container ptr
    }
    return 0.0;
  }

  if (kind == js::wasm::WJH_PREBARRIER) {
    // Incremental-GC pre-write barrier on the OLD value being overwritten. Only
    // reached when the zone's marking-barrier flag is set (fast path skips otherwise).
    // ValuePreWriteBarrier only ASSERTS isGCThing (off in release) then traces
    // v.toGCThing() -- so a NON-GC or GARBAGE-pointer old value flows straight into
    // TraceEdgeForBarrier -> uncatchable `unreachable` (the discord.com crash: a JIT
    // store's old-value read yielded a GC-tagged garbage pointer, e.g. a stale object
    // ptr after a minor GC while incremental marking is active). GUARD it: only trace a
    // real GC thing whose cell pointer is VALID (in a live GC chunk/arena) -- mirrors
    // the engine's own defensive skip at gc/Marking-inl.h IsGCThingValidAfterMovingGC.
    // Skipping an INvalid "cell" is sound (garbage isn't a live edge to preserve) and
    // never rejects a valid cell (IsCellPointerValid has no false negatives).
    JS::Value v = JS::Value::fromRawBits(gWJHelpVal);
    if (v.isGCThing()) {
      js::gc::Cell* cell = v.toGCThing();
      if (cell && js::gc::IsCellPointerValid(cell)) {
        js::gc::ValuePreWriteBarrier(v);
      } else {
        gWJPreBarBadSkips++;  // diagnostic: a store handed the barrier a bad old value
      }
    }
    return 0.0;
  }

  if (kind == js::wasm::WJH_SETSLOT) {
    JSObject* obj = reinterpret_cast<JSObject*>(uintptr_t(gWJHelpObj));
    static int dbgSetslot = getenv("GECKO_WJ_SLOTDBG2") ? 1 : 0;
    if (dbgSetslot) {
      js::NativeObject& no = obj->as<js::NativeObject>();
      fprintf(stderr, "[wb-setslot] obj=%p class=%s nfixed=%u slotSpan=%u slot=%u\n",
              (void*)obj, obj->getClass()->name, no.numFixedSlots(),
              uint32_t(no.slotSpan()), gWJHelpSlot);
    }
    obj->as<js::NativeObject>().setSlot(gWJHelpSlot,
                                        JS::Value::fromRawBits(gWJHelpVal));
    return 0.0;
  }

  // Generic VM-op helpers. gWJScratch is GC-traced (WJTraceRoots), so the staged
  // operands and result survive any GC the operation triggers.
  if (kind == js::wasm::WJH_GETPROP || kind == js::wasm::WJH_GETELEM) {
    RootedValue lref(cx, JS::Value::fromRawBits(gWJScratch[0]));
    RootedValue rref(cx, JS::Value::fromRawBits(gWJScratch[1]));
    RootedValue res(cx);
    if (!js::GetElementOperation(cx, lref, rref, &res)) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] = res.asRawBits();
    return 0.0;
  }

  // Generic binary arithmetic (MBinaryCache): a cold/unspecialized arith IC. The
  // operation is identified by the bytecode JSOp passed as `site`. Mirrors the
  // bytecode interpreter / DoBinaryArithFallback -- correct for any operand types
  // (int/double/bigint/string-concat). Operands boxed in scratch[0]/[1].
  if (kind == js::wasm::WJH_BINARYARITH) {
    JSOp op = JSOp(int(siteF));
    static int dbgArithBin = getenv("GECKO_WJ_ARITHDBG") ? 1 : 0;
    if (dbgArithBin)
      fprintf(stderr, "[wj-arith] BIN op=%d lhs=%016llx rhs=%016llx\n", int(op),
              (unsigned long long)gWJScratch[0], (unsigned long long)gWJScratch[1]);
    RootedValue lhs(cx, JS::Value::fromRawBits(gWJScratch[0]));
    RootedValue rhs(cx, JS::Value::fromRawBits(gWJScratch[1]));
    RootedValue res(cx);
    bool ok;
    switch (op) {
      case JSOp::Add: ok = js::AddValues(cx, &lhs, &rhs, &res); break;
      case JSOp::Sub: ok = js::SubValues(cx, &lhs, &rhs, &res); break;
      case JSOp::Mul: ok = js::MulValues(cx, &lhs, &rhs, &res); break;
      case JSOp::Div: ok = js::DivValues(cx, &lhs, &rhs, &res); break;
      case JSOp::Mod: ok = js::ModValues(cx, &lhs, &rhs, &res); break;
      case JSOp::Pow: ok = js::PowValues(cx, &lhs, &rhs, &res); break;
      case JSOp::BitOr: ok = js::BitOr(cx, &lhs, &rhs, &res); break;
      case JSOp::BitAnd: ok = js::BitAnd(cx, &lhs, &rhs, &res); break;
      case JSOp::BitXor: ok = js::BitXor(cx, &lhs, &rhs, &res); break;
      case JSOp::Lsh: ok = js::BitLsh(cx, &lhs, &rhs, &res); break;
      case JSOp::Rsh: ok = js::BitRsh(cx, &lhs, &rhs, &res); break;
      case JSOp::Ursh: ok = js::UrshValues(cx, &lhs, &rhs, &res); break;
      default: return 1.0;  // unsupported op (shouldn't be emitted)
    }
    if (!ok) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] = res.asRawBits();
    return 0.0;
  }

  // Generic unary arithmetic (MUnaryCache). Operand boxed in scratch[0]; JSOp in
  // `site`. Matches jit::DoUnaryArithFallback exactly (Inc/Dec assume an already-
  // numeric operand -- the bytecode applies ToNumeric before them).
  if (kind == js::wasm::WJH_UNARYARITH) {
    JSOp op = JSOp(int(siteF));
    static int dbgArithUn = getenv("GECKO_WJ_ARITHDBG") ? 1 : 0;
    if (dbgArithUn)
      fprintf(stderr, "[wj-arith] UN op=%d val=%016llx\n", int(op),
              (unsigned long long)gWJScratch[0]);
    RootedValue val(cx, JS::Value::fromRawBits(gWJScratch[0]));
    RootedValue res(cx, val);
    bool ok;
    switch (op) {
      case JSOp::BitNot: ok = js::BitNot(cx, &res, &res); break;
      case JSOp::Pos: ok = js::ToNumber(cx, &res); break;
      case JSOp::Neg: ok = js::NegOperation(cx, &res, &res); break;
      case JSOp::Inc: ok = js::IncOperation(cx, val, &res); break;
      case JSOp::Dec: ok = js::DecOperation(cx, val, &res); break;
      case JSOp::ToNumeric: ok = js::ToNumeric(cx, &res); break;
      default: return 1.0;
    }
    if (!ok) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] = res.asRawBits();
    return 0.0;
  }

  // Generic comparison (Boolean MBinaryCache): a cold/untyped ==,!=,<,<=,>,>=,===,
  // !== that our inline Compare codegen can't type. scratch[0]=lhs, scratch[1]=rhs,
  // site=JSOp. Mirrors jit::DoCompareFallback -> the VM comparison ops. Result is a
  // boxed Boolean. (Without this, such a function bails WHOLE to PBL -- deltablue's
  // constraint comparisons -> slow.)
  if (kind == js::wasm::WJH_COMPARE) {
    JSOp op = JSOp(int(siteF));
    RootedValue lhs(cx, JS::Value::fromRawBits(gWJScratch[0]));
    RootedValue rhs(cx, JS::Value::fromRawBits(gWJScratch[1]));
    bool b = false, ok = true;
    switch (op) {
      case JSOp::Eq: ok = js::LooselyEqual(cx, lhs, rhs, &b); break;
      case JSOp::Ne: ok = js::LooselyEqual(cx, lhs, rhs, &b); b = !b; break;
      case JSOp::StrictEq: ok = js::StrictlyEqual(cx, lhs, rhs, &b); break;
      case JSOp::StrictNe: ok = js::StrictlyEqual(cx, lhs, rhs, &b); b = !b; break;
      case JSOp::Lt: ok = js::LessThan(cx, &lhs, &rhs, &b); break;
      case JSOp::Le: ok = js::LessThanOrEqual(cx, &lhs, &rhs, &b); break;
      case JSOp::Gt: ok = js::GreaterThan(cx, &lhs, &rhs, &b); break;
      case JSOp::Ge: ok = js::GreaterThanOrEqual(cx, &lhs, &rhs, &b); break;
      default: return 1.0;
    }
    if (!ok) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] = JS::BooleanValue(b).asRawBits();
    return 0.0;
  }

  // Object-literal allocation (MNewObject, ObjectLiteral mode -- no compile-time
  // template). The shape is in the bytecode; NewObjectOperation(script, pc) builds
  // it. splay GeneratePayloadTree's `{array:..,string:..}` / `{left:..,right:..}`.
  if (kind == js::wasm::WJH_NEWOBJECT) {
    RootedScript script(cx, reinterpret_cast<JSScript*>(uintptr_t(gWJNewObjScript)));
    if (!script) return 1.0;
    jsbytecode* pc = script->code() + gWJNewObjPcOff;
    JSObject* o = js::NewObjectOperation(cx, script, pc);
    if (!o) return 1.0;
    static int dbgNewobj = getenv("GECKO_WJ_NEWOBJDBG") ? 1 : 0;
    if (dbgNewobj) {
      static uint64_t c = 0;
      if ((++c % 50000) == 1) {
        js::NativeObject* no = &o->as<js::NativeObject>();
        fprintf(stderr,
                "[wj-newobj-rt] op=%s nfixed=%u slotSpan=%u class=%s\n",
                js::CodeName(JSOp(*pc)), no->numFixedSlots(),
                unsigned(no->slotSpan()), o->getClass()->name);
      }
    }
    gWJScratch[js::wasm::kWJResultSlot] = JS::ObjectValue(*o).asRawBits();
    return 0.0;
  }

  // BindName (MBindNameCache): resolve the env object holding `name`'s binding for
  // a following SetName. scratch[0]=env chain, scratch[1]=name(StringValue).
  // gbemu deopt-stormed here before this case existed.
  if (kind == js::wasm::WJH_BINDNAME) {
    RootedObject env(cx, &JS::Value::fromRawBits(gWJScratch[0]).toObject());
    JSString* s = JS::Value::fromRawBits(gWJScratch[1]).toString();
    Rooted<js::PropertyName*> name(cx, s->asAtom().asPropertyName());
    JSObject* holder = js::LookupNameUnqualified(cx, name, env);
    if (!holder) return 1.0;  // error -> deopt/throw path
    gWJScratch[js::wasm::kWJResultSlot] = JS::ObjectValue(*holder).asRawBits();
    return 0.0;
  }

  // AllocateAndStoreSlot's slot growth (MAllocateAndStoreSlot). scratch[0]=object,
  // gWJNewAux=new dynamic-slot capacity. growSlotsPure mallocs the slots buffer
  // (no GC of JS objects, so obj doesn't move) and returns false on OOM without a
  // pending exception -> report it so the [1.0] flag path throws cleanly.
  if (kind == js::wasm::WJH_GROWSLOTS) {
    js::NativeObject* obj =
        &JS::Value::fromRawBits(gWJScratch[0]).toObject().as<js::NativeObject>();
    if (!js::NativeObject::growSlotsPure(cx, obj, uint32_t(gWJNewAux))) {
      js::ReportOutOfMemory(cx);
      return 1.0;
    }
    return 0.0;
  }

  // Global/lexical name lookup (MGetNameCache) + IC fill. scratch[0]=env chain
  // object, scratch[1]=name (StringValue atom, baked). site=siteF.
  if (kind == js::wasm::WJH_GETNAME) {
    uint32_t rawSite = uint32_t(siteF);
    // High bit = typeof context: an unresolved name must yield undefined, not throw
    // (mirrors the interpreter's GetNameOperation kludge). Low bits = the IC site.
    bool isTypeofName = (rawSite & 0x80000000u) != 0;
    uint32_t site = rawSite & 0x7FFFFFFFu;
    static int dbgName = getenv("GECKO_WJ_NAMEDBG") ? 1 : 0;
    if (dbgName) {
      static uint64_t calls = 0, fills = 0;
      if ((++calls % 200000) == 0)
        fprintf(stderr, "[wj-name] calls=%llu (helper hit/miss path)\n",
                (unsigned long long)calls);
      (void)fills;
    }
    RootedObject env(cx, &JS::Value::fromRawBits(gWJScratch[0]).toObject());
    JSString* s = JS::Value::fromRawBits(gWJScratch[1]).toString();
    Rooted<js::PropertyName*> name(cx, s->asAtom().asPropertyName());
    // Resolve holder+property without GC (fast); on success fill the per-site IC
    // ONLY when the holder is a realm singleton (global object / global lexical),
    // whose data-property SLOT is stable for the run (value may still mutate).
    js::PropertyResult prop;
    js::NativeObject* pobj = nullptr;
    if (js::LookupNameNoGC(cx, name, env, &pobj, &prop)) {
      JS::Value v;
      if (js::FetchNameNoGC(pobj, prop, &v)) {
        gWJScratch[js::wasm::kWJResultSlot] = v.asRawBits();
        if (site && prop.isNativeProperty() &&
            prop.propertyInfo().isDataProperty() && !v.isMagic() &&
            (pobj->is<js::GlobalObject>() ||
             pobj->is<js::GlobalLexicalEnvironmentObject>())) {
          js::TaggedSlotOffset t =
              pobj->getTaggedSlotOffset(prop.propertyInfo().slot());
          gWJNameHolder[site] = uintptr_t(static_cast<void*>(pobj));
          gWJNameShape[site] = uint32_t(uintptr_t(static_cast<void*>(pobj->shape())));
          gWJNameOff[site] = (t.offset() << js::TaggedSlotOffset::OffsetShift) |
                             (t.isFixedSlot() ? js::TaggedSlotOffset::IsFixedSlotFlag : 0);
          if (dbgName) {
            // Self-check: decode the cached offset exactly as the JIT fast path
            // does and confirm it loads `v`. A mismatch = bad offset/holder.
            uint32_t tag = uint32_t(gWJNameOff[site]);
            char* obj = reinterpret_cast<char*>(pobj);
            char* base = (tag & 1) ? obj
                         : *reinterpret_cast<char**>(
                               obj + js::NativeObject::offsetOfSlots());
            uint64_t loaded = *reinterpret_cast<uint64_t*>(base + (tag >> 1));
            if (loaded != v.asRawBits()) {
              static int nm = 0;
              if (nm++ < 20)
                fprintf(stderr,
                        "[wj-name-BAD] site=%u fixed=%d off=%u loaded=%llx v=%llx\n",
                        site, int(tag & 1), tag >> 1,
                        (unsigned long long)loaded, (unsigned long long)v.asRawBits());
            }
          }
        }
        return 0.0;
      }
    }
    // Slow path (proxies / accessors / not found): no caching. Under typeof, use
    // TypeOf mode so an unresolved name returns undefined instead of throwing
    // ReferenceError (real-site feature detection: `typeof window`, etc.).
    RootedValue res(cx);
    bool ok = isTypeofName
                  ? js::GetEnvironmentName<js::GetNameMode::TypeOf>(cx, env, name, &res)
                  : js::GetEnvironmentName<js::GetNameMode::Normal>(cx, env, name, &res);
    if (!ok) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] = res.asRawBits();
    return 0.0;
  }

  // ToPropertyKey (MToPropertyKeyCache): scratch[0]=input value -> property key
  // as a Value (string/symbol/int).
  if (kind == js::wasm::WJH_TOPROPKEY) {
    RootedValue in(cx, JS::Value::fromRawBits(gWJScratch[0]));
    JS::RootedId id(cx);
    if (!js::ToPropertyKey(cx, in, &id)) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] = js::IdToValue(id).asRawBits();
    return 0.0;
  }

  // String.charCodeAt (MCharCodeAt): scratch[0]=string, scratch[1]=index(int32).
  if (kind == js::wasm::WJH_CHARCODEAT) {
    RootedString str(cx, JS::Value::fromRawBits(gWJScratch[0]).toString());
    int32_t index = JS::Value::fromRawBits(gWJScratch[1]).toInt32();
    // siteF==1: MCharCodeAtOrNegative -> -1 for an out-of-bounds index (no throw).
    if (int(siteF) == 1 && (index < 0 || uint32_t(index) >= str->length())) {
      gWJScratch[js::wasm::kWJResultSlot] = JS::Int32Value(-1).asRawBits();
      return 0.0;
    }
    uint32_t code = 0;
    if (!js::jit::CharCodeAt(cx, str, index, &code)) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] = JS::Int32Value(int32_t(code)).asRawBits();
    return 0.0;
  }
  if (kind == js::wasm::WJH_TYPEOF) {
    // MTypeOf -> the JSType enum as an Int32 (a separate MTypeOfName maps it to a
    // string). TypeOfValue is pure (no GC/throw).
    JSType t = js::TypeOfValue(JS::Value::fromRawBits(gWJScratch[0]));
    gWJScratch[js::wasm::kWJResultSlot] = JS::Int32Value(int32_t(t)).asRawBits();
    return 0.0;
  }
  if (kind == js::wasm::WJH_ISARRAY) {
    // MIsArray (Array.isArray): non-object -> false; else IsArrayFromJit (handles
    // proxies-to-array correctly).
    RootedValue v(cx, JS::Value::fromRawBits(gWJScratch[0]));
    bool isArr = false;
    if (v.isObject()) {
      RootedObject obj(cx, &v.toObject());
      if (!js::IsArrayFromJit(cx, obj, &isArr)) return 1.0;
    }
    gWJScratch[js::wasm::kWJResultSlot] = JS::BooleanValue(isArr).asRawBits();
    return 0.0;
  }
  if (kind == js::wasm::WJH_TYPEOFNAME) {
    // MTypeOfName: map the JSType enum int -> its permanent typeof-name atom.
    JSType t = JSType(JS::Value::fromRawBits(gWJScratch[0]).toInt32());
    JSString* s = js::TypeName(t, cx->names());
    gWJScratch[js::wasm::kWJResultSlot] = JS::StringValue(s).asRawBits();
    return 0.0;
  }
  if (kind == js::wasm::WJH_REGEXPCLONE) {
    // MRegExp: clone the (baked) source RegExpObject for a regex literal.
    Rooted<js::RegExpObject*> src(
        cx, &JS::Value::fromRawBits(gWJScratch[0]).toObject().as<js::RegExpObject>());
    JSObject* clone = js::CloneRegExpObject(cx, src);
    if (!clone) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] = JS::ObjectValue(*clone).asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_NEWVARENV) {
    // MNewVarEnvironmentObject: allocate a var env from the baked VarScope.
    Rooted<js::VarScope*> scope(
        cx, reinterpret_cast<js::VarScope*>(uintptr_t(gWJVarScope)));
    JSObject* env = js::VarEnvironmentObject::createWithoutEnclosing(cx, scope);
    if (!env) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] = JS::ObjectValue(*env).asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_NEWLEXENV) {
    // MNewLexicalEnvironmentObject: allocate a block lexical env from the baked scope.
    Rooted<js::LexicalScope*> scope(
        cx, reinterpret_cast<js::LexicalScope*>(uintptr_t(gWJLexScope)));
    static int lexDbg = getenv("GECKO_WJ_LEXDBG") ? 1 : 0;  // cached: hot-path getenv tax
    JSObject* env = js::BlockLexicalEnvironmentObject::createWithoutEnclosing(cx, scope);
    if (!env) return 1.0;
    if (lexDbg) {
      js::NativeObject* nenv = &env->as<js::NativeObject>();
      fprintf(stderr,
              "[wj-lexenv] env=%p nfixed=%u slotSpan=%u numDynamic=%u "
              "scopeEnvShapeSlotSpan=%u scopeNfixed=%u\n",
              (void*)env, nenv->numFixedSlots(), nenv->slotSpan(),
              nenv->numDynamicSlots(),
              scope->environmentShape()->slotSpan(),
              scope->environmentShape()->numFixedSlots());
    }
    gWJScratch[js::wasm::kWJResultSlot] = JS::ObjectValue(*env).asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_ARRAYSLICE) {
    // MArraySlice: array.slice(begin,end). The JIT guarded IsPackedArray before
    // calling (ArraySliceDense asserts it). result=nullptr -> ArraySliceDense
    // allocates the result array.
    JS::RootedObject obj(cx, &JS::Value::fromRawBits(gWJScratch[0]).toObject());
    int32_t begin = JS::Value::fromRawBits(gWJScratch[1]).toInt32();
    int32_t end = JS::Value::fromRawBits(gWJScratch[2]).toInt32();
    JS::RootedObject none(cx, nullptr);
    JSObject* res = js::ArraySliceDense(cx, obj, begin, end, none);
    if (!res) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] = JS::ObjectValue(*res).asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_ARRAYJOIN) {
    // MArrayJoin: array.join(sep). js::jit::ArrayJoin handles any array (calls
    // array_join). scratch[0]=array(Object), scratch[1]=separator(String).
    JS::RootedObject arr(cx, &JS::Value::fromRawBits(gWJScratch[0]).toObject());
    JS::RootedString sep(cx, JS::Value::fromRawBits(gWJScratch[1]).toString());
    JSString* res = js::jit::ArrayJoin(cx, arr, sep);
    if (!res) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] = JS::StringValue(res).asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_INCACHE) {
    // MInCache (`key in obj`): general OperatorIn (ToPropertyKey + HasProperty).
    // scratch[0]=key(Value), scratch[1]=obj(Object).
    RootedValue key(cx, JS::Value::fromRawBits(gWJScratch[0]));
    JS::RootedObject obj(cx, &JS::Value::fromRawBits(gWJScratch[1]).toObject());
    bool out = false;
    if (!js::jit::OperatorIn(cx, key, obj, &out)) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] = JS::BooleanValue(out).asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_HASPROP) {
    // MMegamorphicHasProp: scratch[0]=obj(Object), scratch[1]=idVal(Value).
    // site != 0 -> hasOwn (own property only); site == 0 -> `in` (proto walk).
    JS::RootedObject obj(cx, &JS::Value::fromRawBits(gWJScratch[0]).toObject());
    RootedValue idVal(cx, JS::Value::fromRawBits(gWJScratch[1]));
    RootedId id(cx);
    if (!JS_ValueToId(cx, idVal, &id)) return 1.0;
    bool out = false;
    if (int(siteF)) {
      if (!js::HasOwnProperty(cx, obj, id, &out)) return 1.0;
    } else {
      if (!js::HasProperty(cx, obj, id, &out)) return 1.0;
    }
    gWJScratch[js::wasm::kWJResultSlot] = JS::BooleanValue(out).asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_PARSEINT) {
    // MNumberParseInt: scratch[0]=string(Value), scratch[1]=radix(Int32 Value).
    RootedString str(cx, JS::Value::fromRawBits(gWJScratch[0]).toString());
    int32_t radix = JS::Value::fromRawBits(gWJScratch[1]).toInt32();
    RootedValue result(cx);
    if (!js::NumberParseInt(cx, str, radix, &result)) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] = result.asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_OBJTOITER) {
    // MObjectToIterator: scratch[0]=obj(Object) -> for-of iterator object.
    RootedValue val(cx, JS::Value::fromRawBits(gWJScratch[0]));
    JSObject* iter = js::ValueToIterator(cx, val);
    if (!iter) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] = JS::ObjectValue(*iter).asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_ISCALLABLE) {
    // MIsCallable: scratch[0]=value -> Boolean.
    JS::Value v = JS::Value::fromRawBits(gWJScratch[0]);
    bool out = v.isObject() && v.toObject().isCallable();
    gWJScratch[js::wasm::kWJResultSlot] = JS::BooleanValue(out).asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_STRTOINDEX) {
    // MGuardStringToIndex: scratch[0]=string -> Int32 array index, or -1 if the
    // string is not a canonical array index. Matches the Ion guard semantics.
    JSString* str = JS::Value::fromRawBits(gWJScratch[0]).toString();
    int32_t result = js::jit::GetIndexFromString(str);
    gWJScratch[js::wasm::kWJResultSlot] = JS::Int32Value(result).asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_STREQATOM) {
    // MGuardSpecificAtom: scratch[0]=str, scratch[1]=atom(StringValue).
    JSString* str = JS::Value::fromRawBits(gWJScratch[0]).toString();
    JSString* atom = JS::Value::fromRawBits(gWJScratch[1]).toString();
    bool eq = (str == atom);
    if (!eq && !js::EqualStrings(cx, str, atom, &eq)) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] = JS::BooleanValue(eq).asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_HOMEPROTO) {
    // MHomeObjectSuperBase: scratch[0]=home object -> its [[Prototype]].
    JS::RootedObject obj(cx, &JS::Value::fromRawBits(gWJScratch[0]).toObject());
    JS::RootedObject proto(cx);
    if (!js::GetPrototype(cx, obj, &proto)) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] =
        proto ? JS::ObjectValue(*proto).asRawBits() : JS::NullValue().asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_SETARRLEN) {
    // MCallSetArrayLength: arr.length = rhs. obj is guaranteed an ArrayObject.
    JS::RootedObject obj(cx, &JS::Value::fromRawBits(gWJScratch[0]).toObject());
    RootedValue val(cx, JS::Value::fromRawBits(gWJScratch[1]));
    if (!js::jit::SetArrayLength(cx, obj, val, int(siteF) != 0)) return 1.0;
    return 0.0;
  }

  if (kind == js::wasm::WJH_INSHAPELIST) {
    // MGuardMultipleShapes: is obj->shape() one of the shapes in shapeList? The
    // shapes are stored as PrivateGCThing values in shapeList's dense elements.
    JSObject* obj = &JS::Value::fromRawBits(gWJScratch[0]).toObject();
    js::NativeObject* list =
        &JS::Value::fromRawBits(gWJScratch[1]).toObject().as<js::NativeObject>();
    js::Shape* objShape = obj->shape();
    bool found = false;
    uint32_t len = list->getDenseInitializedLength();
    for (uint32_t i = 0; i < len; i++) {
      JS::Value v = list->getDenseElement(i);
      if (v.toGCThing() == reinterpret_cast<js::gc::Cell*>(objShape)) {
        found = true;
        break;
      }
    }
    gWJScratch[js::wasm::kWJResultSlot] = JS::BooleanValue(found).asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_INSHAPELISTOFFSET) {
    // MGuardMultipleShapesToOffset: scan shapeList (dense [shape,offset] pairs) for
    // obj->shape(); return the paired int32 slot offset, or -1 to signal deopt. The
    // shapes are PrivateGCThing values; the offset is the next dense element (Int32).
    // Mirrors MacroAssembler::branchTestObjShapeListSetOffset (stride 2*sizeof(Value)).
    JSObject* obj = &JS::Value::fromRawBits(gWJScratch[0]).toObject();
    js::NativeObject* list =
        &JS::Value::fromRawBits(gWJScratch[1]).toObject().as<js::NativeObject>();
    js::Shape* objShape = obj->shape();
    int32_t offset = -1;
    uint32_t len = list->getDenseInitializedLength();
    for (uint32_t i = 0; i + 1 < len; i += 2) {
      JS::Value v = list->getDenseElement(i);
      if (v.toGCThing() == reinterpret_cast<js::gc::Cell*>(objShape)) {
        offset = list->getDenseElement(i + 1).toInt32();
        break;
      }
    }
    gWJScratch[js::wasm::kWJResultSlot] = JS::Int32Value(offset).asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_MATHFN) {
    // MMathFunction: x = scratch[0] (numeric), site = UnaryMathFunction id. Calls the
    // exact fdlibm/native fn Ion uses (sin/cos/tan/log/exp/acos/.../floor/ceil/round).
    double x = JS::Value::fromRawBits(gWJScratch[0]).toNumber();
    js::UnaryMathFunctionType fp =
        js::GetUnaryMathFunctionPtr(js::UnaryMathFunction(int(siteF)));
    gWJScratch[js::wasm::kWJResultSlot] = JS::DoubleValue(fp(x)).asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_NEWARRDYN) {
    // MNewArrayDynamicLength: new Array(length) with the baked template's shape.
    // ArrayConstructorOneArg throws RangeError on a negative/too-large length, which
    // matches the op's negative-length guard. scratch[0]=length, scratch[1]=template.
    int32_t len = JS::Value::fromRawBits(gWJScratch[0]).toInt32();
    JS::Rooted<js::ArrayObject*> tmpl(
        cx, &JS::Value::fromRawBits(gWJScratch[1]).toObject().as<js::ArrayObject>());
    js::ArrayObject* arr = js::ArrayConstructorOneArg(cx, tmpl, len, nullptr);
    if (!arr) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] = JS::ObjectValue(*arr).asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_OVERRECURSED) {
    // JIT recursion guard tripped: set the SAME catchable over-recursion exception
    // PBL throws via its native-stack quota. The emitted code follows this with an
    // exception-exit so it propagates (catchable) instead of overflowing V8 uncatchably.
    js::ReportOverRecursed(cx);
    return 0.0;
  }

  if (kind == js::wasm::WJH_TONUMBERINT32) {
    // MToNumberInt32: ToInt32(ToNumber(input)). Full semantics via JS::ToInt32.
    JS::RootedValue v(cx, JS::Value::fromRawBits(gWJScratch[0]));
    int32_t i;
    if (!JS::ToInt32(cx, v, &i)) return 1.0;  // threw (e.g. valueOf) -> propagate
    gWJScratch[js::wasm::kWJResultSlot] = JS::Int32Value(i).asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_TYPEDARRELEMSIZE) {
    // MTypedArrayElementSize: bytesPerElement (1/2/4/8) of the typed array.
    js::TypedArrayObject& ta =
        JS::Value::fromRawBits(gWJScratch[0]).toObject().as<js::TypedArrayObject>();
    gWJScratch[js::wasm::kWJResultSlot] =
        JS::Int32Value(int32_t(ta.bytesPerElement())).asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_NEWTYPEDARRDYN) {
    // MNewTypedArrayDynamicLength: new TypedArray(length) with the baked template.
    int32_t len = JS::Value::fromRawBits(gWJScratch[0]).toInt32();
    JS::RootedObject tmpl(cx, &JS::Value::fromRawBits(gWJScratch[1]).toObject());
    js::TypedArrayObject* ta =
        js::NewTypedArrayWithTemplateAndLength(cx, tmpl, len);
    if (!ta) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] = JS::ObjectValue(*ta).asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_NEWTYPEDARRFROMBUF) {
    // MNewTypedArrayFromArrayBuffer: new TypedArray(buffer, byteOffset, length)
    // with the baked template. byteOffset/length are boxed Values (may be
    // undefined). Mirrors Ion's js::NewTypedArrayWithTemplateAndBuffer VM call.
    JS::RootedObject buf(cx, &JS::Value::fromRawBits(gWJScratch[0]).toObject());
    JS::RootedValue byteOffset(cx, JS::Value::fromRawBits(gWJScratch[1]));
    JS::RootedValue length(cx, JS::Value::fromRawBits(gWJScratch[2]));
    JS::RootedObject tmpl(cx, &JS::Value::fromRawBits(gWJScratch[3]).toObject());
    js::TypedArrayObject* ta =
        js::NewTypedArrayWithTemplateAndBuffer(cx, tmpl, buf, byteOffset, length);
    if (!ta) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] = JS::ObjectValue(*ta).asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_NEWARGUMENTS) {
    // MCreateArgumentsObject: materialize the `arguments` object from the staged
    // actuals. scratch[0..argc-1]=actuals (boxed), [8]=callee, [9]=scopeChain,
    // [10]=argc. createForWasmJit roots the actuals + builds via the mapped/unmapped
    // create() path (uglify-js etc. are non-strict -> mapped args alias formals via
    // the CallObject). argc is bounded by kWJMaxArgs (fns with more already bail).
    uint32_t argc = uint32_t(gWJScratch[10]);  // raw (backend stores zero-extended i32)
    if (argc > js::wasm::kWJMaxArgs) argc = js::wasm::kWJMaxArgs;
    JS::RootedFunction callee(
        cx, &JS::Value::fromRawBits(gWJScratch[8]).toObject().as<JSFunction>());
    JS::RootedObject scopeChain(
        cx, &JS::Value::fromRawBits(gWJScratch[9]).toObject());
    JS::Value actuals[js::wasm::kWJMaxArgs];
    for (uint32_t i = 0; i < argc; i++) {
      actuals[i] = JS::Value::fromRawBits(gWJScratch[i]);
    }
    js::ArgumentsObject* obj =
        js::ArgumentsObject::createForWasmJit(cx, callee, scopeChain, actuals, argc);
    if (!obj) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] = JS::ObjectValue(*obj).asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_OBJCLASSTOSTRING) {
    // MObjectClassToString: Object.prototype.toString fast path -> "[object X]".
    // Returns nullptr when the object may have an @@toStringTag (needs the slow
    // path) -- NOT an error; store a 0-ptr sentinel so the emitted code deopts to
    // PBL (mirrors Ion's bailout-on-null). No GC/throw (AutoUnsafeCallWithABI).
    JSObject* obj = &JS::Value::fromRawBits(gWJScratch[0]).toObject();
    JSString* str = js::ObjectClassToString(cx, obj);
    gWJScratch[js::wasm::kWJResultSlot] =
        str ? JS::StringValue(str).asRawBits() : uint64_t(0);
    return 0.0;
  }

  if (kind == js::wasm::WJH_NEWNAMEDLAMBDA) {
    // MNewNamedLambdaObject: the named-lambda scope object for a named function
    // expression. callee = the running function (staged from the runtime callee).
    // enclosing is nullptr here (a later MIR store sets it, mirroring Ion).
    JS::RootedFunction callee(
        cx, &JS::Value::fromRawBits(gWJScratch[0]).toObject().as<JSFunction>());
    js::NamedLambdaObject* obj =
        js::NamedLambdaObject::createWithoutEnclosing(cx, callee, js::gc::Heap::Default);
    if (!obj) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] = JS::ObjectValue(*obj).asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_GETSPARSEELEM) {
    // MCallGetSparseElement: obj[index] slow path for sparse/holey arrays.
    JS::Rooted<js::NativeObject*> obj(
        cx, &JS::Value::fromRawBits(gWJScratch[0]).toObject().as<js::NativeObject>());
    int32_t index = JS::Value::fromRawBits(gWJScratch[1]).toInt32();
    JS::RootedValue result(cx);
    if (!js::GetSparseElementHelper(cx, obj, index, &result)) return 1.0;  // threw
    gWJScratch[js::wasm::kWJResultSlot] = result.get().asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_REST) {
    // MRest: build the rest-parameter array from the staged actuals. The
    // actuals live in gWJScratch[0..argc-1] (boxed, layout-compatible with
    // JS::Value's raw bits), numFormals in [8], argc in [9]. argc<=kWJMaxArgs is
    // guaranteed by the function-entry argc guard (needsArgcGuard). Passing a
    // null arrRes makes InitRestParameter allocate via NewDenseCopiedArray.
    uint32_t argc = uint32_t(gWJScratch[9]);
    if (argc > js::wasm::kWJMaxArgs) argc = js::wasm::kWJMaxArgs;
    uint32_t numFormals = uint32_t(gWJScratch[8]);
    uint32_t length = argc > numFormals ? argc - numFormals : 0;
    JS::Value* rest = reinterpret_cast<JS::Value*>(&gWJScratch[numFormals]);
    JS::Rooted<js::ArrayObject*> arrRes(cx, nullptr);
    js::ArrayObject* arr = js::jit::InitRestParameter(cx, length, rest, arrRes);
    if (!arr) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] = JS::ObjectValue(*arr).asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_OPTSPREADCALL) {
    // MOptimizeSpreadCallCache: `f(...value)` fast-path -> the array to spread, or
    // undefined if not optimizable. May GC/throw (arguments-object spread path).
    JS::RootedValue arg(cx, JS::Value::fromRawBits(gWJScratch[0]));
    JS::RootedValue rval(cx);
    if (!js::OptimizeSpreadCall(cx, arg, &rval)) return 1.0;  // threw
    gWJScratch[js::wasm::kWJResultSlot] = rval.get().asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_OPTGETITER) {
    // MOptimizeGetIteratorCache: can `value` use the fast for-of/spread iterator?
    // Pure check (no throw/alloc); the arg is GC-traced in gWJScratch[0].
    JS::Value v = JS::Value::fromRawBits(gWJScratch[0]);
    bool res = js::OptimizeGetIterator(v, cx);
    gWJScratch[js::wasm::kWJResultSlot] = JS::BooleanValue(res).asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_HASOWN) {
    // MHasOwnCache: `id in obj` / obj.hasOwnProperty(id) fast path. Operand order
    // matches DoHasOwnFallback's HasOwnProperty(cx, objValue, keyValue): value()=obj
    // in [0], idval()=key in [1] (both GC-traced boxed slots). May GC/throw
    // (ToPropertyKey / ToObject on a non-object value). Result Boolean.
    JS::RootedValue val(cx, JS::Value::fromRawBits(gWJScratch[0]));
    JS::RootedValue idval(cx, JS::Value::fromRawBits(gWJScratch[1]));
    bool res;
    if (!js::HasOwnProperty(cx, val, idval, &res)) return 1.0;  // threw
    gWJScratch[js::wasm::kWJResultSlot] = JS::BooleanValue(res).asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_NEWMAP) {
    // MNewMapObject: `new Map()`. Cold alloc (Map's internal hashtable) -> VM call,
    // nullptr proto = default Map.prototype (mirrors Ion's visitNewMapObject OOL).
    js::MapObject* m = js::MapObject::create(cx, nullptr);
    if (!m) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] = JS::ObjectValue(*m).asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_NEWSET) {
    // MNewSetObject: `new Set()`. Same pattern as WJH_NEWMAP.
    js::SetObject* s = js::SetObject::create(cx, nullptr);
    if (!s) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] = JS::ObjectValue(*s).asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_NEWITERATOR) {
    // MNewIterator: for-of / spread iterator object. site selects the type
    // (0=ArrayIterator, 1=StringIterator, 2=RegExpStringIterator), matching
    // Ion's visitNewIterator OOL VM calls. Cold alloc -> VM call.
    JSObject* it = nullptr;
    switch (int(siteF)) {
      case 0: it = js::NewArrayIterator(cx); break;
      case 1: it = js::NewStringIterator(cx); break;
      case 2: it = js::NewRegExpStringIterator(cx); break;
      default: return 1.0;
    }
    if (!it) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] = JS::ObjectValue(*it).asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_CODEPOINTAT) {
    // MCodePointAt: str.codePointAt(index) -> full Unicode code point (combines a
    // high+low surrogate pair). VM fn (Ion's OOL fallback); may GC (rope flatten).
    JS::RootedString str(cx, JS::Value::fromRawBits(gWJScratch[0]).toString());
    int32_t index = JS::Value::fromRawBits(gWJScratch[1]).toInt32();
    uint32_t out;
    if (!js::jit::CodePointAt(cx, str, index, &out)) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] = JS::Int32Value(int32_t(out)).asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_FROMCODEPOINT) {
    // MFromCodePoint: String.fromCodePoint(cp). The JIT deopt-guards cp to
    // [0, NonBMPMax] before calling, so StringFromCodePoint's MOZ_ASSERT holds
    // (invalid code points route to PBL, which throws RangeError). May GC/OOM.
    int32_t cp = JS::Value::fromRawBits(gWJScratch[0]).toInt32();
    JSLinearString* s = js::StringFromCodePoint(cx, char32_t(cp));
    if (!s) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] = JS::StringValue(s).asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_ISPACKEDARRAY) {
    // MIsPackedArray: obj is a dense, hole-free ArrayObject? Pure read (no GC/throw);
    // js::IsPackedArray handles non-array objects (returns false). Result Boolean.
    JSObject* obj = &JS::Value::fromRawBits(gWJScratch[0]).toObject();
    gWJScratch[js::wasm::kWJResultSlot] =
        JS::BooleanValue(js::IsPackedArray(obj)).asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_CANSKIPAWAIT) {
    // MCanSkipAwait: true if the awaited value isn't a thenable (await can resolve
    // synchronously). js::CanSkipAwait returns success + the flag via out-param.
    JS::RootedValue val(cx, JS::Value::fromRawBits(gWJScratch[0]));
    bool canSkip = false;
    if (!js::CanSkipAwait(cx, val, &canSkip)) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] = JS::BooleanValue(canSkip).asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_EXTRACTAWAITVALUE) {
    // MMaybeExtractAwaitValue (can-skip branch): unwrap the already-resolved value.
    JS::RootedValue val(cx, JS::Value::fromRawBits(gWJScratch[0]));
    JS::RootedValue out(cx);
    if (!js::ExtractAwaitValue(cx, val, &out)) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] = out.get().asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_ASYNCRESOLVE) {
    // MAsyncResolve: resolve an async fn's promise with `value`, return the promise.
    // generator (scratch[0]) is an AsyncFunctionGeneratorObject boxed as an Object Value.
    JS::Rooted<js::AsyncFunctionGeneratorObject*> gen(
        cx, &JS::Value::fromRawBits(gWJScratch[0])
                 .toObject()
                 .as<js::AsyncFunctionGeneratorObject>());
    JS::RootedValue value(cx, JS::Value::fromRawBits(gWJScratch[1]));
    JSObject* promise = js::AsyncFunctionResolve(cx, gen, value);
    if (!promise) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] = JS::ObjectValue(*promise).asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_GETPROPSUPER) {
    // MGetPropSuperCache: `super.prop` / `super[expr]`. Read `id` off the super base
    // `object` but invoke getters with `receiver` (this) -- js::GetProperty's
    // receiver overload does exactly that. object may be null (e.g. `extends null`)
    // -> ToObject throws TypeError, matching spec. May GC/throw.
    JS::RootedValue objVal(cx, JS::Value::fromRawBits(gWJScratch[0]));
    JS::RootedValue receiver(cx, JS::Value::fromRawBits(gWJScratch[1]));
    JS::RootedValue idval(cx, JS::Value::fromRawBits(gWJScratch[2]));
    JS::RootedObject obj(cx, JS::ToObject(cx, objVal));  // null super base -> TypeError
    if (!obj) return 1.0;
    JS::RootedId id(cx);
    if (!ToPropertyKey(cx, idval, &id)) return 1.0;
    JS::RootedValue res(cx);
    if (!js::GetProperty(cx, obj, receiver, id, &res)) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] = res.get().asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_REGEXPHASCAPS) {
    // MRegExpHasCaptureGroups: re.hasCaptureGroups for `input`.
    JS::Rooted<js::RegExpObject*> re(
        cx, &JS::Value::fromRawBits(gWJScratch[0]).toObject().as<js::RegExpObject>());
    JS::RootedString input(cx, JS::Value::fromRawBits(gWJScratch[1]).toString());
    bool res;
    if (!js::RegExpHasCaptureGroups(cx, re, input, &res)) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] = JS::BooleanValue(res).asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_REGEXPMATCHER) {
    // MRegExpMatcher: re match -> array or null. nullptr matches == IC path.
    JS::RootedObject re(cx, &JS::Value::fromRawBits(gWJScratch[0]).toObject());
    JS::RootedString input(cx, JS::Value::fromRawBits(gWJScratch[1]).toString());
    int32_t lastIndex = JS::Value::fromRawBits(gWJScratch[2]).toInt32();
    JS::RootedValue out(cx);
    if (!js::RegExpMatcherRaw(cx, re, input, lastIndex, nullptr, &out)) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] = out.get().asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_REGEXPSEARCHER) {
    // MRegExpSearcher: String.search fast path -> start index or -1. Sets
    // cx->regExpSearcherLastLimit for a following RegExpSearcherLastLimit.
    JS::RootedObject re(cx, &JS::Value::fromRawBits(gWJScratch[0]).toObject());
    JS::RootedString input(cx, JS::Value::fromRawBits(gWJScratch[1]).toString());
    int32_t lastIndex = JS::Value::fromRawBits(gWJScratch[2]).toInt32();
    int32_t res;
    if (!js::RegExpSearcherRaw(cx, re, input, lastIndex, nullptr, &res)) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] = JS::Int32Value(res).asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_REGEXPSEARCHERLASTLIMIT) {
    // MRegExpSearcherLastLimit: read the limit set by the preceding Searcher.
    gWJScratch[js::wasm::kWJResultSlot] =
        JS::Int32Value(int32_t(cx->regExpSearcherLastLimit)).asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_REGEXPEXECMATCH) {
    // MRegExpExecMatch: RegExp.prototype.exec -> array or null.
    JS::Rooted<js::RegExpObject*> re(
        cx, &JS::Value::fromRawBits(gWJScratch[0]).toObject().as<js::RegExpObject>());
    JS::RootedString input(cx, JS::Value::fromRawBits(gWJScratch[1]).toString());
    JS::RootedValue out(cx);
    if (!js::RegExpBuiltinExecMatchFromJit(cx, re, input, nullptr, &out)) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] = out.get().asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_REGEXPEXECTEST) {
    // MRegExpExecTest: RegExp.prototype.test -> Boolean.
    JS::Rooted<js::RegExpObject*> re(
        cx, &JS::Value::fromRawBits(gWJScratch[0]).toObject().as<js::RegExpObject>());
    JS::RootedString input(cx, JS::Value::fromRawBits(gWJScratch[1]).toString());
    bool res;
    if (!js::RegExpBuiltinExecTestFromJit(cx, re, input, &res)) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] = JS::BooleanValue(res).asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_GETFIRSTDOLLARINDEX) {
    // MGetFirstDollarIndex: first '$' index in the replacement string, or -1.
    JSString* s = JS::Value::fromRawBits(gWJScratch[0]).toString();
    int32_t idx;
    if (!js::GetFirstDollarIndexRaw(cx, s, &idx)) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] = JS::Int32Value(idx).asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_DELPROP) {
    // MDeleteProperty: delete obj.name. site = strict. name baked as StringValue (atom).
    JS::RootedValue val(cx, JS::Value::fromRawBits(gWJScratch[0]));
    JSString* s = JS::Value::fromRawBits(gWJScratch[1]).toString();
    JS::Rooted<js::PropertyName*> name(cx, s->asAtom().asPropertyName());
    bool res;
    bool ok = int(siteF) ? js::DelPropOperation<true>(cx, val, name, &res)
                         : js::DelPropOperation<false>(cx, val, name, &res);
    if (!ok) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] = JS::BooleanValue(res).asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_RANDOM) {
    // MRandom: Math.random() -> [0,1) from the realm's XorShift128+ RNG.
    double d = cx->realm()->getOrCreateRandomNumberGenerator().nextDouble();
    gWJScratch[js::wasm::kWJResultSlot] = JS::DoubleValue(d).asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_ISCONSTRUCTOR) {
    // MIsConstructor: is obj a constructor? Leaf (no GC).
    JSObject* o = &JS::Value::fromRawBits(gWJScratch[0]).toObject();
    gWJScratch[js::wasm::kWJResultSlot] =
        JS::BooleanValue(js::jit::ObjectIsConstructor(o)).asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_OBJECTKEYS) {
    // MObjectKeys: Object.keys(obj) -> fresh array of own enumerable string keys.
    JS::RootedObject obj(cx, &JS::Value::fromRawBits(gWJScratch[0]).toObject());
    JSObject* keys = js::jit::ObjectKeys(cx, obj);
    if (!keys) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] = JS::ObjectValue(*keys).asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_DELELEM) {
    // MDeleteElement: delete obj[index]. site = strict.
    JS::RootedValue val(cx, JS::Value::fromRawBits(gWJScratch[0]));
    JS::RootedValue idx(cx, JS::Value::fromRawBits(gWJScratch[1]));
    bool res;
    bool ok = int(siteF) ? js::DelElemOperation<true>(cx, val, idx, &res)
                         : js::DelElemOperation<false>(cx, val, idx, &res);
    if (!ok) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] = JS::BooleanValue(res).asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_STRINGREPLACE) {
    // MStringReplace: str.replace(pattern, repl). site = isFlatReplacement.
    JS::RootedString s(cx, JS::Value::fromRawBits(gWJScratch[0]).toString());
    JS::RootedString pat(cx, JS::Value::fromRawBits(gWJScratch[1]).toString());
    JS::RootedString rep(cx, JS::Value::fromRawBits(gWJScratch[2]).toString());
    JSString* out = int(siteF) ? js::StringFlatReplaceString(cx, s, pat, rep)
                               : js::jit::StringReplace(cx, s, pat, rep);
    if (!out) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] = JS::StringValue(out).asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_STRINGSPLIT) {
    // MStringSplit: str.split(sep). Fresh array, limit = INT32_MAX.
    JS::RootedString s(cx, JS::Value::fromRawBits(gWJScratch[0]).toString());
    JS::RootedString sep(cx, JS::Value::fromRawBits(gWJScratch[1]).toString());
    js::ArrayObject* arr = js::StringSplitString(cx, s, sep, INT32_MAX);
    if (!arr) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] = JS::ObjectValue(*arr).asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_STRINGTONUMBER) {
    // MGuardStringTo{Int32,Double}: speculation guard string -> number. site 0 =
    // exact int32 (jit::GetInt32FromStringPure); site 1 = double (StringToNumberPure).
    // Both pure (no GC). Parse-fail => return 1.0 (deopt to PBL), matching the guard.
    JSString* str = JS::Value::fromRawBits(gWJScratch[0]).toString();
    if (int(siteF) == 0) {
      int32_t i;
      if (!js::jit::GetInt32FromStringPure(cx, str, &i)) return 1.0;
      gWJScratch[js::wasm::kWJResultSlot] = JS::Int32Value(i).asRawBits();
    } else {
      double d;
      if (!js::StringToNumberPure(cx, str, &d)) return 1.0;
      gWJScratch[js::wasm::kWJResultSlot] = JS::DoubleValue(d).asRawBits();
    }
    return 0.0;
  }

  if (kind == js::wasm::WJH_STRINGTRIMINDEX) {
    // MStringTrim{Start,End}Index: trim() boundary scan. Leaf (AutoUnsafeCallWithABI,
    // no GC/cx). site 0 = start; site 1 = end (scratch[1]=start offset).
    JSString* str = JS::Value::fromRawBits(gWJScratch[0]).toString();
    int32_t idx;
    if (int(siteF) == 0) {
      idx = js::jit::StringTrimStartIndex(str);
    } else {
      int32_t start = JS::Value::fromRawBits(gWJScratch[1]).toInt32();
      idx = js::jit::StringTrimEndIndex(str, start);
    }
    gWJScratch[js::wasm::kWJResultSlot] = JS::Int32Value(idx).asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_INT32TOSTRINGBASE) {
    // MInt32ToStringWithBase: n.toString(base). scratch[0]=n (Int32 Value),
    // scratch[1]=base (Int32 Value), site=lowerCase. Cold/uninlinable.
    int32_t n = JS::Value::fromRawBits(gWJScratch[0]).toInt32();
    int32_t base = JS::Value::fromRawBits(gWJScratch[1]).toInt32();
    bool lower = int(siteF) != 0;
    JSLinearString* s = js::Int32ToStringWithBase<js::CanGC>(cx, n, base, lower);
    if (!s) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] = JS::StringValue(s).asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_ITEREND) {
    // MIteratorEnd: for-in loop cleanup -- unlink the NativeIterator. Non-throwing
    // (no user return() for property iterators). scratch[0] = iterator object.
    js::CloseIterator(&JS::Value::fromRawBits(gWJScratch[0]).toObject());
    gWJScratch[js::wasm::kWJResultSlot] = JS::UndefinedValue().asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_NEWBOUNDFN) {
    // MNewBoundFunction: allocate a bound function from the template. scratch[0]=template.
    JS::Rooted<js::BoundFunctionObject*> tmpl(
        cx, &JS::Value::fromRawBits(gWJScratch[0]).toObject().as<js::BoundFunctionObject>());
    JSObject* bf = js::BoundFunctionObject::createWithTemplate(cx, tmpl);
    if (!bf) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] = JS::ObjectValue(*bf).asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_BINDFUNCTION) {
    // MBindFunction: Function.prototype.bind. scratch[0]=target, scratch[1..argc]=bound
    // args (args[0]=boundThis), site=argc. Copy the args into a rooted vector (functionBindImpl
    // allocs -> can GC; gWJScratch is not traced) and call with maybeBound=null (C++ allocs).
    uint32_t argc = (uint32_t)siteF;
    JS::RootedObject target(cx, &JS::Value::fromRawBits(gWJScratch[0]).toObject());
    JS::RootedValueVector av(cx);
    if (!av.reserve(argc)) return 1.0;
    for (uint32_t i = 0; i < argc; i++) {
      av.infallibleAppend(JS::Value::fromRawBits(gWJScratch[1 + i]));
    }
    JS::Rooted<js::BoundFunctionObject*> maybeBound(cx, nullptr);
    js::BoundFunctionObject* bf =
        js::BoundFunctionObject::functionBindImpl(cx, target, av.begin(), argc, maybeBound);
    if (!bf) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] = JS::ObjectValue(*bf).asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_LOADITERELEM) {
    // MLoadIteratorElement: the index-th iterated key string from the NativeIterator's
    // property array (Object.keys(o)[i]). scratch[0]=iterator, scratch[1]=index(Int32).
    // Pure read; index in-bounds by the ObjectKeysReplacer invariant. .asString() strips
    // the DeletedBit. No GC/throw.
    js::NativeIterator* ni = JS::Value::fromRawBits(gWJScratch[0])
                                 .toObject()
                                 .as<js::PropertyIteratorObject>()
                                 .getNativeIterator();
    int32_t index = JS::Value::fromRawBits(gWJScratch[1]).toInt32();
    JSLinearString* str = ni->propertiesBegin()[index].asString();
    gWJScratch[js::wasm::kWJResultSlot] = JS::StringValue(str).asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_GETDOMPROP) {
    // MGetDOMProperty: invoke the DOM getter (replicates js::jit::CallDOMGetter). scratch[0]=
    // the (guarded) DOM object, scratch[1]=baked JSJitGetterOp (fn-table index). Load
    // DOM_OBJECT_SLOT (reserved slot 0, the native this as a PrivateValue) and call the getter.
    // May GC/throw.
    JS::RootedObject obj(cx, &JS::Value::fromRawBits(gWJScratch[0]).toObject());
    JSJitGetterOp getter =
        reinterpret_cast<JSJitGetterOp>(uintptr_t(gWJScratch[1]));
    JS::Value slotVal = obj->as<js::NativeObject>().getReservedSlot(0);
    JS::RootedValue result(cx);
    if (!getter(cx, obj, slotVal.toPrivate(), JSJitGetterCallArgs(&result))) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] = result.get().asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_CHECKPRIVFIELD) {
    // MCheckPrivateFieldCache: `#x in obj` / `obj.#x` brand check. scratch[0]=value,
    // [1]=idval(private-name Symbol), [2]=bytecode pc (the ThrowCondition source).
    jsbytecode* pc = reinterpret_cast<jsbytecode*>(uintptr_t(uint32_t(gWJScratch[2])));
    JS::RootedValue val(cx, JS::Value::fromRawBits(gWJScratch[0]));
    JS::RootedValue idval(cx, JS::Value::fromRawBits(gWJScratch[1]));
    if (getenv("GECKO_WJ_CPFCDBG")) {
      fprintf(stderr, "[cpfcdbg] val.isObj=%d idval.isSym=%d idval.isPrivName=%d op=%d\n",
              val.isObject(), idval.isSymbol(),
              idval.isSymbol() && idval.toSymbol()->isPrivateName(), int(*pc));
    }
    bool result = false;
    if (!js::CheckPrivateFieldOperation(cx, pc, val, idval, &result)) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] = JS::BooleanValue(result).asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_APPLYARRAY) {
    // MApplyArray: fn.apply(thisArg, argsArray) -- spread the array's dense elements.
    // scratch[0]=callee, [1]=thisArg, [2]=argsArray(Object). Copy the elements into a
    // rooted vector (before JS::Call can GC), then JS::Call.
    JS::RootedValue callee(cx, JS::Value::fromRawBits(gWJScratch[0]));
    JS::RootedValue thisv(cx, JS::Value::fromRawBits(gWJScratch[1]));
    JS::RootedObject arrObj(cx, &JS::Value::fromRawBits(gWJScratch[2]).toObject());
    uint32_t len = arrObj->as<js::NativeObject>().getDenseInitializedLength();
    JS::RootedValueVector av(cx);
    if (!av.reserve(len)) return 1.0;  // may GC; arrObj is rooted (re-deref below)
    {
      js::NativeObject& arr = arrObj->as<js::NativeObject>();
      for (uint32_t i = 0; i < len; i++) av.infallibleAppend(arr.getDenseElement(i));
    }
    JS::RootedValue rv(cx);
    if (!JS::Call(cx, thisv, callee, JS::HandleValueArray(av), &rv)) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] = rv.get().asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_GFADBG) {
    uint64_t v = gWJScratch[0];  // the param[i] value the JIT read
    JS::Value val = JS::Value::fromRawBits(v);
    // also peek the caller-staged slots: gWJScratch[10..] are free scratch we don't clobber;
    // print gWJCallArgc + the boxed callee to see argc and which entry path. (gWJScratch[0]
    // was just overwritten with the param value, so read argc/callee globals instead.)
    JS::Value cal = JS::Value::fromRawBits(gWJCallCallee);
    fprintf(stderr, "[gfadbg] i=%d param.int=%d (bits=%llx isInt=%d isObj=%d) gWJCallArgc=%u calleeIsFn=%d\n",
            int(siteF), val.isInt32() ? val.toInt32() : -99999, (unsigned long long)v,
            val.isInt32(), val.isObject(), gWJCallArgc,
            cal.isObject() && cal.toObject().is<JSFunction>());
    gWJScratch[js::wasm::kWJResultSlot] = v;
    return 0.0;
  }

  if (kind == js::wasm::WJH_LOADSLOTBYITER) {
    // MLoadSlotByIteratorIndex: load object's slot/element at the iterator's CURRENT
    // PropertyIndex. The cursor was pre-incremented by nextIteratedValueAndAdvance, so
    // the current property is at cursor-1. indicesBegin() = the PropertyIndex array.
    js::NativeObject* nobj =
        &JS::Value::fromRawBits(gWJScratch[0]).toObject().as<js::NativeObject>();
    js::NativeIterator* ni = JS::Value::fromRawBits(gWJScratch[1])
                                 .toObject()
                                 .as<js::PropertyIteratorObject>()
                                 .getNativeIterator();
    uint32_t cursor = *reinterpret_cast<const uint32_t*>(
        reinterpret_cast<const char*>(ni) +
        js::NativeIterator::offsetOfPropertyCursor());
    js::PropertyIndex pi = ni->indicesBegin()[cursor - 1];
    JS::Value result;
    switch (pi.kind()) {
      case js::PropertyIndex::Kind::FixedSlot:
        result = nobj->getFixedSlot(pi.index());
        break;
      case js::PropertyIndex::Kind::DynamicSlot:
        result = nobj->getSlot(nobj->numFixedSlots() + pi.index());
        break;
      case js::PropertyIndex::Kind::Element:
        result = nobj->getDenseElement(pi.index());
        break;
      default:
        result = JS::UndefinedValue();
        break;
    }
    gWJScratch[js::wasm::kWJResultSlot] = result.asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_ITERHASINDICES) {
    // MIteratorHasIndices: indices available + object shape matches iterator's stored shape.
    JSObject* obj = &JS::Value::fromRawBits(gWJScratch[0]).toObject();
    js::NativeIterator* ni = JS::Value::fromRawBits(gWJScratch[1])
                                 .toObject()
                                 .as<js::PropertyIteratorObject>()
                                 .getNativeIterator();
    uint8_t flags = *(reinterpret_cast<const uint8_t*>(ni) +
                      js::NativeIterator::offsetOfFlags());
    // GECKO_WJ_ITERIDX=1 re-enables the indices fast path. DEFAULT OFF
    // (2026-07-02): with it on, one for-in value read per walk returned the
    // WRONG property's value (acorn-bench node count off by one; walkcount
    // micro: a plain object read as its sibling array -> Array.isArray true).
    // The megamorphic fallback (GetElementOperation) is sound. Root cause of
    // the indices/cursor divergence TBD.
    static int iterIdx = getenv("GECKO_WJ_ITERIDX") ? 1 : 0;
    bool r = iterIdx && (flags & js::NativeIterator::Flags::IndicesAvailable) &&
             (obj->shape() == ni->objShape());
    gWJScratch[js::wasm::kWJResultSlot] = JS::BooleanValue(r).asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_ITERLENGTH) {
    // MIteratorLength: NativeIterator ownPropertyCount (the count of own enumerable keys).
    js::NativeIterator* ni = JS::Value::fromRawBits(gWJScratch[0])
                                 .toObject()
                                 .as<js::PropertyIteratorObject>()
                                 .getNativeIterator();
    uint32_t len = *reinterpret_cast<const uint32_t*>(
        reinterpret_cast<const char*>(ni) +
        js::NativeIterator::offsetOfOwnPropertyCount());
    gWJScratch[js::wasm::kWJResultSlot] = JS::Int32Value(int32_t(len)).asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_GETITER) {
    // MGetIteratorCache: for-in iterator setup (JSOp::Iter). scratch[0] = value.
    JS::RootedValue v(cx, JS::Value::fromRawBits(gWJScratch[0]));
    JSObject* iter = js::ValueToIterator(cx, v);
    if (!iter) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] = JS::ObjectValue(*iter).asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_ITERMORE) {
    // MIteratorMore: next for-in key (String) or MagicValue(JS_NO_ITER_VALUE) when
    // exhausted. NativeIterator cursor walk; no GC/throw.
    js::NativeIterator* ni = JS::Value::fromRawBits(gWJScratch[0])
                                 .toObject()
                                 .as<js::PropertyIteratorObject>()
                                 .getNativeIterator();
    gWJScratch[js::wasm::kWJResultSlot] =
        ni->nextIteratedValueAndAdvance().asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_OBJKEYSITER) {
    // MObjectKeysFromIterator: Object.keys fast path from an already-built iterator.
    JS::RootedObject it(cx, &JS::Value::fromRawBits(gWJScratch[0]).toObject());
    JSObject* keys = js::jit::ObjectKeysFromIterator(cx, it);
    if (!keys) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] = JS::ObjectValue(*keys).asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_CHARCASE) {
    // MCharCodeConvertCase: single-char unicode case fold (1:1). site=0 lower/1 upper.
    int32_t code = JS::Value::fromRawBits(gWJScratch[0]).toInt32();
    char16_t ch = char16_t(uint16_t(code));
    char16_t out = int(siteF) ? js::unicode::ToUpperCase(ch) : js::unicode::ToLowerCase(ch);
    JSString* s = js::StringFromCharCode(cx, int32_t(out));
    if (!s) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] = JS::StringValue(s).asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_CLOSEITER) {
    // MCloseIterCache: for-of early-exit cleanup -- call iter's return() if present.
    // site = CompletionKind (0 Normal / 1 Throw / 2 Return).
    JS::RootedObject iter(cx, &JS::Value::fromRawBits(gWJScratch[0]).toObject());
    if (!js::CloseIterOperation(cx, iter, js::CompletionKind(int(siteF))))
      return 1.0;
    return 0.0;
  }

  // String.fromCharCode (MFromCharCode): scratch[0]=code(int32) -> 1-char string.
  if (kind == js::wasm::WJH_FROMCHARCODE) {
    int32_t code = JS::Value::fromRawBits(gWJScratch[0]).toInt32();
    JSString* s = js::StringFromCharCode(cx, code);
    if (!s) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] = JS::StringValue(s).asRawBits();
    return 0.0;
  }

  // ToInt32 (MTruncateToInt32 on a boxed Value): scratch[0]=value -> int32. Full
  // JS ToInt32 semantics (number trunc, undefined/null->0, bool->0/1, string parse),
  // correct for ANY value with NO deopt -- so the emitted fast path can route only
  // its rare non-number case here instead of a (resume-unsound) bail.
  if (kind == js::wasm::WJH_TOINT32) {
    RootedValue in(cx, JS::Value::fromRawBits(gWJScratch[0]));
    int32_t out = 0;
    if (!JS::ToInt32(cx, in, &out)) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] = JS::Int32Value(out).asRawBits();
    return 0.0;
  }

  // ToString (MToString): scratch[0]=input value -> string.
  if (kind == js::wasm::WJH_TOSTRING) {
    RootedValue in(cx, JS::Value::fromRawBits(gWJScratch[0]));
    JSString* s = js::ToString<js::CanGC>(cx, in);
    if (!s) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] = JS::StringValue(s).asRawBits();
    return 0.0;
  }
  if (kind == js::wasm::WJH_LINEARIZE) {
    // MLinearizeString / MLinearizeForCharAccess: flatten a rope to a linear string
    // (valid for any char index). scratch[0] = the string value.
    RootedString s(cx, JS::Value::fromRawBits(gWJScratch[0]).toString());
    JSLinearString* lin = s->ensureLinear(cx);
    if (!lin) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] = JS::StringValue(lin).asRawBits();
    return 0.0;
  }
  if (kind == js::wasm::WJH_STROP) {
    // Generic string method, dispatched on siteF (the sub-op). scratch[0]=string,
    // scratch[1]=search/begin, scratch[2]=length.
    int sub = int(siteF);
    RootedString s(cx, JS::Value::fromRawBits(gWJScratch[0]).toString());
    if (sub <= 4) {
      RootedString search(cx, JS::Value::fromRawBits(gWJScratch[1]).toString());
      if (sub == 1 || sub == 2) {
        int32_t r = 0;
        bool ok = (sub == 1) ? js::StringIndexOf(cx, s, search, &r)
                             : js::StringLastIndexOf(cx, s, search, &r);
        if (!ok) return 1.0;
        gWJScratch[js::wasm::kWJResultSlot] = JS::Int32Value(r).asRawBits();
      } else {
        bool r = false;
        bool ok = (sub == 0)   ? js::StringIncludes(cx, s, search, &r)
                  : (sub == 3) ? js::StringStartsWith(cx, s, search, &r)
                               : js::StringEndsWith(cx, s, search, &r);
        if (!ok) return 1.0;
        gWJScratch[js::wasm::kWJResultSlot] = JS::BooleanValue(r).asRawBits();
      }
    } else if (sub == 5) {
      int32_t begin = JS::Value::fromRawBits(gWJScratch[1]).toInt32();
      int32_t len = JS::Value::fromRawBits(gWJScratch[2]).toInt32();
      JSString* r = js::SubstringKernel(cx, s, begin, len);
      if (!r) return 1.0;
      gWJScratch[js::wasm::kWJResultSlot] = JS::StringValue(r).asRawBits();
    } else {
      JSLinearString* r =
          (sub == 6) ? js::StringToLowerCase(cx, s) : js::StringToUpperCase(cx, s);
      if (!r) return 1.0;
      gWJScratch[js::wasm::kWJResultSlot] = JS::StringValue(r).asRawBits();
    }
    return 0.0;
  }

  if (kind == js::wasm::WJH_PROPIC) {
    // Inline property-load IC miss: the JIT'd fast path didn't match a cached
    // shape. Get the value AND, for an OWN data property of a native receiver,
    // fill the per-site IC (shape -> TaggedSlotOffset) so the next access loads
    // the slot inline. Property name is baked per-site in gWJPropKey.
    uint32_t site = uint32_t(siteF);
    // Miss-category histogram (GECKO_WJ_PROPICSTATS): why does this PROPIC helper
    // hop happen? Distinguishes polymorphic-overflow (way eviction -> a shared
    // global megamorphic cache would help) from fresh/cold fills and uncacheable
    // accessor/proxy fallbacks (neither more-ways nor a global cache helps). Drives
    // the goal-1 lever decision. Counters: 0=own-fresh 1=own-EVICT 2=own-rehit
    // 3=proto-fill 4=missing-fill 5=uncacheable-fallback. Printed every 500k hops.
    static int propStats = -1;
    if (propStats < 0) propStats = getenv("GECKO_WJ_PROPICSTATS") ? 1 : 0;
    static uint64_t pstat[6] = {0};
    static uint64_t pstatTot = 0;
    auto pbump = [&](int c) {
      if (!propStats) return;
      pstat[c]++;
      if ((++pstatTot % 500000) == 0)
        fprintf(stderr,
                "[wj-propicstats] %llu hops: own-fresh=%llu own-EVICT=%llu "
                "own-rehit=%llu proto=%llu missing=%llu uncacheable=%llu\n",
                (unsigned long long)pstatTot, (unsigned long long)pstat[0],
                (unsigned long long)pstat[1], (unsigned long long)pstat[2],
                (unsigned long long)pstat[3], (unsigned long long)pstat[4],
                (unsigned long long)pstat[5]);
    };
    RootedValue objv(cx, JS::Value::fromRawBits(gWJScratch[0]));
    JS::RootedId id(cx, JS::PropertyKey::fromRawBits(uintptr_t(gWJPropKey[site])));
    if (objv.isObject() && objv.toObject().is<js::NativeObject>()) {
      js::NativeObject* nobj = &objv.toObject().as<js::NativeObject>();
      mozilla::Maybe<js::PropertyInfo> prop = nobj->lookupPure(id);
      if (prop.isSome() && prop->isDataProperty()) {
        uint32_t base = site * js::wasm::kWJPropWays;
        uint32_t shapeBits =
            uint32_t(uintptr_t(static_cast<void*>(nobj->shape())));
        js::TaggedSlotOffset t = nobj->getTaggedSlotOffset(prop->slot());
        uint32_t offBits =
            (t.offset() << js::TaggedSlotOffset::OffsetShift) |
            (t.isFixedSlot() ? js::TaggedSlotOffset::IsFixedSlotFlag : 0);
        uint32_t w = 0;
        for (; w < js::wasm::kWJPropWays; w++) {
          if (gWJPropShape[base + w] == 0 || gWJPropShape[base + w] == shapeBits)
            break;
        }
        pbump(w == js::wasm::kWJPropWays ? 1 : (gWJPropShape[base + w] == shapeBits ? 2 : 0));
        if (w == js::wasm::kWJPropWays) w = 0;  // evict way 0
        static int noFill = -1;
        if (noFill < 0) noFill = getenv("GECKO_WJ_PROPNOFILL") ? 1 : 0;
        if (!noFill) {
          gWJPropShape[base + w] = shapeBits;
          gWJPropOff[base + w] = offBits;
          gWJPropHolder[base + w] = 0;  // OWN property -> load from receiver
        }
        gWJScratch[js::wasm::kWJResultSlot] = nobj->getSlot(prop->slot()).asRawBits();
        return 0.0;
      }
    }
    // A receiver whose CLASS has a RESOLVE HOOK may have an UNRESOLVED OWN property
    // that lookupPure (pure = no resolve) missed above -- e.g. a JSFunction's lazy
    // `name`/`length` (fun_resolve). That own property would SHADOW any proto prop,
    // so the proto/missing caches below are UNSOUND here (they'd return the proto's
    // value, e.g. Function.prototype.name="" for `fn.name` on an un-accessed fn ->
    // chai getFuncName miscompile). Skip them -> fall to the resolving
    // GetElementOperation, which runs the resolve hook and reads the real own prop.
    bool recvHasResolveHook =
        objv.isObject() && objv.toObject().getClass()->getResolve() != nullptr;
    // PROTO-DATA cache: the property isn't OWN, but if it's a plain DATA property on
    // a native object in the proto chain (richards' `.run` method lookup, 798K/run),
    // cache (receiverShape -> holder + holder's tagged slot offset). A receiver-shape
    // match alone is sound: the shape encodes proto IDENTITY and absence of an own
    // shadow, the holder's existing-prop slot offset is stable, and the value is
    // loaded fresh. Holder ptr is traced. GECKO_WJ_NOPROTOIC disables.
    static int noProtoIC = getenv("GECKO_WJ_NOPROTOIC") ? 1 : 0;
    if (!noProtoIC && !recvHasResolveHook && objv.isObject() &&
        objv.toObject().is<js::NativeObject>()) {
      js::NativeObject* recv = &objv.toObject().as<js::NativeObject>();
      for (JSObject* p = recv->staticPrototype(); p && p->is<js::NativeObject>();
           p = p->staticPrototype()) {
        js::NativeObject* holder = &p->as<js::NativeObject>();
        mozilla::Maybe<js::PropertyInfo> pp = holder->lookupPure(id);
        if (pp.isNothing()) continue;
        if (pp->isDataProperty()) {
          uint32_t base = site * js::wasm::kWJPropWays;
          uint32_t recvShape =
              uint32_t(uintptr_t(static_cast<void*>(recv->shape())));
          js::TaggedSlotOffset t = holder->getTaggedSlotOffset(pp->slot());
          uint32_t offBits = (t.offset() << js::TaggedSlotOffset::OffsetShift) |
                             (t.isFixedSlot() ? js::TaggedSlotOffset::IsFixedSlotFlag : 0);
          uint32_t w = 0;
          for (; w < js::wasm::kWJPropWays; w++) {
            if (gWJPropShape[base + w] == 0 || gWJPropShape[base + w] == recvShape)
              break;
          }
          if (w == js::wasm::kWJPropWays) w = 0;
          gWJPropShape[base + w] = recvShape;
          gWJPropOff[base + w] = offBits;
          gWJPropHolder[base + w] =
              uint32_t(uintptr_t(static_cast<void*>(holder)));
          pbump(3);
          gWJScratch[js::wasm::kWJResultSlot] = holder->getSlot(pp->slot()).asRawBits();
          return 0.0;
        }
        break;  // accessor / non-data on the chain -> generic
      }
    }
    // MISSING-PROPERTY cache (~92% of ubo PROPIC misses): if the property is absent
    // on the receiver AND the entire NATIVE proto chain, cache (receiverShape ->
    // MISSING sentinel) so the inline fast path returns `undefined` with no C++ hop.
    // Same soundness tier as the proto-read cache: the receiver shape encodes proto
    // IDENTITY + own absence; a receiver-shape match => same chain => still absent
    // (unless a proto dynamically GAINS the prop without a receiver-shape change --
    // the proto-read cache accepts the same caveat; uBlock protos are static).
    // GECKO_WJ_NOMISSINGIC disables.
    static int noMissingIC = -1;
    if (noMissingIC < 0) noMissingIC = getenv("GECKO_WJ_NOMISSINGIC") ? 1 : 0;
    if (!noMissingIC && objv.isObject() &&
        objv.toObject().is<js::NativeObject>()) {
      bool allNativeMissing = true;
      for (JSObject* p = &objv.toObject(); p; p = p->staticPrototype()) {
        if (!p->is<js::NativeObject>()) { allNativeMissing = false; break; }
        // A class RESOLVE HOOK can lazily materialize a property that lookupPure
        // (pure = no resolve) does NOT see -- e.g. a JSFunction's lazy `name`/
        // `length` (fun_resolve). Treating such as "missing" cached undefined for
        // `fn.name` on an un-accessed function (chai getFuncName miscompile). If any
        // object in the chain has a resolve hook, don't conclude missing -> fall to
        // the resolving GetElementOperation below.
        if (p->getClass()->getResolve()) { allNativeMissing = false; break; }
        if (p->as<js::NativeObject>().lookupPure(id).isSome()) {
          allNativeMissing = false;
          break;
        }
      }
      if (allNativeMissing) {
        js::NativeObject* recv = &objv.toObject().as<js::NativeObject>();
        uint32_t base = site * js::wasm::kWJPropWays;
        uint32_t recvShape =
            uint32_t(uintptr_t(static_cast<void*>(recv->shape())));
        uint32_t w = 0;
        for (; w < js::wasm::kWJPropWays; w++) {
          if (gWJPropShape[base + w] == 0 || gWJPropShape[base + w] == recvShape)
            break;
        }
        if (w == js::wasm::kWJPropWays) w = 0;
        gWJPropShape[base + w] = recvShape;
        gWJPropOff[base + w] = js::wasm::kWJPropMissingSentinel;
        gWJPropHolder[base + w] = 0;
        pbump(4);
        gWJScratch[js::wasm::kWJResultSlot] = JS::UndefinedValue().asRawBits();
        return 0.0;
      }
    }
    // Fallback: accessor/non-native/proxy -> generic get (no caching).
    pbump(5);
    RootedValue keyv(cx, js::IdToValue(id));
    RootedValue res(cx);
    if (!js::GetElementOperation(cx, objv, keyv, &res)) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] = res.asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_SETPROPIC) {
    // Set-prop IC miss: store value into obj.name. For a WRITABLE own data
    // property of a native receiver, fill the per-site IC (shape ->
    // TaggedSlotOffset) so the next store writes the slot inline, and store via
    // setSlot (which runs the proper pre/post barriers). Otherwise fall back to
    // the generic set (no caching).
    uint32_t site = uint32_t(siteF);
    RootedValue objv(cx, JS::Value::fromRawBits(gWJScratch[0]));
    RootedValue idv(cx, JS::Value::fromRawBits(gWJScratch[1]));
    RootedValue val(cx, JS::Value::fromRawBits(gWJScratch[2]));
    JS::RootedId id(cx);
    if (!js::ToPropertyKey(cx, idv, &id)) return 1.0;
    static int spicDbg = -1;
    if (spicDbg < 0) spicDbg = getenv("GECKO_WJ_SPICDBG") ? 1 : 0;
    if (objv.isObject() && objv.toObject().is<js::NativeObject>()) {
      js::NativeObject* nobj = &objv.toObject().as<js::NativeObject>();
      mozilla::Maybe<js::PropertyInfo> prop = nobj->lookupPure(id);
      if (spicDbg) {
        static uint64_t cacheable = 0, uncacheable = 0, repeatShape = 0;
        static uint32_t lastSite = 0xffffffff, lastShape = 0;
        if (prop.isSome() && prop->isDataProperty() && prop->writable()) {
          cacheable++;
          uint32_t sb = uint32_t(uintptr_t(static_cast<void*>(nobj->shape())));
          if (site == lastSite && sb == lastShape) repeatShape++;
          lastSite = site; lastShape = sb;
        } else {
          uncacheable++;
        }
        static uint64_t intId = 0, strAdd = 0, strAccessor = 0, strProto = 0, other = 0;
        static uint64_t intArrayInbounds = 0, intArrayOOB = 0, intNonArray = 0;
        if (!(prop.isSome() && prop->isDataProperty() && prop->writable())) {
          if (id.isInt()) {
            intId++;
            if (nobj->is<js::ArrayObject>()) {
              uint32_t idx = uint32_t(id.toInt());
              if (idx < nobj->getDenseInitializedLength()) intArrayInbounds++;
              else intArrayOOB++;
            } else {
              intNonArray++;
              if (nobj->is<js::TypedArrayObject>()) {
                static uint64_t taValInt = 0, taValOther = 0, taClaspMatch = 0,
                                taClaspMiss = 0, taInbounds = 0;
                int tt = int(nobj->as<js::TypedArrayObject>().type());
                if (val.isInt32()) taValInt++; else taValOther++;
                const JSClass* expect = &js::TypedArrayObject::fixedLengthClasses[tt];
                if (nobj->getClass() == expect) taClaspMatch++; else taClaspMiss++;
                if (uint32_t(id.toInt()) <
                    nobj->as<js::TypedArrayObject>().length().valueOr(0))
                  taInbounds++;
                // Verify the inline clasp chain (obj->shape->base->clasp via raw
                // offsets) against getClass() -- the suspected inline-IC bug.
                {
                  char* o = reinterpret_cast<char*>(nobj);
                  uintptr_t shape = *reinterpret_cast<uintptr_t*>(
                      o + offsetof(JS::shadow::Object, shape));
                  uintptr_t bptr = *reinterpret_cast<uintptr_t*>(
                      shape + js::Shape::offsetOfBaseShape());
                  uintptr_t clasp = *reinterpret_cast<uintptr_t*>(
                      bptr + js::BaseShape::offsetOfClasp());
                  static int chk = 0;
                  if (chk < 3) {
                    chk++;
                    fprintf(stderr,
                            "[wj-spic-chain] chainClasp=%p getClass=%p expectFLC=%p shapeOff=%zu baseOff=%zu claspOff=%zu\n",
                            (void*)clasp, (void*)nobj->getClass(), (void*)expect,
                            (size_t)offsetof(JS::shadow::Object, shape),
                            (size_t)js::Shape::offsetOfBaseShape(),
                            (size_t)js::BaseShape::offsetOfClasp());
                  }
                }
                static uint64_t taN = 0;
                if ((++taN % 400000) == 0)
                  fprintf(stderr,
                          "[wj-spic-ta] elemHits=%llu | valInt=%llu valOther=%llu claspMatch=%llu claspMiss=%llu inbounds=%llu (type=%d)\n",
                          (unsigned long long)gWJElemHits,
                          (unsigned long long)taValInt, (unsigned long long)taValOther,
                          (unsigned long long)taClaspMatch, (unsigned long long)taClaspMiss,
                          (unsigned long long)taInbounds, tt);
              }
            }
          }
          else if (prop.isNothing()) strAdd++;        // named: property not present -> add/proto
          else if (!prop->isDataProperty()) strAccessor++;  // accessor
          else other++;
        }
        if (spicDbg && ((cacheable + uncacheable) % 400000) == 0)
          fprintf(stderr,
                  "[wj-spic-elem] intArrayInbounds=%llu intArrayOOB=%llu intNonArray=%llu\n",
                  (unsigned long long)intArrayInbounds,
                  (unsigned long long)intArrayOOB, (unsigned long long)intNonArray);
        if (((cacheable + uncacheable) % 200000) == 0)
          fprintf(stderr,
                  "[wj-spic] cacheable=%llu uncacheable=%llu (intIdx=%llu namedAdd/proto=%llu accessor=%llu other=%llu) repeatShape=%llu\n",
                  (unsigned long long)cacheable, (unsigned long long)uncacheable,
                  (unsigned long long)intId, (unsigned long long)strAdd,
                  (unsigned long long)strAccessor, (unsigned long long)other,
                  (unsigned long long)repeatShape);
      }
      if (prop.isSome() && prop->isDataProperty() && prop->writable()) {
        uint32_t base = site * js::wasm::kWJPropWays;
        uint32_t shapeBits =
            uint32_t(uintptr_t(static_cast<void*>(nobj->shape())));
        js::TaggedSlotOffset t = nobj->getTaggedSlotOffset(prop->slot());
        uint32_t offBits =
            (t.offset() << js::TaggedSlotOffset::OffsetShift) |
            (t.isFixedSlot() ? js::TaggedSlotOffset::IsFixedSlotFlag : 0);
        // Only ATOM-string keys are cacheable: the inline hit path guards the
        // boxed idval against the cached atom by pointer identity, and one site
        // can see MANY keys for ONE shape (`obj[k] = v` copy loops -- babylon
        // __clone). A way is chosen by (shape, key) match, not shape alone.
        uint32_t keyBits =
            (idv.isString() && idv.toString()->isAtom())
                ? uint32_t(uintptr_t(static_cast<void*>(idv.toString())))
                : 0;
        static int noFillS = -1;
        if (noFillS < 0) noFillS = getenv("GECKO_WJ_PROPNOFILL") ? 1 : 0;
        if (!noFillS && keyBits) {
          uint32_t w = 0;
          for (; w < js::wasm::kWJPropWays; w++) {
            if (gWJPropShape[base + w] == 0 ||
                (gWJPropShape[base + w] == shapeBits &&
                 gWJPropWayKey[base + w] == keyBits))
              break;
          }
          if (w == js::wasm::kWJPropWays) w = 0;
          gWJPropShape[base + w] = shapeBits;
          gWJPropOff[base + w] = offBits;
          gWJPropWayKey[base + w] = keyBits;
        }
        nobj->setSlot(prop->slot(), val);  // pre+post write barriers included
        return 0.0;
      }
    }
    // Fallback: setter / non-writable / proto / add / non-native -> generic set.
    if (!objv.isObject()) return 1.0;
    // ADD-IC: if this is a NAMED property ADD to a native object, cache the
    // (oldShape -> newShape, fixedSlotOffset) transition so the next same-shape add
    // at this site stores inline (no helper). Intern the PRE-add shape NOW (into the
    // traced gWJShapePool, GC-current) before SetObjectElement runs the transition.
    static int addIC = -1;
    if (addIC < 0) addIC = getenv("GECKO_WJ_NOADDIC") ? 0 : 1;  // DEFAULT-ON
    uint32_t preShapeSlot = 0;
    // Same atom-key rule as the way fill above: the inline ADD hit guards
    // (oldShape AND key), so only cache adds whose runtime key is an atom.
    bool tryAdd = addIC && site < js::wasm::kWJPropSites && !id.isInt() &&
                  idv.isString() && idv.toString()->isAtom() &&
                  objv.toObject().is<js::NativeObject>();
    if (tryAdd) {
      js::NativeObject* nobj = &objv.toObject().as<js::NativeObject>();
      if (nobj->lookupPure(id).isNothing()) {
        preShapeSlot =
            uint32_t(js::wasm::WJInternShape(uintptr_t(nobj->shape())));
      } else {
        tryAdd = false;  // not an add (accessor/proto/non-writable own)
      }
    }
    RootedObject obj(cx, &objv.toObject());
    RootedValue keyv(cx, js::IdToValue(id));
    bool strict = gWJPropStrict[site] != 0;
    if (!js::SetObjectElement(cx, obj, keyv, val, strict)) return 1.0;
    if (tryAdd && preShapeSlot && obj->is<js::NativeObject>()) {
      js::NativeObject* nobj = &obj->as<js::NativeObject>();
      mozilla::Maybe<js::PropertyInfo> np = nobj->lookupPure(id);
      if (np.isSome() && np->isDataProperty() && np->writable()) {
        js::TaggedSlotOffset t = nobj->getTaggedSlotOffset(np->slot());
        if (t.isFixedSlot() &&
            uintptr_t(nobj->shape()) !=
                *reinterpret_cast<uintptr_t*>(uintptr_t(preShapeSlot))) {
          gWJAddOldShape[site] = preShapeSlot;
          gWJAddNewShape[site] =
              uint32_t(js::wasm::WJInternShape(uintptr_t(nobj->shape())));
          gWJAddOff[site] = t.offset();
          gWJAddKey[site] = uint32_t(uintptr_t(static_cast<void*>(idv.toString())));
        }
      }
    }
    return 0.0;
  }

  if (kind == js::wasm::WJH_SETPROP) {
    RootedValue objv(cx, JS::Value::fromRawBits(gWJScratch[0]));
    if (!objv.isObject()) return 1.0;
    RootedObject obj(cx, &objv.toObject());
    RootedValue idx(cx, JS::Value::fromRawBits(gWJScratch[1]));
    RootedValue val(cx, JS::Value::fromRawBits(gWJScratch[2]));
    // Site word: bit0 strict, bit1 property-INIT (define) semantics, attrs<<2.
    // Init ops (InitElem/InitProp families) must DEFINE: mirrors the Baseline
    // SetElem fallback's InitElemOperation (ToPropertyKey + DefineDataProperty
    // with the op's attrs) -- an OOB typed-array define THROWS where a set
    // silently skips (jit-test typedarray/define-property-oob).
    uint32_t hsite = uint32_t(siteF);
    if (hsite & 2) {
      RootedId id(cx);
      if (!js::ToPropertyKey(cx, idx, &id)) return 1.0;
      unsigned attrs = hsite >> 2;
      if (!js::DefineDataProperty(cx, obj, id, val, attrs)) return 1.0;
      gWJScratch[js::wasm::kWJResultSlot] = val.asRawBits();
      return 0.0;
    }
    bool strict = (hsite & 1) != 0;
    if (!js::SetObjectElement(cx, obj, idx, val, strict)) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] = val.asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_ARRAYPUSH) {
    // arr.push(val): set arr[length] = val (auto-grows dense + bumps length),
    // result = new length (Int32). MArrayPush guarantees an Array operand.
    RootedValue objv(cx, JS::Value::fromRawBits(gWJScratch[0]));
    if (!objv.isObject() || !objv.toObject().is<js::ArrayObject>()) return 1.0;
    RootedObject arr(cx, &objv.toObject());
    RootedValue val(cx, JS::Value::fromRawBits(gWJScratch[1]));
    uint32_t len = arr->as<js::ArrayObject>().length();
    RootedValue idx(cx, JS::NumberValue(double(len)));
    if (!js::SetObjectElement(cx, arr, idx, val, false)) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] =
        JS::Int32Value(int32_t(len + 1)).asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_ARRAYPOPSHIFT) {
    // site 0 = pop (remove last), site 1 = shift (remove first). Result = the
    // removed element (or undefined if empty). Routed through the native ops.
    RootedValue objv(cx, JS::Value::fromRawBits(gWJScratch[0]));
    if (!objv.isObject() || !objv.toObject().is<js::ArrayObject>()) return 1.0;
    RootedObject arr(cx, &objv.toObject());
    uint32_t len = arr->as<js::ArrayObject>().length();
    RootedValue res(cx, JS::UndefinedValue());
    if (len != 0) {
      if (int(siteF) == 1) {
        // shift: remove element 0. Fast path for PACKED dense arrays (the common
        // case -- marked's inlineQueue, etc.) via the exported element-move helper.
        // This previously ALWAYS deopted to PBL (return 1.0); but a JIT'd fn with a
        // hot shift then deopts on EVERY call, and repeatedly resuming the
        // ArrayPopShift op TRAPS (marked.parse: uncatchable crash after warmup --
        // a single `queue.shift()` reproduces it). Doing the shift here keeps it in
        // JIT and avoids the resume entirely. ArrayShiftMoveElements moves the dense
        // elements down and setLengthToInitializedLength() (= len-1 for a packed
        // array); a packed array's element 0 is never a hole.
        js::ArrayObject* aobj = &arr->as<js::ArrayObject>();
        if (js::IsPackedArray(aobj) && aobj->isExtensible() &&
            aobj->lengthIsWritable() &&
            !aobj->denseElementsHaveMaybeInIterationFlag()) {
          res = aobj->getDenseElement(0);
          js::ArrayShiftMoveElements(aobj);
        } else {
          return 1.0;  // non-packed/sparse/frozen shift: deopt to PBL (rare)
        }
      } else {
        // pop: read last element, truncate length.
        RootedValue idxv(cx, JS::NumberValue(double(len - 1)));
        if (!js::GetObjectElementOperation(cx, JSOp::GetElem, arr, objv, idxv,
                                           &res)) {
          return 1.0;
        }
        if (!js::SetLengthProperty(cx, arr, len - 1)) return 1.0;
      }
    }
    gWJScratch[js::wasm::kWJResultSlot] = res.asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_INSTANCEOF) {
    RootedValue v(cx, JS::Value::fromRawBits(gWJScratch[0]));
    RootedValue ctorv(cx, JS::Value::fromRawBits(gWJScratch[1]));
    if (!ctorv.isObject()) return 1.0;
    RootedObject ctor(cx, &ctorv.toObject());
    bool res = false;
    if (!js::InstanceofOperator(cx, ctor, v, &res)) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] = JS::BooleanValue(res).asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_INSTANCEOFPROTO) {
    // MInstanceOf: scratch[0]=obj (value), scratch[1]=proto (the RHS's .prototype
    // object, already resolved). Result = proto is on obj's prototype chain. A
    // non-object obj is never an instance -> false. (earley/Boyer does millions of
    // `x instanceof sc_Pair` -- this lets those functions JIT instead of bailing.)
    RootedValue v(cx, JS::Value::fromRawBits(gWJScratch[0]));
    RootedValue protov(cx, JS::Value::fromRawBits(gWJScratch[1]));
    bool res = false;
    if (v.isObject() && protov.isObject()) {
      RootedObject proto(cx, &protov.toObject());
      if (!js::IsPrototypeOf(cx, proto, &v.toObject(), &res)) return 1.0;
    }
    gWJScratch[js::wasm::kWJResultSlot] = JS::BooleanValue(res).asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_LAMBDA) {
    // MLambda: scratch[0]=envChain (object), scratch[1]=template function (object
    // constant). Clone the closure over the env. Matches Ion's OOL fallback. earley/
    // Boyer creates many closures (Scheme lambdas) -- lets those functions JIT.
    RootedValue envv(cx, JS::Value::fromRawBits(gWJScratch[0]));
    RootedValue funv(cx, JS::Value::fromRawBits(gWJScratch[1]));
    if (!funv.isObject() || !funv.toObject().is<JSFunction>() || !envv.isObject())
      return 1.0;
    RootedFunction fun(cx, &funv.toObject().as<JSFunction>());
    RootedObject env(cx, &envv.toObject());
    JSObject* res =
        js::LambdaOptimizedFallback(cx, fun, env, js::gc::Heap::Default);
    if (!res) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] = JS::ObjectValue(*res).asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_TYPEOFIS) {
    // MTypeOfIs: scratch[0]=operand value; site = (jstype<<1 | invert) where invert
    // is set for Ne/StrictNe. result = (typeof operand == jstype) XOR invert.
    RootedValue v(cx, JS::Value::fromRawBits(gWJScratch[0]));
    JSType t = js::TypeOfValue(v);
    uint32_t packed = uint32_t(siteF);
    JSType want = JSType(packed >> 1);
    bool invert = (packed & 1) != 0;
    bool match = (t == want) ^ invert;
    gWJScratch[js::wasm::kWJResultSlot] = JS::BooleanValue(match).asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_CTORALLOC) {
    // GC-correct ctor-`this` alloc for the inline construct: createWithShape via the
    // GC machinery (handles nursery registration + mid-life promotion correctly,
    // unlike the manual bump whose half-built `this` got swept). gWJNewShapeSlot
    // holds the ADDRESS of a Shape* (the per-site cached shape).
    js::Shape* sh = *reinterpret_cast<js::Shape**>(uintptr_t(gWJNewShapeSlot));
    if (!sh || !sh->isShared()) return 1.0;
    JS::Rooted<js::SharedShape*> shape(cx, &sh->asShared());
    js::PlainObject* obj = js::PlainObject::createWithShape(cx, shape);
    if (!obj) return 1.0;
    gWJScratch[js::wasm::kWJThisSlot] = JS::ObjectValue(*obj).asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_NEWCALLOBJ) {
    // MNewCallObject: allocate a CallObject (closure scope) with the traced shared
    // shape. Enclosing-env + callee slots are filled by subsequent StoreFixedSlot
    // ops (matches Ion's visitNewCallObject, which just calls createWithShape).
    js::Shape* sh = *reinterpret_cast<js::Shape**>(uintptr_t(gWJNewShapeSlot));
    if (!sh) return 1.0;
    JS::Rooted<js::SharedShape*> shape(cx, &sh->asShared());
    js::CallObject* obj =
        js::CallObject::createWithShape(cx, shape, js::gc::Heap(gWJNewHeap));
    if (!obj) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] = JS::ObjectValue(*obj).asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_CREATETHIS) {
    static int dbgCtCount = getenv("GECKO_WJ_CTCOUNT") ? 1 : 0;
    if (dbgCtCount) {
      static uint64_t ctc = 0;
      if ((++ctc % 100000) == 0)
        fprintf(stderr, "[wj-createthis] %llu\n", (unsigned long long)ctc);
    }
    // callee/newTarget are pre-staged by the emitter into the TRACED 62/63 slots
    // (scratch 0/1 now hold the construct's pre-staged args). Read them here and
    // LEAVE them in place: the following WJH_CONSTRUCT reuses 62/63 (and clears
    // them). They are GC-current (traced) across this helper's own allocation.
    RootedValue calleev(cx,
                        JS::Value::fromRawBits(gWJScratch[js::wasm::kWJCalleeSlot]));
    RootedValue ntv(
        cx, JS::Value::fromRawBits(gWJScratch[js::wasm::kWJNewTargetSlot]));
    if (!calleev.isObject() || !ntv.isObject()) return 1.0;
    RootedObject callee(cx, &calleev.toObject());
    RootedObject newTarget(cx, &ntv.toObject());
    RootedValue rval(cx);
    if (!js::jit::CreateThisFromIon(cx, callee, newTarget, &rval)) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] = rval.asRawBits();
    // Re-store the GC-current (post-alloc) callee/newTarget so the construct call
    // reads relocated pointers even if CreateThisFromIon moved them.
    gWJScratch[js::wasm::kWJCalleeSlot] = calleev.get().asRawBits();
    gWJScratch[js::wasm::kWJNewTargetSlot] = ntv.get().asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_CONSTRUCT) {
    static int ctorCount = getenv("GECKO_WJ_CTORCOUNT") ? 1 : 0;
    if (ctorCount) {
      static uint64_t n = 0;
      n++;
      if (n > 89999) fprintf(stderr, "[ctorcount] WJH_CONSTRUCT #%llu\n", (unsigned long long)n);
    }
    RootedValue fval(
        cx, JS::Value::fromRawBits(gWJScratch[js::wasm::kWJCalleeSlot]));
    RootedValue thisv(cx,
                      JS::Value::fromRawBits(gWJScratch[js::wasm::kWJThisSlot]));
    RootedValue newTarget(
        cx, JS::Value::fromRawBits(gWJScratch[js::wasm::kWJNewTargetSlot]));
    if (getenv("GECKO_WJ_CTORINLINEDBG")) {
      static uint64_t ent = 0;
      if ((++ent % 200000) == 0)
        fprintf(stderr, "[wj-ctorentry] %llu site=%u thisMagic=%d thisObj=%d nt==fval=%d\n",
                (unsigned long long)ent, uint32_t(siteF), thisv.isMagic(),
                thisv.isObject(), int(newTarget.asRawBits() == fval.asRawBits()));
    }
    static int cdbg = getenv("GECKO_WJ_CONSTRUCTDBG") ? 1 : 0;
    if (cdbg && !(fval.isObject() && fval.toObject().is<JSFunction>())) {
      const char* cn = fval.isObject() ? fval.toObject().getClass()->name : "?";
      fprintf(stderr,
              "[wjconstruct] BAD fval bits=%llx isObj=%d class=%s nt==fval=%d\n",
              (unsigned long long)fval.asRawBits(), fval.isObject(), cn,
              fval.asRawBits() == newTarget.asRawBits());
    }
    uint32_t argc = gWJCallArgc;
    // FUSED CreateThis: the backend now emits the JS_IS_CONSTRUCTING sentinel for
    // `this` (no separate WJH_CREATETHIS hop). Allocate `this` here -- exactly what
    // the old WJH_CREATETHIS did (CreateThisFromIon) -- folding two helper boundary
    // crossings into one per `new`. If CreateThisFromIon returns magic/null (derived
    // ctor / getter .prototype), thisv stays non-object and the generic construct
    // path below (InternalConstructWithProvidedThis) handles uninitialized-this.
    if (thisv.isMagic() && fval.isObject() && fval.toObject().is<JSFunction>() &&
        newTarget.isObject()) {
      // If this class was constructed before, the per-class cache holds the `this`
      // shape -> allocate directly via createWithShape, skipping CreateThisFromIon +
      // ThisShapeForFunction + their Rooted churn (~6% of raytrace).
      // GECKO_WJ_NOCTORALLOC reverts to CreateThisFromIon (this cached-shape alloc
      // is ~2.1x but has a GC-staleness ERR at default nursery -- 100% correct at
      // huge/no nursery -- pending the GC sweep).
      static int noCtorAlloc = getenv("GECKO_WJ_NOCTORALLOC") ? 1 : 0;
      uint32_t ntK = uint32_t(uintptr_t(static_cast<void*>(&fval.toObject())));
      js::SharedShape* cshape = nullptr;
      if (!noCtorAlloc && newTarget.asRawBits() == fval.asRawBits()) {
        for (int k = 0; k < kWJCtorCacheN; k++) {
          if (gWJCC_key[k] == ntK) { cshape = gWJCC_shape[k]; break; }
        }
      }
      if (cshape) {
        Rooted<js::SharedShape*> shape(cx, cshape);
        // GECKO_WJ_TENURECTOR: pretenure constructed `this` (allocate tenured)
        // to skip nursery churn for long-lived objects (splay tree nodes) -- tests
        // whether the minor-GC pause is splay's bottleneck. Ion pretenures hot
        // long-lived alloc sites via PretenuringInfo; this is the forced version.
        static int tenureCtor = getenv("GECKO_WJ_TENURECTOR") ? 1 : 0;
        js::PlainObject* obj = js::PlainObject::createWithShape(
            cx, shape, tenureCtor ? js::TenuredObject : js::GenericObject);
        if (!obj) return 1.0;
        thisv = JS::ObjectValue(*obj);
      } else {
        RootedObject calleeObj(cx, &fval.toObject());
        RootedObject ntObj(cx, &newTarget.toObject());
        RootedValue created(cx);
        if (!js::jit::CreateThisFromIon(cx, calleeObj, ntObj, &created)) return 1.0;
        thisv = created;
      }
      gWJScratch[js::wasm::kWJThisSlot] = thisv.get().asRawBits();
      // CTORINLINE: cache per-site ctor info from the FRESHLY-CREATED (pre-ctor,
      // EMPTY-shape) `this`. The backend inline path allocates `this` with THIS
      // initial shape so the ctor's compiled property-ADD (shape-transition) ops run
      // on the layout they expect. (Capturing the FINAL post-ctor shape corrupted
      // ~13% of earley's ctors: their transition ICs mismatched the pre-shaped obj.)
      static int ctorInlineFill = getenv("GECKO_WJ_NOCTORINLINE") ? 0 : 1;
      uint32_t cs = uint32_t(siteF);
      if (ctorInlineFill && cs && cs < js::wasm::kWJCtorSites &&
          !gWJCtorNoFill[cs] && thisv.isObject() &&
          thisv.toObject().is<js::PlainObject>() &&
          newTarget.asRawBits() == fval.asRawBits() && gEntries) {
        js::NativeObject& no = thisv.toObject().as<js::NativeObject>();
        js::Shape* sh = no.shape();
        JSFunction* cf = &fval.toObject().as<JSFunction>();
        // Mark forwarding wrappers permanently no-fill so we stop re-running this
        // whole block (WJIsForwardingWrapper hash-lookup + shape/fn checks) on EVERY
        // construct -- that per-construct churn, not the inline path, was raytrace's
        // ~29% CTORINLINE regression (its Class.create wrappers never fill).
        if (cf->isInterpreted() && cf->hasBytecode() &&
            WJIsForwardingWrapper(cx, cf->nonLazyScript())) {
          gWJCtorNoFill[cs] = 1;
        }
        // Do NOT inline-cache FORWARDING WRAPPERS (Prototype.js Class.create:
        // `function(){ this.initialize.apply(this, arguments) }`). call_indirect'ing
        // the wrapper would run its `.apply(this, arguments)` (arguments-object alloc
        // + fun_apply) -- exactly what the forwarding-construct fast path below skips
        // by calling `initialize` directly. CTORINLINE (direct ctors, earley cons)
        // and forwarding-construct (wrapper ctors, raytrace) are COMPLEMENTARY: let
        // wrappers fall through to the forwarding path (the inline gate then misses).
        if (sh->isShared() && no.numDynamicSlots() == 0 && cf->isInterpreted() &&
            cf->hasBytecode() && !WJIsForwardingWrapper(cx, cf->nonLazyScript())) {
          js::wasm::WasmJitObserveCall(cf->nonLazyScript());
          auto ce = gEntries->find(cf->nonLazyScript());
          if (ce != gEntries->end() &&
              ce->second.state == WJEntry::State::Compiled &&
              ce->second.tblSlot >= 0) {
            js::gc::AllocKind ak = no.allocKind();
            gWJCtorShape[cs] = uint32_t(uintptr_t(static_cast<void*>(sh)));
            gWJCtorSize[cs] = uint32_t(js::gc::Arena::thingSize(ak)) +
                              uint32_t(js::Nursery::nurseryCellHeaderSize());
            gWJCtorNfixed[cs] = no.numFixedSlots();
            gWJCtorTblIdx[cs] = ce->second.tblSlot;
            gWJCtorEnv[cs] =
                uint32_t(uintptr_t(static_cast<void*>(cf->environment())));
            gWJCtorCallee[cs] = uint32_t(uintptr_t(static_cast<void*>(cf)));
          }
        }
      }
    }
    // FORWARDING CONSTRUCT FAST PATH (default-on; GECKO_WJ_NOFWDCTOR disables).
    // For the Prototype.js `Class.create` wrapper `function(){ this.initialize.
    // apply(this, arguments); }`, skip the wrapper PBL frame + arguments object +
    // fun_apply native and call `this.initialize(args...)` DIRECTLY -- in JIT if
    // initialize is compiled. `this` (kWJThisSlot) was already created by the
    // preceding WJH_CREATETHIS; args are staged at gWJScratch[0..argc). The wrapper
    // returns undefined so the construct result is always `this`.
    static int noFwd = getenv("GECKO_WJ_NOFWDCTOR") ? 1 : 0;
    if (!noFwd && thisv.isObject() && fval.isObject() &&
        fval.toObject().is<JSFunction>() &&
        newTarget.asRawBits() == fval.asRawBits()) {
      // CACHE FAST PATH: same class constructed before -> call cached initialize
      // handle directly, skipping WJIsForwardingWrapper + GetProperty + gEntries.
      uint32_t ntKey = uint32_t(uintptr_t(static_cast<void*>(&fval.toObject())));
      int cslot = -1;
      for (int k = 0; k < kWJCtorCacheN; k++) {
        if (gWJCC_key[k] == ntKey) { cslot = k; break; }
      }
      if (cslot >= 0 && gWJCC_handle[cslot] >= 0 && argc >= gWJCC_nargs[cslot]) {
        gWJScratch[js::wasm::kWJThisSlot] = thisv.get().asRawBits();
        gWJCurrentEnv = gWJCC_env[cslot];
        double iptr = double(uintptr_t(static_cast<void*>(gWJScratch)));
        double iflag = wasmhost_call(gWJCC_handle[cslot], 0, &iptr, 1);
        if (iflag == 1.0) return 1.0;
        if (iflag == 0.0) {  // flag 2.0 = entry GGG mismatch: fall to the slow path
          gWJFastCalls++;
          gWJScratch[js::wasm::kWJResultSlot] = thisv.get().asRawBits();
          for (uint32_t i = 0; i < argc; i++) gWJScratch[i] = 0;
          gWJScratch[js::wasm::kWJThisSlot] = 0;
          gWJScratch[js::wasm::kWJCalleeSlot] = 0;
          gWJScratch[js::wasm::kWJNewTargetSlot] = 0;
          return 0.0;
        }
      }
      JSFunction* wfun = &fval.toObject().as<JSFunction>();
      if (wfun->isInterpreted() && wfun->hasBytecode() &&
          WJIsForwardingWrapper(cx, wfun->nonLazyScript())) {
        // Cached "initialize" atom -> avoid re-atomizing on every construct.
        static JSAtom* gInitAtom = nullptr;
        if (!gInitAtom) {
          gInitAtom = js::Atomize(cx, "initialize", 10);
          if (!gInitAtom) return 1.0;
        }
        Rooted<js::PropertyName*> initName(cx, gInitAtom->asPropertyName());
        RootedValue initv(cx);
        if (!js::GetProperty(cx, thisv, initName, &initv)) return 1.0;
        if (initv.isObject() && initv.toObject().is<JSFunction>()) {
          JSFunction* ifun = &initv.toObject().as<JSFunction>();
          // Re-store GC-current `this` (GetProperty may have GC'd/moved it).
          gWJScratch[js::wasm::kWJThisSlot] = thisv.get().asRawBits();
          bool done = false;
          if (gEntries && ifun->isInterpreted() && ifun->hasBytecode()) {
            JSScript* iscript = ifun->nonLazyScript();
            auto iit = gEntries->find(iscript);
            if (iit == gEntries->end() ||
                iit->second.state != WJEntry::State::Compiled) {
              js::wasm::WasmJitObserveCall(iscript);  // only when not yet compiled
              iit = gEntries->find(iscript);
            }
            if (iit != gEntries->end() &&
                iit->second.state == WJEntry::State::Compiled &&
                argc >= iit->second.nargs && iit->second.handle >= 0) {
              gWJCurrentEnv = uint32_t(
                  uintptr_t(static_cast<void*>(ifun->environment())));
              // Fill the per-class cache so future `new` of this class skip the
              // resolution (key = wrapper fn ptr; env = initialize's environment).
              uint32_t ntKey2 =
                  uint32_t(uintptr_t(static_cast<void*>(&fval.toObject())));
              int fillSlot = -1;
              for (int k = 0; k < kWJCtorCacheN; k++) {
                if (gWJCC_key[k] == 0 || gWJCC_key[k] == ntKey2) { fillSlot = k; break; }
              }
              if (fillSlot < 0) fillSlot = 0;  // evict slot 0 if full
              gWJCC_key[fillSlot] = ntKey2;
              gWJCC_handle[fillSlot] = iit->second.handle;
              gWJCC_nargs[fillSlot] = iit->second.nargs;
              gWJCC_env[fillSlot] = gWJCurrentEnv;
              // Capture the `this` shape for the inline-alloc fast path (only for
              // plain shared-shaped objects; else leave 0 -> CreateThisFromIon).
              gWJCC_shape[fillSlot] = nullptr;
              if (thisv.isObject() && thisv.toObject().is<js::PlainObject>()) {
                js::Shape* sh = thisv.toObject().as<js::NativeObject>().shape();
                if (sh->isShared()) gWJCC_shape[fillSlot] = &sh->asShared();
              }
              if (js::wasm::kWJEHABI)  // EHABI: stage boxed callee for the trampoline.
                gWJScratch[js::wasm::kWJCalleeSlot] = JS::ObjectValue(*ifun).asRawBits();
              double iptr = double(uintptr_t(static_cast<void*>(gWJScratch)));
              double iflag = wasmhost_call(iit->second.handle, 0, &iptr, 1);
              if (iflag == 1.0) return 1.0;  // initialize threw
              if (iflag == 0.0) {  // flag 2.0 -> !done -> VM path runs it
                gWJFastCalls++;
                done = true;
              }
            }
          }
          if (!done) {
            // initialize not compiled yet: call it via the VM (runs in PBL until
            // it warms up; observe above drives its compilation).
            RootedValue ignored(cx);
            JS::RootedValueVector iargv(cx);
            if (!iargv.reserve(argc)) return 1.0;
            for (uint32_t i = 0; i < argc; i++) {
              iargv.infallibleAppend(JS::Value::fromRawBits(gWJScratch[i]));
            }
            if (!JS::Call(cx, thisv, initv, JS::HandleValueArray(iargv),
                          &ignored)) {
              return 1.0;
            }
            gWJSlowCalls++;
          }
          // Construct result is `this` (the wrapper returns undefined).
          gWJScratch[js::wasm::kWJResultSlot] = thisv.get().asRawBits();
          for (uint32_t i = 0; i < argc; i++) gWJScratch[i] = 0;
          gWJScratch[js::wasm::kWJThisSlot] = 0;
          gWJScratch[js::wasm::kWJCalleeSlot] = 0;
          gWJScratch[js::wasm::kWJNewTargetSlot] = 0;
          return 0.0;
        }
      }
    }
    // FAST CONSTRUCT PATH: if the constructor is a compiled WJ function, run its
    // wasm body DIRECTLY on the already-created `this` (CreateThis staged it at
    // kWJThisSlot; args at gWJScratch[0..argc)), skipping InternalConstructWith-
    // ProvidedThis -> CreateThisFromIon -> PortableBaselineTrampoline -> the ctor
    // running in PBL. raytrace's `new Vector/Color/Ray/IntersectionInfo` were ~15-20%
    // of runtime on that VM dispatch + PBL-ctor + Rooted churn (profiled). The ctor
    // body is just field assignments, so running it as a normal call on the provided
    // `this` is correct; JS construct semantics (return `this` unless the ctor returns
    // an object) are applied below. GECKO_WJ_NOFASTCONSTRUCT disables for debugging.
    // OPT-IN (GECKO_WJ_FASTCONSTRUCT): direct-run the compiled ctor wasm. Disabled
    // by default -- it neither sped raytrace up (the PBL% is the recursive deopt
    // cascade, not the ctors) nor stayed correct (raised raytrace ERR rate). Kept
    // for future debugging of the construct dispatch path.
    static int fastC = getenv("GECKO_WJ_FASTCONSTRUCT") ? 1 : 0;
    if (fastC && gEntries && thisv.isObject() && fval.isObject() &&
        fval.toObject().is<JSFunction>()) {
      JSFunction* cfun = &fval.toObject().as<JSFunction>();
      if (cfun->isInterpreted() && cfun->hasBytecode() &&
          newTarget.asRawBits() == fval.asRawBits()) {  // new.target == callee (plain `new F`)
        JSScript* ccs = cfun->nonLazyScript();
        js::wasm::WasmJitObserveCall(ccs);
        auto cit = gEntries->find(ccs);
        if (cit != gEntries->end() &&
            cit->second.state == WJEntry::State::Compiled &&
            argc >= cit->second.nargs && cit->second.handle >= 0) {
          gWJCurrentEnv =
              uint32_t(uintptr_t(static_cast<void*>(cfun->environment())));
          if (js::wasm::kWJEHABI)  // EHABI: stage boxed callee for the trampoline.
            gWJScratch[js::wasm::kWJCalleeSlot] = JS::ObjectValue(*cfun).asRawBits();
          double cptr = double(uintptr_t(static_cast<void*>(gWJScratch)));
          double cflag = wasmhost_call(cit->second.handle, 0, &cptr, 1);
          if (cflag == 1.0) return 1.0;  // threw
          if (cflag == 0.0) {  // flag 2.0 = entry GGG mismatch: fall to slow construct
            // Construct result: the ctor's return if it's an object, else `this`.
            JS::Value cret = JS::Value::fromRawBits(gWJScratch[js::wasm::kWJResultSlot]);
            if (!cret.isObject()) {
              gWJScratch[js::wasm::kWJResultSlot] = thisv.asRawBits();
            }
            for (uint32_t i = 0; i < argc; i++) gWJScratch[i] = 0;
            gWJScratch[js::wasm::kWJThisSlot] = 0;
            gWJScratch[js::wasm::kWJCalleeSlot] = 0;
            gWJScratch[js::wasm::kWJNewTargetSlot] = 0;
            gWJFastCalls++;
            return 0.0;
          }
        }
      }
    }
    // (CTORINLINE cache is now filled from the PRE-ctor empty `this` above, right
    // after CreateThisFromIon -- the backend inline path needs the INITIAL shape the
    // ctor's transition ICs expect, not the final post-ctor shape.)
    js::ConstructArgs cargs(cx);
    if (!cargs.init(cx, argc)) return 1.0;
    for (uint32_t i = 0; i < argc; i++) {
      cargs[i].set(JS::Value::fromRawBits(gWJScratch[i]));
    }
    RootedValue rval(cx);
    if (!js::InternalConstructWithProvidedThis(cx, fval, thisv, cargs, newTarget,
                                               &rval)) {
      return 1.0;
    }
    gWJScratch[js::wasm::kWJResultSlot] = rval.asRawBits();
    // Clear staged operands: WJTraceRoots scans gWJScratch[0..kWJThisSlot] as GC
    // roots on EVERY GC, but these slots are dead after the helper returns. A
    // later GC tracing a stale (freed/moved) nursery pointer left here corrupts
    // the heap -- the raytrace `ray`-goes-garbage bug. (Regular calls pass args
    // in locals, not scratch, which is why only construct/helper paths hit this.)
    for (uint32_t i = 0; i < argc; i++) gWJScratch[i] = 0;
    gWJScratch[js::wasm::kWJThisSlot] = 0;
    gWJScratch[js::wasm::kWJCalleeSlot] = 0;
    gWJScratch[js::wasm::kWJNewTargetSlot] = 0;
    return 0.0;
  }

  if (kind == js::wasm::WJH_NEWPLAIN) {
    // gWJNewShapeSlot = address of the traced shape-pool slot (relocated by
    // WJTraceRoots); load the GC-current Shape* from it.
    js::Shape* sh =
        *reinterpret_cast<js::Shape**>(uintptr_t(gWJNewShapeSlot));
    if (!sh) return 1.0;
    JS::Rooted<js::SharedShape*> shape(cx, &sh->asShared());
    JSObject* obj = js::NewPlainObjectOptimizedFallback(
        cx, shape, js::gc::AllocKind(gWJNewAux), js::gc::Heap(gWJNewHeap));
    if (!obj) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] = JS::ObjectValue(*obj).asRawBits();
    return 0.0;
  }
  if (kind == js::wasm::WJH_NEWARROBJ) {
    js::gc::AllocKind ak = GuessArrayGCKind(gWJNewAux);
    js::NewObjectKind nk = gWJNewHeap == uint32_t(js::gc::Heap::Tenured)
                               ? js::TenuredObject
                               : js::GenericObject;
    js::ArrayObject* arr =
        js::NewArrayObjectOptimizedFallback(cx, gWJNewAux, ak, nk);
    if (!arr) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] = JS::ObjectValue(*arr).asRawBits();
    return 0.0;
  }
  if (kind == js::wasm::WJH_NEWARR) {
    js::ArrayObject* arr = js::NewArrayOperation(cx, gWJNewAux);
    if (!arr) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] = JS::ObjectValue(*arr).asRawBits();
    return 0.0;
  }

  return 1.0;
}

// Compiled-callee invoke trampoline ("m"."call" import). Unused until the call
// path is rebuilt (Phase 5).
extern "C" EMSCRIPTEN_KEEPALIVE double wasmjit_invoke(int site, int argc) {
  return 1.0;
}

// Registered as an extra GC roots tracer (gc/RootMarking.cpp). The scratch buffer
// holds live boxed JS::Values (args + receiver + result) across a JIT'd call.
// GC-root shadow stack (see WasmJitBackend.h). Active region is [0, gWJRootSP).
namespace js {
namespace wasm {
alignas(8) uint64_t gWJCallRoots[kWJCallRootsSize];
uint32_t gWJRootSP = 0;
// JIT call-recursion depth. JIT'd wasm frames don't move the shadow stack that
// SpiderMonkey's native-stack quota watches, so deep JIT->JIT recursion silently
// overflows V8's wasm execution stack (uncatchable RangeError) instead of throwing.
// Each JIT fn bumps this at entry and, past the limit, deopts to PBL (whose deep C++
// frames DO trip the real quota -> catchable "too much recursion"). Self-correcting:
// each frame RESTORES it to the entry value at exit, so cold/leaked paths re-balance.
int32_t gWJJitDepth = 0;
}  // namespace wasm
}  // namespace js

// DEBUG (GECKO_WJ_ROOTLOG): log every OBJECT-valued Value root right before the GC
// traces it during a minor (tenuring) collection. The LAST line printed before a
// crash pinpoints the stale root's array+index (the misrooted JIT value of #19).
static inline void WJDbgLogRoot(JSTracer* trc, const char* cat, uint32_t idx,
                                uint64_t* slot) {
  static int rootLog = getenv("GECKO_WJ_ROOTLOG") ? 1 : 0;
  if (!rootLog) return;
  if (trc->kind() == JS::TracerKind::Marking) return;  // minor GC only
  JS::Value v = JS::Value::fromRawBits(*slot);
  if (!v.isObject()) return;
  JSObject* o = &v.toObject();
  uintptr_t classWord = *reinterpret_cast<uintptr_t*>(o);  // shape ptr (wasm32)
  fprintf(stderr, "[wjrootlog] %s[%u] obj=%p shapeWord=%#lx\n", cat, idx,
          (void*)o, (unsigned long)classWord);
}

// Trace a JIT Value-root slot, GUARDING against a GC-tagged-but-INVALID cell during
// MAJOR-GC marking. A stale/garbage staging or spill slot whose bits happen to carry an
// object/string tag but a dangling pointer would otherwise be pushed to the mark stack and
// crash js::gc::MarkingTracerT::processMarkStackTop with "memory access out of bounds" in the
// mark phase (the in-browser NonIncrementalGC->markPhase crash; repro'd here via
// GECKO_GCZEAL=2,1 on realapp/acorn). During major-GC marking the nursery has been evicted, so
// every LIVE cell is tenured and passes IsCellPointerValid -> an invalid cell is provably
// garbage, never a real root, and SKIPPING it is sound (drops only garbage). Restricted to
// TracerKind::Marking: during a minor/tenuring trace a valid nursery cell may NOT pass
// IsCellPointerValid, so there we must trace unconditionally (the existing, working path).
static inline void WJTraceValueRoot(JSTracer* trc, uint64_t* slot, const char* cat,
                                    uint32_t idx) {
  if (trc->kind() == JS::TracerKind::Marking) {
    JS::Value v = JS::Value::fromRawBits(*slot);
    if (v.isGCThing()) {
      js::gc::Cell* c = v.toGCThing();
      if (!js::gc::IsCellPointerValid(c)) {
        static int logBad = getenv("GECKO_WJ_ROOTVALIDATE") ? 1 : 0;
        if (logBad)
          fprintf(stderr, "[wj-badroot] %s[%u] bits=%016llx cell=%p SKIPPED (garbage root)\n",
                  cat, idx, (unsigned long long)*slot, (void*)c);
        return;  // garbage, not a live root
      }
      // DANGLING-root guard (default-on). A JIT call-root shadow slot can hold a STALE
      // boxed OBJECT pointer to a cell that has since been SWEPT (freed): the cell is
      // still a valid LOCATION (passes IsCellPointerValid above) but its shape word is
      // poison (JS_SWEPT_TENURED_PATTERN 0x4b) / not a valid Shape. Tracing it makes the
      // full-GC mark phase dereference the freed cell -> memory-access-out-of-bounds in
      // js::gc::MarkingTracerT::processMarkStackTop (the user-reported gecko.wasm crash;
      // repro GECKO_GCZEAL=2,1 on realapp/acorn, localized to gWJCallRoots[8]). A LIVE
      // object always has a valid Shape*, so skipping an object whose shape is invalid is
      // SOUND -- it never drops a live edge. SOUND + COMPLETE here: a stale leftover call-
      // root slot is only ever READ by the GC tracer, never reloaded by the JIT (the JIT
      // reloads exactly the slots it spilled), so not tracing it has no value effect.
      // For an OBJECT, the strong check: a LIVE object always has a valid Shape*, so an
      // invalid shape word == dangling-to-swept/dead object -> skip.
      bool dangling = false;
      if (v.isObject()) {
        uintptr_t shapeWord = uintptr_t(*reinterpret_cast<uint32_t*>(
            reinterpret_cast<char*>(c) + offsetof(JS::shadow::Object, shape)));
        dangling =
            !js::gc::IsCellPointerValid(reinterpret_cast<js::gc::Cell*>(shapeWord));
      } else {
        // For a non-object GC thing (string/symbol/bigint) a stale spill can likewise
        // dangle to a SWEPT-tenured cell whose whole body is poison. Its first word
        // (flags/header) is then JS_SWEPT_TENURED_PATTERN (0x4b4b4b4b) -- a value a LIVE
        // string/symbol/bigint header never holds (headers are small bitfields), so
        // skipping it is SOUND. (No shape word to validate for these types.)
        dangling = (*reinterpret_cast<uint32_t*>(c) == 0x4b4b4b4bu);
      }
      if (dangling) {
        static int logBad = getenv("GECKO_WJ_ROOTVALIDATE") ? 1 : 0;
        if (logBad)
          fprintf(stderr,
                  "[wj-danglingroot] %s[%u] cell=%p bits=%016llx SKIPPED (dangling to "
                  "swept cell)\n",
                  cat, idx, (void*)c, (unsigned long long)*slot);
        return;
      }
    }
  }
  JS::TraceRoot(trc, reinterpret_cast<JS::Value*>(slot), cat);
}

// Same guard for a raw GC-pointer root slot (wasm32: the stored word IS the T*; the slot
// storage may be uint32_t / uintptr_t / an actual T*). Skips a garbage/dangling pointer
// during major-GC marking so it never reaches the mark stack. T = cell type (explicit),
// Slot = storage type (deduced).
template <typename T, typename Slot>
static inline void WJTracePtrRoot(JSTracer* trc, Slot* slot, const char* cat) {
  uintptr_t raw = (uintptr_t)(*slot);
  if (raw == 0) return;  // empty slot
  if (trc->kind() == JS::TracerKind::Marking) {
    if (!js::gc::IsCellPointerValid(reinterpret_cast<js::gc::Cell*>(raw))) {
      static int logBad = getenv("GECKO_WJ_ROOTVALIDATE") ? 1 : 0;
      if (logBad)
        fprintf(stderr, "[wj-badroot] %s bits=%08x SKIPPED (garbage ptr)\n", cat,
                (unsigned)raw);
      return;  // garbage cell -> skip (see WJTraceValueRoot rationale)
    }
  }
  js::TraceRoot(trc, reinterpret_cast<T**>(slot), cat);
}

extern "C" EMSCRIPTEN_KEEPALIVE void WJTraceRoots(JSTracer* trc, void*) {
  static int rootLogM = getenv("GECKO_WJ_ROOTLOG") ? 1 : 0;
  if (rootLogM && trc->kind() != JS::TracerKind::Marking)
    fprintf(stderr, "[wjrootlog] === WJTraceRoots START (resumeActive=%d sp=%u) ===\n",
            int(gWJResumeActive), gWJRootSP);
  if (getenv("GECKO_WJ_TRACEDBG")) {
    static uint64_t n = 0;
    // trc->kind(): Marking (major) vs Tenuring/MinorSweeping (minor). Log kind+count.
    fprintf(stderr, "[wjtrace] #%llu kind=%d\n", (unsigned long long)(++n),
            int(trc->kind()));
  }
  for (uint32_t i = 0; i <= js::wasm::kWJThisSlot; i++) {
    WJDbgLogRoot(trc, "scratch", i, &gWJScratch[i]);
    WJTraceValueRoot(trc, &gWJScratch[i], "wjscratch", i);
  }
  // Deferred-compile queue: keep queued scripts live + pointer-current across the
  // defer window (enqueue in one task, drain at a later idle/task boundary).
  for (JSScript*& s : gWJDeferQueue) {
    js::TraceRoot(trc, &s, "wjdeferq");
  }
  for (uint32_t i = 0; i < gWJConstPoolCount; i++) {
    WJDbgLogRoot(trc, "const", i, &gWJConstPool[i]);
    WJTraceValueRoot(trc, &gWJConstPool[i], "wjconst", i);
  }
  for (uint32_t i = 0; i < gWJShapePoolCount; i++) {
    WJTracePtrRoot<js::Shape>(trc, &gWJShapePool[i], "wjshape");
  }
  // Deopt-resume script pool: trace+RELOCATE so the deopt spill (which loads these
  // slots at runtime) stores a GC-current script into gWJResumeScriptPtr under compaction.
  for (uint32_t i = 0; i < gWJScriptPoolCount; i++) {
    WJTracePtrRoot<JSScript>(trc, &gWJScriptPool[i], "wjscript");
  }
  // Per-class construct-cache `this` shapes: trace+relocate so cache-hit alloc
  // uses a GC-current shape (and the key/env are re-validated on miss).
  for (int i = 0; i < kWJCtorCacheN; i++) {
    WJTracePtrRoot<js::Shape>(trc, &gWJCC_shape[i], "wjcc");
  }
  // Prop-IC cached shapes: trace+relocate (wasm32, so the uint32 IS the Shape*).
  // Keeps cached shapes live and pointer-current, so a shape match is always
  // correct (no stale/reused-pointer hazard). The paired offset stays valid
  // because a Shape's property layout is immutable.
  {
    uint32_t n = js::wasm::gWJNextPropSite * js::wasm::kWJPropWays;
    for (uint32_t i = 0; i < n; i++) {
      WJTracePtrRoot<js::Shape>(trc, &gWJPropShape[i], "wjpropic");
      WJTracePtrRoot<JSObject>(trc, &gWJPropHolder[i], "wjpropholder");
      // Store-IC way keys are atom JSString*s compared by pointer in the hit
      // path: trace+relocate so the compare stays current across a moving GC.
      WJTracePtrRoot<JSString>(trc, &gWJPropWayKey[i], "wjpropkey");
    }
    for (uint32_t s = 0; s < js::wasm::gWJNextPropSite; s++) {
      WJTracePtrRoot<JSString>(trc, &gWJAddKey[s], "wjaddkey");
    }
  }
  // GetName IC: trace+relocate the cached holder object AND its shape so both
  // stay pointer-current across a compacting GC (else the holder load reads a
  // moved object / the shape guard compares a stale pointer -> wrong value).
  {
    uint32_t n = js::wasm::gWJNextNameSite;
    for (uint32_t i = 0; i < n; i++) {
      if (gWJNameHolder[i]) {
        WJTracePtrRoot<JSObject>(trc, &gWJNameHolder[i], "wjnameholder");
        WJTracePtrRoot<js::Shape>(trc, &gWJNameShape[i], "wjnameshape");
      }
    }
  }
  // Polymorphic CALL IC: trace+relocate the cached callee function pointers so the
  // IC's `gWJCallFn[w] == currentCallee` dispatch compare stays pointer-current
  // across a GC that MOVES a cached function (nursery promotion on a minor GC, or
  // compaction). Without this, a moved function leaves a stale cached pointer; if a
  // DIFFERENT function is later allocated at that stale address the compare FALSE-
  // MATCHES and dispatches via the stale table slot -> calls the WRONG function ->
  // wrong result (raytrace "Scene rendered incorrectly"; NO_NURSERY masked it by
  // never moving objects). gWJCallTblIdx is a plain table index (not a pointer) and
  // needs no tracing. Functions kept alive this way (a strong root) is a bounded
  // acceptable leak, matching gWJShapePool/gWJPropShape.
  {
    uint32_t n = js::wasm::gWJNextCallSite * js::wasm::kWJCallWays;
    for (uint32_t i = 0; i < n; i++) {
      WJTracePtrRoot<JSObject>(trc, &gWJCallFn[i], "wjcallfn");
    }
  }
  // Per-site ctor inline cache: callee fn, `this` shape, and ctor env are GC ptrs.
  // If any moves and a slot isn't updated, the backend's `callee==gWJCtorCallee`
  // gate would mis-hit / the cached shape would be stale -> trace them all.
  {
    uint32_t n = js::wasm::gWJNextCtorSite + 1;
    if (n > js::wasm::kWJCtorSites) n = js::wasm::kWJCtorSites;
    for (uint32_t i = 0; i < n; i++) {
      WJTracePtrRoot<JSObject>(trc, &gWJCtorCallee[i], "wjctorcallee");
      WJTracePtrRoot<js::Shape>(trc, &gWJCtorShape[i], "wjctorshape");
      WJTracePtrRoot<JSObject>(trc, &gWJCtorEnv[i], "wjctorenv");
    }
  }
  uint32_t sp = gWJRootSP;
  if (sp > js::wasm::kWJCallRootsSize) sp = js::wasm::kWJCallRootsSize;
  // DEBUG (GECKO_WJ_SLOTWATCH=N): at each GC, dump slot N's contents + validity + script
  // + the current gWJRootSP, so slot corruption can be localized to a specific GC (does
  // the slot go foreign while SP < N+1, i.e. untraced?). Gated; no behavior change.
  {
    static const char* swEnv = getenv("GECKO_WJ_SLOTWATCH");
    if (swEnv) {
      uint32_t wslot = uint32_t(atoi(swEnv));
      JS::Value v = JS::Value::fromRawBits(gWJCallRoots[wslot]);
      const char* kind = "prim";
      void* scr = nullptr;
      bool cellOk = false;
      if (v.isObject()) {
        js::gc::Cell* c = reinterpret_cast<js::gc::Cell*>(&v.toObject());
        cellOk = js::gc::IsCellPointerValid(c);
        if (cellOk && v.toObject().is<JSFunction>()) {
          JSFunction* fn = &v.toObject().as<JSFunction>();
          kind = "fn";
          if (fn->isInterpreted() && fn->hasBaseScript()) scr = (void*)fn->baseScript();
        } else if (cellOk) kind = "obj";
      }
      static uint64_t swN = 0;
      if (swN++ < 4000)
        fprintf(stderr, "[wj-slotwatch] gc slot%u sp=%u traced=%d kind=%s cellOk=%d script=%p bits=%016llx tracer=%d\n",
                wslot, sp, int(wslot < sp), kind, int(cellOk), scr,
                (unsigned long long)gWJCallRoots[wslot], int(trc->kind()));
    }
  }
  for (uint32_t i = 0; i < sp; i++) {
    WJDbgLogRoot(trc, "callroot", i, &gWJCallRoots[i]);
    WJTraceValueRoot(trc, &gWJCallRoots[i], "wjcallroot", i);
  }
  // Active deopt-resume spill area: the boxed pointers spilled at the deopt must
  // survive a GC triggered between the spill and WJH_RESUME reading them.
  if (gWJResumeActive) {
    uint32_t rc = gWJResumeValsCount;
    if (rc > 1024) rc = 1024;
    for (uint32_t i = 0; i < rc; i++) {
      WJDbgLogRoot(trc, "resumeval", i, &gWJResumeVals[i]);
      WJTraceValueRoot(trc, &gWJResumeVals[i], "wjresumeval", i);
    }
    // gWJResumeActuals is NOT traced: the graft consumes it into gWJResumeVals
    // (traced) before any allocation, and stale slots from earlier deopts may
    // hold dangling pointers (tracing them crashed the object-args repro).
  }
  if (rootLogM && trc->kind() != JS::TracerKind::Marking)
    fprintf(stderr, "[wjrootlog] === WJTraceRoots DONE ===\n");
}
