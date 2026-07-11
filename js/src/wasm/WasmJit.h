/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*-
 * vim: set ts=8 sts=2 et sw=2 tw=80:
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef wasm_WasmJit_h
#define wasm_WasmJit_h

// Re-architected JS->wasm-bytecode JIT. See WASMJIT_REARCH_PLAN.md.
//
// This build has no native JIT backend (JS_CODEGEN_NONE): SpiderMonkey's
// Baseline/Ion emit machine code that can't run in the wasm sandbox. Instead,
// hot JS functions are lowered to a *guest* WebAssembly module which the host
// engine (e.g. V8) compiles to native code. The front-end reuses Ion's
// WarpBuilder (real JSScript => full JS-tier MIR), runs OptimizeMIR, and a
// first-class MIR->wasm back-end emits the module.
//
// Modules:
//   WasmJitWarp.cpp     -- front-end: WarpOracle + WarpBuilder + OptimizeMIR
//   WasmJitBackend.cpp  -- back-end: optimized MIR -> wasm bytes
//   WasmJitRuntime.cpp  -- trigger, host compile/instantiate, call routing,
//                          arg/result marshalling, wjhelp dispatch, GC tracing

#include <stdint.h>

#include "js/TypeDecls.h"  // JSScript / JSObject / JS::Value (correct visibility)

namespace js {
namespace wasm {

// Called on every hot scripted call from the interpreter / PBL fast path.
// Returns true once `script` is compiled and installed (the caller then routes
// the call through WasmJitRunCall); false to keep running in the interpreter.
extern bool WasmJitObserveCall(JSScript* script);
extern int WasmJitPreCall(JSScript* script);  // 0=no-jit 1=compiled 2=may-compile

// Compile any functions that crossed the warmup threshold while GECKO_WJ_DEFERCOMPILE
// deferred their (synchronous) compile off the critical path. Call at an idle / task
// boundary (browser: between event-loop tasks; node: between top-level runs). No-op
// unless deferral queued something.
extern void WasmJitDrainDeferred();

// Run the installed wasm for `script`. Returns 1 if it handled the call (result
// in *retBits), 0 to fall back to the interpreter.
extern int WasmJitRunCall(JSScript* script, uint64_t thisBits,
                          const JS::Value* args, uint32_t argc,
                          JSObject* envChain, uint64_t* retBits);

// True while emitted wasm-JIT code is currently executing on the stack. Backs
// the shell's inWasmJit()/inJit()/inIon() so the jit-test warm-up spin loops
// (`do { f(); } while (!inIon())`) can detect that `f` got JIT-compiled.
extern bool WasmJitInWasm();

// Discard ALL installed wasm-JIT code (entries reset to cold, call ICs
// cleared). Called when an invalidating fuse pops: compiled code may have
// baked a fuse-guarded elision (e.g. the array-destructuring iterator
// protocol), and unlike Ion we register no per-script fuse dependencies --
// a full flush on this rare event is the sound equivalent. Re-warmed
// recompiles read the popped fuse and drop the elision.
extern void WasmJitInvalidateAll(const char* reason);

}  // namespace wasm
}  // namespace js

#endif  // wasm_WasmJit_h
