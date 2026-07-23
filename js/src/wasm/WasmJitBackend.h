/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*-
 * vim: set ts=8 sts=2 et sw=2 tw=80:
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef wasm_WasmJitBackend_h
#define wasm_WasmJitBackend_h

#include <stdint.h>

namespace js {
namespace jit {
class MIRGenerator;
class MIRGraph;
class WarpSnapshot;
}  // namespace jit
namespace wasm {

class Encoder;

// MIR->wasm back-end (Phase 1+). Emits the wasm function BODY (the locals
// declaration + instruction stream that goes inside the single Code-section
// entry) for an OptimizeMIR'd Warp graph, using the scratch-buffer ABI:
//   param0 (f64) = base address of the runtime's gWJScratch buffer
//   args are boxed JS::Value bits at gWJScratch[i] (i64), `this` at kWJThisOff
//   result (boxed) is stored to gWJScratch[kWJResultOff]
//   the function returns an f64 deopt flag: 0.0 = ok, 1.0 = bail back to PBL
// Returns false (caller keeps the function in PBL) if the graph uses a node the
// back-end can't lower yet.
bool WJEmitBody(jit::MIRGenerator& mir, jit::MIRGraph& graph, uint32_t nargs,
                bool useThis, jit::WarpSnapshot* snapshot, Encoder& e);

// Emit the per-module entry trampoline body (wasm type (f64 scratchbase)->f64):
// loads `this` + args from gWJScratch and calls the module's main (func idx 1)
// with the register-argument ABI. Used for the host-entry path. Returns false on
// encoder failure.
bool WJEmitTrampoline(Encoder& e);
void WJDumpDeoptSiteHist();

// GC-constant pool (WASMJIT_REARCH_PLAN.md §4.2). Interns a boxed JS::Value (a
// GC pointer baked into emitted wasm cannot be relocated by a moving GC) and
// returns the *byte address* of its pool slot, which the emitted code loads at
// runtime. WJTraceRoots traces + updates the pool. Defined in WasmJitRuntime.cpp.
// Returns 0 if the pool is full (caller bails the compile).
uintptr_t WJInternConstant(uint64_t valueBits);

// Intern a Shape* for a GuardShape: returns the byte address of a GC-traced pool
// slot holding the (relocatable) shape pointer. The emitted guard loads the slot
// at runtime so a GC-moved shape doesn't cause a permanent deopt. 0 if pool full.
uintptr_t WJInternShape(uintptr_t shapeBits);

// Intern a JSScript* for a deopt resume spill: returns the byte address of a
// GC-traced+relocated pool slot holding the (relocatable) script pointer. The
// emitted deopt LOADS the slot at runtime and stores it into gWJResumeScriptPtr, so
// a COMPACTING GC that moved the script since compile-time doesn't spill a stale
// pointer (was: baked I32Const(script) -> stale -> resume OOB). 0 if pool full.
uintptr_t WJInternScript(uintptr_t scriptBits);

// wjhelp() helper kinds (the f64 first argument to the imported "m"."help").
enum WJHelpKind : int {
  WJH_RESUME = 1,   // deopt: rebuild a PBL frame from gWJResume* and run from pc
  WJH_CALL = 2,     // invoke gWJCallCallee with argc args staged in gWJScratch
  WJH_SETSLOT = 3,  // gWJHelpObj->setSlot(gWJHelpSlot, gWJHelpVal) (barriered)
  // Generic VM-op helpers. Operands are staged boxed in gWJScratch[0..]; the
  // (boxed) result is left in gWJScratch[kWJResultSlot]; returns 0.0 ok / 1.0
  // threw. May GC -- caller roots live values around the call.
  WJH_GETPROP = 4,   // scratch[0]=obj, scratch[1]=id  -> GetElementOperation
  WJH_SETPROP = 5,   // scratch[0]=obj, scratch[1]=id, scratch[2]=val, strict flag=site
  WJH_GETELEM = 6,   // (alias of GETPROP path; kept for clarity at the call site)
  WJH_INSTANCEOF = 7,  // scratch[0]=obj, scratch[1]=ctor -> InstanceofOperation
  WJH_ARRAYPUSH = 8,   // scratch[0]=array(obj), scratch[1]=val -> push, result=newlen
  WJH_ARRAYPOPSHIFT = 9,  // scratch[0]=array(obj), site: 0=pop 1=shift -> result=elem
  WJH_CREATETHIS = 10,    // scratch[0]=callee, scratch[1]=newTarget -> result=this
  WJH_CONSTRUCT = 11,     // gWJCallCallee + scratch[0..argc-1] args + this@kWJThisSlot
                          // + gWJConstructNewTarget -> InternalConstructWithProvidedThis
  WJH_POSTBARRIER = 12,   // gWJHelpObj = container -> jit::PostWriteBarrier (store buffer)
  WJH_PREBARRIER = 13,    // gWJHelpVal = old value -> gc::ValuePreWriteBarrier (incremental mark)
  WJH_PROPIC = 14,        // prop-IC miss: scratch[0]=obj, site arg -> get gWJPropKey[site],
                          // fill gWJPropShape/Off[site], result -> scratch[kWJResultSlot]
  WJH_SETPROPIC = 15,     // set-prop-IC miss: scratch[0]=obj, scratch[1]=value, site arg
                          // -> fill gWJPropShape/Off[site] (writable own data prop) + store
  WJH_NEWPLAIN = 16,      // gWJNewShapeSlot(pool idx)/gWJNewAux(allocKind)/gWJNewHeap
                          // -> NewPlainObjectOptimizedFallback; result -> kWJResultSlot
  WJH_NEWARROBJ = 17,     // gWJNewShapeSlot/gWJNewAux(length)/gWJNewHeap
                          // -> NewArrayObjectOptimizedFallback
  WJH_NEWARR = 18,        // gWJNewAux(length) -> NewArrayOperation (NewArray/DynamicLength)
  WJH_BINARYARITH = 19,   // scratch[0]=lhs, scratch[1]=rhs, site=JSOp -> *Values; boxed result
  WJH_UNARYARITH = 20,    // scratch[0]=operand, site=JSOp -> Neg/BitNot/Inc/Dec/Pos/ToNumeric
  WJH_GETNAME = 21,       // scratch[0]=envObj, scratch[1]=name(StringValue) -> GetEnvironmentName
  WJH_TOPROPKEY = 22,     // scratch[0]=input -> ToPropertyKey -> IdToValue (boxed)
  WJH_CHARCODEAT = 23,    // scratch[0]=string, scratch[1]=index -> jit::CharCodeAt (Int32)
  WJH_FROMCHARCODE = 24,  // scratch[0]=code(Int32) -> StringFromCharCode (String)
  WJH_TOSTRING = 25,      // scratch[0]=input -> ToString (String)
  WJH_COMPARE = 26,       // scratch[0]=lhs, scratch[1]=rhs, site=JSOp -> Boolean (==,!=,<,<=,>,>=,===,!==)
  WJH_NEWOBJECT = 27,     // gWJNewObjScript/gWJNewObjPcOff -> NewObjectOperation (object literal)
  WJH_BINDNAME = 28,      // scratch[0]=envChain, scratch[1]=name(StringValue) -> LookupNameUnqualified (Object)
  WJH_GROWSLOTS = 29,     // scratch[0]=object, gWJNewAux=newCapacity -> NativeObject::growSlotsPure (AllocateAndStoreSlot)
  WJH_TOINT32 = 30,       // scratch[0]=value -> JS::ToInt32 (Int32); no-deopt ToInt32 for `x|0`/`&` on a boxed value
  WJH_CHECKCELL = 251,    // DEBUG (GECKO_WJ_STOREVALIDATE): gWJHelpObj = ptr to validate; abort if forwarded/invalid.
                          // High UNIQUE sentinel — MUST NOT collide with any real WJHelpKind (was 30 == WJH_TOINT32,
                          // which would make TruncateToInt32(non-number `~~x`/`&`) hit the crash validator). See the
                          // WJH_TRACE=250 note; debug-helper kinds live in the 250+ sentinel range.
  WJH_INSTANCEOFPROTO = 31,  // scratch[0]=obj(value), scratch[1]=proto(object) -> IsPrototypeOf (Boolean); MInstanceOf
  WJH_LAMBDA = 32,           // scratch[0]=envChain(object), scratch[1]=templateFun(object) -> LambdaOptimizedFallback (Object); MLambda
  WJH_TYPEOFIS = 33,         // scratch[0]=operand(value), site=(jstype<<1|invert) -> TypeOfValue==jstype (Boolean); MTypeOfIs
  WJH_NEWCALLOBJ = 34,       // gWJNewShapeSlot=traced CallObject shared shape, gWJNewHeap -> CallObject::createWithShape (Object); MNewCallObject
  WJH_CTORALLOC = 35,        // gWJNewShapeSlot=&shapeptr -> PlainObject::createWithShape -> gWJScratch[kWJThisSlot]; GC-correct ctor `this` alloc (CTORINLINE isolation/fix)
  WJH_DBGPTR = 36,           // DEBUG (GECKO_WJ_VGDBG): gWJHelpObj=payload, gWJHelpVal=val bits -> log the value the valNursery check is about to chunk-load
  WJH_LINEARIZE = 37,        // scratch[0]=string(value) -> str->ensureLinear (flatten rope) (String); MLinearizeString / MLinearizeForCharAccess
  WJH_STROP = 38,            // generic string method, site=sub-op: 0 includes,1 indexOf,2 lastIndexOf,3 startsWith,4 endsWith (str,search -> Bool/Int32); 5 substr (str,begin,len -> String); 6 toLowerCase,7 toUpperCase (str -> String)
  WJH_TYPEOF = 39,           // scratch[0]=value -> js::TypeOfValue (Int32 JSType); MTypeOf
  WJH_ISARRAY = 40,          // scratch[0]=value -> Array.isArray (Boolean); MIsArray
  WJH_REGEXPCLONE = 41,      // scratch[0]=source RegExpObject(boxed) -> CloneRegExpObject (Object); MRegExp
  WJH_TYPEOFNAME = 42,       // scratch[0]=int32 JSType -> TypeName atom (String); MTypeOfName
  WJH_NEWLEXENV = 43,        // gWJLexScope=LexicalScope* (raw) -> BlockLexicalEnvironmentObject::createWithoutEnclosing (Object); MNewLexicalEnvironmentObject
  WJH_CLOSEITER = 44,        // scratch[0]=iter(boxed Object), site=completionKind -> js::CloseIterOperation (for-of cleanup); MCloseIterCache
  WJH_ARRAYSLICE = 45,       // scratch[0]=array(boxed Object),[1]=begin(int32),[2]=end(int32) -> js::ArraySliceDense (Object); MArraySlice (caller guards packed)
  WJH_ARRAYJOIN = 46,        // scratch[0]=array(boxed Object),[1]=sep(boxed String) -> js::jit::ArrayJoin (String); MArrayJoin
  WJH_INCACHE = 47,          // scratch[0]=key(Value),[1]=obj(boxed Object) -> js::jit::OperatorIn (Boolean); MInCache (`key in obj`)
  WJH_HASPROP = 48,          // scratch[0]=obj(boxed Object),[1]=idVal(Value), site=hasOwn -> Boolean; MMegamorphicHasProp
  WJH_PARSEINT = 49,         // scratch[0]=string(Value),[1]=radix(Int32 Value) -> Value; MNumberParseInt
  WJH_OBJTOITER = 50,        // scratch[0]=obj(boxed Object) -> for-of iterator (Object); MObjectToIterator
  WJH_ISCALLABLE = 51,       // scratch[0]=value -> Boolean; MIsCallable
  WJH_STRTOINDEX = 52,       // scratch[0]=string(Value) -> Int32 (array index or -1); MGuardStringToIndex
  WJH_STREQATOM = 53,        // scratch[0]=str(Value),[1]=atom(StringValue) -> Boolean (pointer- or char-equal); MGuardSpecificAtom
  WJH_HOMEPROTO = 54,        // scratch[0]=homeObject(Object) -> its [[Prototype]] (Object); MHomeObjectSuperBase
  WJH_SETARRLEN = 55,        // scratch[0]=array(Object),[1]=rhs(Value), site=strict -> SetArrayLength; MCallSetArrayLength
  WJH_INSHAPELIST = 56,      // scratch[0]=obj(Object),[1]=shapeList(Object) -> Boolean (obj->shape() in the list); MGuardMultipleShapes
  WJH_MATHFN = 57,           // scratch[0]=x(Value/number), site=UnaryMathFunction -> Double; MMathFunction (sin/cos/tan/log/exp/...)
  WJH_NEWARRDYN = 58,        // scratch[0]=length(Int32),[1]=templateObject(ArrayObject Value) -> Array; MNewArrayDynamicLength
  WJH_INT32TOSTRINGBASE = 59,// scratch[0]=input(Int32 Value),[1]=base(Int32 Value), site=lowerCase -> String; MInt32ToStringWithBase (n.toString(base); cold/uninlinable: digit conv + string alloc)
  WJH_STRINGTRIMINDEX = 60,  // scratch[0]=str(String Value), site 0=start: jit::StringTrimStartIndex; site 1=end: scratch[1]=start(Int32), jit::StringTrimEndIndex -> Int32 (leaf, no GC); MStringTrim{Start,End}Index
  WJH_STRINGTONUMBER = 61,   // scratch[0]=str(String Value); site 0: GetInt32FromStringPure -> Int32; site 1: StringToNumberPure -> Double; return 1.0 (deopt) on parse-fail; MGuardStringTo{Int32,Double} (pure, no GC)
  WJH_STRINGREPLACE = 62,    // scratch[0]=string,[1]=pattern,[2]=replacement (all String Values), site=isFlatReplacement -> String; MStringReplace (jit::StringReplace / StringFlatReplaceString)
  WJH_STRINGSPLIT = 63,      // scratch[0]=string,[1]=separator (String Values) -> Array (Object); MStringSplit (js::StringSplitString, limit=INT32_MAX)
  WJH_OBJECTKEYS = 64,       // scratch[0]=obj(Object Value) -> Object.keys array (Object); MObjectKeys (jit::ObjectKeys); cold/uninlinable (enumerate + alloc)
  WJH_DELELEM = 65,          // scratch[0]=obj(Value),[1]=index(Value), site=strict -> Boolean; MDeleteElement (js::DelElemOperation<strict>)
  WJH_ISCONSTRUCTOR = 66,    // scratch[0]=obj(Object Value) -> Boolean; MIsConstructor (jit::ObjectIsConstructor, leaf)
  WJH_RANDOM = 67,           // (no operands) -> Double in [0,1); MRandom (realm XorShift128PlusRNG.nextDouble())
  WJH_REGEXPHASCAPS = 68,    // scratch[0]=regexp(Object),[1]=input(String) -> Boolean; MRegExpHasCaptureGroups (js::RegExpHasCaptureGroups)
  WJH_DELPROP = 69,          // scratch[0]=value(Value),[1]=name(PropertyName as StringValue), site=strict -> Boolean; MDeleteProperty (js::DelPropOperation<strict>)
  WJH_NEWTYPEDARRDYN = 70,   // scratch[0]=length(Int32),[1]=templateObject(Value) -> TypedArray (Object); MNewTypedArrayDynamicLength (js::NewTypedArrayWithTemplateAndLength)
  WJH_OVERRECURSED = 71,     // (no operands) sets the pending too-much-recursion exception (js::ReportOverRecursed); the JIT recursion guard calls this then propagates via EmitExceptionExit (CATCHABLE, like PBL)
  WJH_TYPEDARRELEMSIZE = 72, // scratch[0]=typedArray(Object Value) -> Int32 bytesPerElement (1/2/4/8); MTypedArrayElementSize (cold)
  WJH_TONUMBERINT32 = 73,    // scratch[0]=input(boxed Value) -> Int32 (JS::ToInt32 = ToNumber->ToInt32); MToNumberInt32; return 1.0 on throw
  WJH_ITEREND = 74,          // scratch[0]=iterator(Object Value), effect -> js::CloseIterator (for-in NativeIterator unlink); MIteratorEnd
  WJH_GETITER = 75,          // scratch[0]=value(boxed) -> js::ValueToIterator -> PropertyIteratorObject (Object); MGetIteratorCache (for-in setup, JSOp::Iter); GC+throw
  WJH_ITERMORE = 76,         // scratch[0]=iterator(Object) -> NativeIterator::nextIteratedValueAndAdvance -> next key (String) or MagicValue(JS_NO_ITER_VALUE); MIteratorMore; no GC/throw
  WJH_OBJKEYSITER = 77,      // scratch[0]=iterator(Object) -> js::ObjectKeysFromIterator -> array of own keys (Object); MObjectKeysFromIterator; GC+throw
  WJH_CHARCASE = 78,         // scratch[0]=code(boxed Int32), site=case(0 lower/1 upper) -> 1-char String (unicode case-fold); MCharCodeConvertCase; GC(alloc)
  WJH_APPLYARRAY = 79,       // scratch[0]=callee(Value),[1]=thisArg(Value),[2]=argsArray(Object) -> JS::Call spreading the array's dense elements -> result Value; MApplyArray (fn.apply); GC+throw
  WJH_CHECKPRIVFIELD = 80,   // scratch[0]=value(Value),[1]=idval(private-name Symbol),[2]=bytecode pc(i32) -> js::CheckPrivateFieldOperation -> Boolean; MCheckPrivateFieldCache (#x in obj / obj.#x brand check); GC+throw
  WJH_NEWBOUNDFN = 81,       // scratch[0]=template(BoundFunctionObject) -> js::BoundFunctionObject::createWithTemplate -> Object; MNewBoundFunction (fn.bind fast path); GC(alloc)
  WJH_ITERLENGTH = 82,       // scratch[0]=iterator(PropertyIteratorObject) -> NativeIterator ownPropertyCount (Int32); MIteratorLength; no GC/throw
  WJH_GFADBG = 83,           // DEBUG (GECKO_WJ_GFADBG): scratch[0]=param[i] value, site=i -> prints the decoded value, returns it; diagnoses the GetFrameArgument-beyond-formals stale-param bug
  WJH_ITERHASINDICES = 84,   // scratch[0]=object,[1]=iterator -> Boolean (NativeIterator IndicesAvailable flag && object shape matches iter's stored objShape); MIteratorHasIndices; no GC/throw
  WJH_LOADSLOTBYITER = 85,   // scratch[0]=object(NativeObject),[1]=iterator -> Value at the iterator's current PropertyIndex (for-in obj[key] fast read); MLoadSlotByIteratorIndex; no GC/throw
  WJH_BINDFUNCTION = 86,     // scratch[0]=target(Object),[1..argc]=bound args (args[0]=boundThis), site=argc -> BoundFunctionObject::functionBindImpl -> Object; MBindFunction (fn.bind); GC(alloc)+throw
  WJH_LOADITERELEM = 87,     // scratch[0]=iterator(PropertyIteratorObject),[1]=index(Int32) -> NativeIterator propertiesBegin()[index].asString() (String); MLoadIteratorElement (Object.keys(o)[i] ScalarReplacement); no GC/throw
  WJH_GETDOMPROP = 88,       // scratch[0]=DOM object(guarded),[1]=JSJitInfo*(baked i32 ptr) -> js::jit::CallDOMGetter -> Value; MGetDOMProperty (DOM getter JSJitGetterOp ABI); GC+throw
  WJH_REGEXPMATCHER = 89,    // scratch[0]=regexp(Object),[1]=string(String),[2]=lastIndex(Int32) -> Value (match array or null); MRegExpMatcher (js::RegExpMatcherRaw, nullptr matches); GC+throw
  WJH_REGEXPSEARCHER = 90,   // scratch[0]=regexp(Object),[1]=string(String),[2]=lastIndex(Int32) -> Int32 (match start or -1); MRegExpSearcher (js::RegExpSearcherRaw, nullptr matches); sets cx->regExpSearcherLastLimit; GC+throw
  WJH_REGEXPSEARCHERLASTLIMIT = 91, // (no operands) -> Int32 (cx->regExpSearcherLastLimit); MRegExpSearcherLastLimit; reads the limit set by the preceding WJH_REGEXPSEARCHER; no GC/throw
  WJH_REGEXPEXECMATCH = 92,  // scratch[0]=regexp(RegExpObject),[1]=string(String) -> Value (match array or null); MRegExpExecMatch (js::RegExpBuiltinExecMatchFromJit, nullptr matches); GC+throw
  WJH_REGEXPEXECTEST = 93,   // scratch[0]=regexp(RegExpObject),[1]=string(String) -> Boolean; MRegExpExecTest (js::RegExpBuiltinExecTestFromJit); GC+throw
  WJH_GETFIRSTDOLLARINDEX = 94, // scratch[0]=string(String) -> Int32 (first '$' index or -1); MGetFirstDollarIndex (js::GetFirstDollarIndexRaw); no throw
  WJH_INSHAPELISTOFFSET = 95,   // scratch[0]=obj(Object),[1]=shapeList(Object) -> Int32 (matched shape's slot offset, or -1 -> deopt); MGuardMultipleShapesToOffset. List is [shape,offset] pairs.
  WJH_NEWTYPEDARRFROMBUF = 96,  // scratch[0]=buffer(Object),[1]=byteOffset(Value),[2]=length(Value),[3]=template(Object) -> Object; MNewTypedArrayFromArrayBuffer (js::NewTypedArrayWithTemplateAndBuffer). May throw.
  WJH_NEWARGUMENTS = 97,        // scratch[0..argc-1]=actuals(Value),[8]=callee(Object),[9]=scopeChain(Object),[10]=argc(Int32) -> Object; MCreateArgumentsObject (ArgumentsObject::createForWasmJit).
  WJH_OBJCLASSTOSTRING = 98,    // scratch[0]=obj(Object) -> String (js::ObjectClassToString "[object X]") or 0-ptr sentinel (interesting @@toStringTag -> emitted code deopts). No throw.
  WJH_OPTGETITER = 99,          // scratch[0]=value(Value) -> Boolean; MOptimizeGetIteratorCache (js::OptimizeGetIterator: can this value use the fast for-of/spread iterator?). No throw.
  WJH_OPTSPREADCALL = 100,      // scratch[0]=value(Value) -> Value (the array to spread, or undefined if not optimizable); MOptimizeSpreadCallCache (js::OptimizeSpreadCall). May GC/throw (arguments-spread path).
  WJH_REST = 101,               // scratch[0..7]=actuals(boxed), [8]=numFormals, [9]=argc(raw) -> Object; MRest rest-parameter array via js::jit::InitRestParameter (nullptr arrRes -> NewDenseCopiedArray). May GC/OOM.
  WJH_GETSPARSEELEM = 102,      // scratch[0]=obj(Object), [1]=index(boxed Int32) -> Value; MCallGetSparseElement via js::GetSparseElementHelper. May GC/throw.
  WJH_NEWNAMEDLAMBDA = 103,     // scratch[0]=callee(function, boxed) -> Object; MNewNamedLambdaObject via js::NamedLambdaObject::createWithoutEnclosing (enclosing set by a later store). May GC/OOM.
  WJH_NEWVARENV = 104,          // gWJVarScope=VarScope* (raw) -> VarEnvironmentObject::createWithoutEnclosing (Object); MNewVarEnvironmentObject. May GC/OOM.
  WJH_HASOWN = 105,             // scratch[0]=value/obj(boxed), [1]=idval/key(boxed) -> Boolean; MHasOwnCache via js::HasOwnProperty (`id in obj` / hasOwnProperty). May GC/throw (ToPropertyKey/ToObject).
  WJH_NEWMAP = 106,             // no operands -> Object; MNewMapObject via js::MapObject::create(cx, nullptr) (default Map.prototype, mirrors Ion OOL). May GC/OOM.
  WJH_NEWSET = 107,             // no operands -> Object; MNewSetObject via js::SetObject::create(cx, nullptr) (default Set.prototype, mirrors Ion OOL). May GC/OOM.
  WJH_NEWITERATOR = 108,        // no operands, site=type(0=Array/1=String/2=RegExpString) -> Object; MNewIterator via js::NewArrayIterator/NewStringIterator/NewRegExpStringIterator. for-of/spread iterator. May GC/OOM.
  WJH_CODEPOINTAT = 109,        // scratch[0]=string(boxed), [1]=index(boxed Int32) -> Int32; MCodePointAt via js::jit::CodePointAt (full code point, combines surrogate pairs). String iteration. May GC (rope flatten).
  WJH_FROMCODEPOINT = 110,      // scratch[0]=codePoint(boxed Int32, guaranteed 0..0x10FFFF by a JIT deopt-guard) -> String; MFromCodePoint via js::StringFromCodePoint. String.fromCodePoint / string iteration. May GC/OOM.
  WJH_ISPACKEDARRAY = 111,      // scratch[0]=object(boxed) -> Boolean; MIsPackedArray via js::IsPackedArray (pure read: ArrayObject && initLen==len && packed). Array fast-path checks (spread/Reflect/from). No GC/throw.
  WJH_GETPROPSUPER = 112,       // scratch[0]=object/super-base(Object), [1]=receiver/this(boxed), [2]=idval/key(boxed) -> Value; MGetPropSuperCache via ToPropertyKey + js::GetProperty(obj,receiver,id) (super.prop / super[expr]). May GC/throw.
  WJH_ASYNCRESOLVE = 113,       // scratch[0]=generator(AsyncFunctionGeneratorObject, boxed Object), [1]=value(boxed) -> Object; MAsyncResolve via js::AsyncFunctionResolve (resolves an async fn's promise at its return/resolve point). May GC/OOM. Async fns bailed whole-fn to PBL here (reddit.com).
  WJH_CANSKIPAWAIT = 114,       // scratch[0]=value(boxed) -> Boolean; MCanSkipAwait via js::CanSkipAwait (await-skip optimization: true when the awaited value isn't a thenable). May GC (getter on .then/.constructor).
  WJH_EXTRACTAWAITVALUE = 115,  // scratch[0]=value(boxed) -> Value; MMaybeExtractAwaitValue's can-skip branch via js::ExtractAwaitValue (unwrap the already-resolved await value). May GC.
};
// MNewLexicalEnvironmentObject: the LexicalScope* baked from the template object.
extern uint32_t gWJLexScope;
// MNewVarEnvironmentObject: the VarScope* baked from the template object.
extern uint32_t gWJVarScope;

// OSR cheap-resume (GECKO_WJ_OSR): runtime sets these to re-enter a dispatch-loop fn
// at a loop-head block with locals loaded from gWJResumeVals. Cleared by the prologue.
extern uint32_t gWJOsrActive;
extern uint32_t gWJOsrBlock;

// Allocation-helper staging (non-GC ints; the shape is in the traced shape pool).
extern uint32_t gWJNewShapeSlot;  // gWJShapePool index of the template shape
extern uint32_t gWJNewAux;        // allocKind (NewPlainObject) or array length
extern uint32_t gWJNewHeap;       // gc::Heap (0=Default nursery)
extern uint32_t gWJNewObjScript;  // JSScript* (raw) for WJH_NEWOBJECT
extern uint32_t gWJNewObjPcOff;   // bytecode offset of the JSOp::NewObject

// Compile-time-baked address of the current zone's needs-marking-barrier flag
// (zone->addressOfNeedsMarkingBarrier()). The emitted pre-write-barrier fast path
// loads + tests this; it's 0 except during an incremental GC slice. Set by
// WJWarpCompile. Single-zone shell, so one address suffices.
extern uintptr_t gWJMarkBarrierAddr;
// Baked &storeBuffer.bufferWholeCell.last_ (single-zone shell). The inline whole-
// cell post-write barrier reads it to dedup consecutive same-cell stores (the
// `last_` check in StoreBuffer::WholeCellBuffer::put) WITHOUT a wjhelp boundary
// crossing -- splay does 1.43M such barriers, most redundant within a tree op.
extern uintptr_t gWJWholeCellLastAddr;

// Inline nursery bump-allocation params (baked in WJWarpCompile; read by the
// backend at emit time, emitted as immediates). gWJNurseryPosAddr = address of
// the zone's nursery position_ (the emitted code loads/bumps/stores it);
// gWJObjHeaderWord = NurseryCellHeader::MakeValue(catchAllAllocSite, Object) for
// the cell header. The rest (header size, currentEnd offset, thingSize,
// emptyObjectSlots/Elements) are compile-time constants the backend computes.
extern uintptr_t gWJNurseryPosAddr;
extern uintptr_t gWJObjHeaderWord;
extern uintptr_t gWJStringHeaderWord;

// Boxed ObjectValue bits of the realm's global lexical environment, refreshed
// each WJWarpCompile. MFunctionEnvironment bakes this for top-level-scoped
// functions (whose canonical JSFunction has no compile-time environment).
extern uint64_t gWJGlobalLexEnvVal;

// Raw pointer to the currently-executing JIT function's environment object. Set
// (to the runtime closure's environment) right before EVERY JIT invocation --
// the entry path (WasmJitRunCall's envChain) and the fast inter-JIT call_indirect
// (callee->environment()). MFunctionEnvironment loads this at function entry,
// giving the CORRECT runtime environment even for closures (unlike a baked one).
extern uint32_t gWJCurrentEnv;

// Set by WJEmitBody (reset at entry) to true iff the compiled function contains an
// alwaysBails block emitted as a deopt-at-entry (a cold/untyped Warp branch). Read
// by the deopt-storm valve: a storming function WITH an alwaysBails block deopts
// from that (cold-IC) block, which recompiling reproduces identically -- so it goes
// straight to PBL (Failed) instead of churning recompiles (deltablue wrong-value/
// crash). A storming function WITHOUT one (a stale monomorphic guard) still
// recompiles to specialize (crypto). Single-threaded synchronous compile.
extern bool gWJHadAlwaysBails;
// Per-compile deopt-site category counts (for the count-trigger discriminator): how
// many GuardShape-family (object/shape) deopt sites vs total. A SHAPE-dominated fn's
// count-trigger respec recompile is unsound (re-bakes a stale GC const -> acorn bug);
// a numeric/type-dominated fn's respec is the beneficial re-typing case (crypto/cdjs).
extern uint32_t gWJEmitShapeDeopts;
extern uint32_t gWJEmitTotalDeopts;
extern bool gWJForceMega;  // next compile: megamorphic property reads (post-storm)
extern bool gWJForceNumberArith;  // next compile: de-speculate Int32 arith/elem ICs (post-deopt)

// Synchronous main-thread tier-up compile cost (GECKO_WJ_COMPILESTAT).
extern uint64_t gWJCompileAttempts;  // total WJWarpCompile calls
extern uint64_t gWJCompileOK;        // compiles that succeeded (handle >= 0)
extern double gWJCompileMs;          // wall-ms across ALL attempts (incl bails)
extern double gWJCompileOKMs;        // wall-ms across successful compiles only
extern double gWJHostCompileMs;      // wall-ms in wasmhost_compile (V8 wasm compile)
extern double gWJHostInstMs;         // wall-ms in wasmhost_instantiate
extern uint64_t gWJEmitBytes;        // total emitted wasm bytes across compiles
extern int64_t gWJTraceVal;          // per-def value tracer staging (GECKO_WJ_VALUETRACE)
extern double gWJSnapshotMs;         // wall-ms in WarpOracle::createSnapshot
extern double gWJBuildMs;            // wall-ms in WarpBuilder::build
extern double gWJOptimizeMs;         // wall-ms in OptimizeMIR

// Compile bail-reason tracking (GECKO_WJ_CDBG): the most recent reason a function
// stayed in PBL plus its source line, printed by WJWarpCompile when WJEmitBody
// returns false. gWJBailReason points to a string literal (no ownership).
extern const char* gWJBailReason;
extern uint32_t gWJBailLine;
// Precise bail localization: the SOURCE LINE + bytecode offset + (inlinee) script
// filename of the specific MIR op whose emit returned false -- distinct from the
// COMPILED function's defining line (script->lineno()). Set by WJSetBailOp at the
// per-op emit entry; printed by the LOGBAIL audit so a bail points at the exact op.
extern uint32_t gWJBailOpLine;
extern uint32_t gWJBailOpOff;
extern const char* gWJBailOpFile;

// Resume state buffer (WASMJIT_REARCH_PLAN.md §4.1). On a guard miss the emitted
// code boxes the resume point's live values [this, args, locals, stack] into
// gWJResumeVals and calls wjhelp(WJH_RESUME), which reconstructs PBL frame(s) and
// continues. Traced by WJTraceRoots.
//
// MULTI-FRAME (inline bailout): a deopt inside an INLINED callee must rebuild the
// whole inline frame chain. gWJResumeNFrames is the chain length; index 0 is the
// INNERMOST (deopt) frame, the last is the OUTERMOST (compiled function). Each
// frame's per-frame state is in the arrays below at index i; its boxed values
// start at gWJResumeVals[gWJResumeValsOff[i]] laid out [this, args.., locals..,
// stack..]. WJH_RESUME runs them innermost->outermost, threading each frame's
// return into the next outer frame's call-result stack slot. nFrames==1 is the
// ordinary (non-inlined) single-frame resume.
static constexpr uint32_t kWJMaxResumeFrames = 16;
extern uint64_t gWJResumeVals[];
extern uint32_t gWJResumeNFrames;
extern uint32_t gWJResumePc[];          // resume bytecode offset, per frame
extern uint32_t gWJResumeStackDepth[];  // expr-stack depth, per frame
extern uint32_t gWJResumeScriptPtr[];   // JSScript* (i32), per frame
extern uint32_t gWJResumeEnvPtr[];      // JSObject* env chain (i32), per frame
// The function's ENCLOSING (captured) environment for the resumed frame -- i.e.
// the closure's environment() = gWJCurrentEnv at entry (snapshotted to envLocal).
// WJH_RESUME passes it so the PBL prologue sets up the frame env from the CORRECT
// runtime enclosing scope, not the CANONICAL script->function()->environment()
// (wrong/null for a re-instantiated closure -> navier/earley crash). Per frame; 0
// = unknown (PBL falls back to func->environment()). Only the outermost frame
// (this compiled fn) has it; inlined frames keep 0.
extern uint32_t gWJResumeEnclosingEnv[];
// try/catch: emitted code sets this to 1 before a WJH_RESUME that is an in-try-region
// exception deopt (so PBL enters error-mode -> HandleException runs the catch).
extern uint32_t gWJResumeInError;
// Debug (GECKO_WJ_DEOPTHIST): per-MIR-op deopt counter. The emitted deopt path
// increments gWJDeoptByOp[curOp]; WJH_RESUME periodically prints the histogram so
// we can see WHICH guard kind deopts most (MIRDUMP is unreliable for nondeterm-
// inistically-compiled fns like navier's). Indexed by MDefinition::Opcode.
static constexpr uint32_t kWJNumOps = 700;
extern uint32_t gWJDeoptByOp[];
extern uint32_t gWJResumeNArgs[];       // per frame
extern uint32_t gWJResumeNLocals[];     // per frame
extern uint32_t gWJResumeValsOff[];     // start index into gWJResumeVals, per frame
extern uint32_t gWJResumeCalleeFn[];    // runtime callee fn* per frame (0 = canonical)
// Beyond-formal actuals for the deopt-resume: an arguments-using fn called with
// MORE actuals than formals must hand PBL the full actuals -- the resume point
// only carries formals, so resuming with argc=nargs made `arguments` empty/short
// (lodash overRest closure saw arguments.length==0). The deopt spill stores the
// staged actuals + the entry argc snapshot; WJH_RESUME grafts slots [nargs,argc)
// into the physical (outermost) frame's arg region. 0 = no graft. Traced while
// gWJResumeActive (boxed Values).
extern uint64_t gWJResumeActuals[];
extern uint32_t gWJResumeActualArgc;
extern uint32_t gWJExitFPLastSite;   // task #60 exitFP setter tracer
extern uint32_t gWJExitFPSetCount;
extern uint32_t gWJLastDeoptOp;         // MIR opcode of the most recent deopt site
extern const uint32_t gWJOpGuardGlobalGeneration;  // numeric MIR opcode (backend-exported)
// Call boundary: callee + argc for wjhelp(WJH_CALL).
extern uint64_t gWJCallCallee;
extern uint32_t gWJCallSiteLine[];  // DEBUG: caller script line per call site
const char* WJMirOpName(uint32_t op);  // MIR opcode -> name (for histograms in other TUs)
extern uint32_t gWJCallArgc;
// Constructing call: newTarget (boxed) for wjhelp(WJH_CREATETHIS isn't this --
// this is for the construct CALL itself). `this` is staged at gWJScratch[kWJThisSlot].
extern uint64_t gWJConstructNewTarget;
// Barriered slot store: wjhelp(WJH_SETSLOT) does gWJHelpObj->setSlot(slot, val).
extern uint32_t gWJHelpObj;
extern uint32_t gWJHelpSlot;
extern uint64_t gWJHelpVal;
extern uint64_t gWJElemHits;  // DEBUG: inline typed-array element-store IC hits

// Monomorphic call-site IC for direct wasm->wasm calls. Per call site: the cached
// callee object pointer and its shared-table index. A JIT'd caller compares the
// actual callee to gWJCallFn[site]; on a hit it `call_indirect`s the callee's
// wasm via the shared table (no C++/JS hop); on a miss it calls wjhelp(WJH_CALL),
// which runs the callee and fills the IC (only for compiled callees).
static constexpr uint32_t kWJCallSites = 16384;
// Polymorphic IC: up to kWJCallWays cached (callee, table-slot) pairs per site,
// so a polymorphic dispatch (e.g. richards' task.run over 4-5 task types) does a
// guarded wasm->wasm call_indirect per way instead of the slow wjhelp hop. Way 0
// is checked first, so monomorphic calls stay one compare. Arrays are
// [kWJCallSites * kWJCallWays], indexed [site * kWJCallWays + way].
static constexpr uint32_t kWJCallWays = 8;
extern uint32_t gWJCallFn[];      // cached callee obj ptr (0 = empty)
extern int32_t gWJCallTblIdx[];   // callee's shared-table index
// Next unused call-site id (process-global; assigned at compile).
uint32_t WJAllocCallSite();

// Per-construct-site monomorphic inline cache (GECKO_WJ_CTORINLINE). See
// WasmJitRuntime.cpp. Lets the backend inline `new X()` (alloc + ctor call) and
// skip the WJH_CONSTRUCT boundary for monomorphic construct sites.
static constexpr uint32_t kWJCtorSites = 16384;
extern uint32_t gWJCtorCallee[];  // cached ctor fn ptr (0 = empty)
extern uint32_t gWJCtorShape[];   // cached `this` SharedShape ptr
extern uint32_t gWJCtorSize[];    // total nursery cell size
extern uint32_t gWJCtorNfixed[];  // fixed slot count
extern int32_t gWJCtorTblIdx[];   // ctor table index (-1 = none)
extern uint32_t gWJCtorEnv[];     // ctor environment ptr
uint32_t WJAllocCtorSite();

// Inline property-load IC (for MegamorphicLoadSlot / named GetPropertyCache of
// own data properties). Mirrors the call IC: per-site cached (receiver shape,
// TaggedSlotOffset). The JIT'd code loads obj->shape(), compares to the cached
// shape; on a hit it loads the slot inline (no C++ hop), on a miss it calls
// wjhelp(WJH_PROPIC) which does a pure lookup, fills the IC, and returns the
// value. A Shape is immutable and uniquely determines an own data property's
// slot, so a shape-identity match is sufficient for correctness; the only hazard
// is shape-pointer REUSE after a compacting GC, handled by clearing this IC on
// major-GC marking in WJTraceRoots. Indexed [site * kWJPropWays + way].
static constexpr uint32_t kWJPropSites = 16384;
static constexpr uint32_t kWJPropWays = 4;
// gWJPropOff sentinel meaning "this shape -> property is MISSING, yield undefined"
// (no slot load). A real TaggedSlotOffset is small; 0xFFFFFFFF can't collide.
static constexpr uint32_t kWJPropMissingSentinel = 0xFFFFFFFFu;
extern uint32_t gWJPropShape[];   // cached receiver Shape* (0 = empty)
extern uint32_t gWJPropOff[];     // cached TaggedSlotOffset bits ((off<<1)|isFixed)
// STORE-IC way key: the atom JSString* the way was filled for (0 = empty). The
// store IC's hit path guards (shape AND key): a SetPropertyCache site with a
// DYNAMIC key (`obj[k] = v`) sees many keys for one receiver shape, so a
// shape-only match would store into the OTHER key's cached slot (babylon
// Node.__clone corrupted every cloned AST node this way). Traced (atoms are GC
// cells) so the pointer compare stays current across a moving GC.
extern uint32_t gWJPropWayKey[];
extern uint32_t gWJAddOldShape[];  // ADD-IC: pool addr of pre-add shape (0 = unset)
extern uint32_t gWJAddNewShape[];  // ADD-IC: pool addr of post-add shape
extern uint32_t gWJAddOff[];       // ADD-IC: added prop's fixed-slot byte offset
extern uint32_t gWJAddKey[];       // ADD-IC: the added key's atom (guarded like
                                   // gWJPropWayKey; one dynamic site adds MANY keys)
extern uint32_t gWJPropHolder[];  // proto-read cache: holder object* (0 = OWN-property way;
                                  // nonzero = load the slot from this proto holder instead
                                  // of the receiver). Validated by receiver shape match alone
                                  // (shape encodes proto identity + no own-shadow). Traced.
uint32_t WJAllocPropSite();
uint32_t WJAllocPropSiteKeyed(uint64_t key);  // reuse a site for the same read
// Per-site baked PropertyKey for the WJH_PROPIC fill helper (the property name).
extern uint64_t gWJPropKey[];     // jsid raw bits per site
extern uint8_t gWJPropStrict[];   // per-site strict flag (set-prop fallback)
// Staging for the prop-IC miss helper: obj (boxed) -> gWJScratch[0]; result in
// gWJScratch[kWJResultSlot]. site passed as wjhelp's 2nd arg.
void WJClearPropIC();

// GetName IC (MGetNameCache). A global/lexical name resolves to a data-property
// slot on a REALM-SINGLETON holder (the global object or global lexical env),
// stable for the whole run -- so we cache the holder pointer (traced + relocated
// by WJTraceRoots, GC-current), its shape (validation), and the tagged slot
// offset, per site. The fast path loads holder->slot directly (the VALUE may
// mutate, e.g. deltablue's `planner`, but the LOCATION is fixed); on holder==0
// (uncached) or shape mismatch it calls wjhelp(WJH_GETNAME, site) which resolves,
// fills the cache (only for global-singleton holders), and returns the value.
static constexpr uint32_t kWJNameSites = 8192;
extern uintptr_t gWJNameHolder[];  // traced holder NativeObject* (0 = uncached)
extern uint32_t gWJNameShape[];    // cached holder Shape* (wasm32: uint32 == ptr)
extern uint32_t gWJNameOff[];      // cached TaggedSlotOffset bits ((off<<1)|isFixed)
extern uint64_t gWJNameKey[];      // baked StringValue(name) bits per site
uint32_t WJAllocNameSite();

// GC-root shadow stack. The value-per-local backend holds object pointers in
// wasm locals, invisible to the GC. Before a call (which can trigger a moving
// minor GC) the emitted code spills its live object/Value locals here, bumps
// gWJRootSP, makes the call, restores gWJRootSP, then reloads the (GC-updated)
// values. WJTraceRoots traces [0, gWJRootSP). A stack discipline handles nested
// JIT->JIT calls. Defined in WasmJitRuntime.cpp.
static constexpr uint32_t kWJCallRootsSize = 1048576;
extern uint64_t gWJCallRoots[];
extern uint32_t gWJRootSP;
extern int32_t gWJJitDepth;  // JIT recursion depth (deopt-to-PBL past the limit)

// Scratch-buffer layout (shared with WasmJitRuntime.cpp's gWJScratch + the host
// call marshalling). Sized for nargs<=64 with slack.
// Register-argument call ABI. JIT'd "main" functions take wasm params
// (f64 scratch-base, i64 this, i64 arg0..arg[kWJMaxArgs-1]) so args pass in
// registers instead of round-tripping through the gWJScratch buffer. A per-
// module trampoline (f64 scratch-base)->f64 reads scratch and calls main, for
// the host-entry path (the JS<->wasm boundary can't carry i64 safely). Functions
// with more than kWJMaxArgs actual args fall back (stay PBL).
static constexpr uint32_t kWJMaxArgs = 8;

// EHABI rewrite (EHABI_REWRITE_PLAN.md): when true, the main wasm function ABI is
// `(f64 sb, i64 callee, i64 this, i64 args...) -> i64 result` with exceptions carried by
// native wasm EH (try/catch_all/throw + a $wjexn tag) instead of the legacy
// `(f64 sb, i64 this, args...) -> (f64 flag, i64 result)` return-flag ABI. Staged: keep
// false (old ABI builds + runs) until the new path is complete + validated, then flip.
static constexpr bool kWJEHABI = false;
// wasm param indices under each ABI (callee shifts this/args by 1 under EHABI).
static constexpr uint32_t kWJThisParam = kWJEHABI ? 2 : 1;
static constexpr uint32_t kWJArg0Param = kWJEHABI ? 3 : 2;
static constexpr uint32_t kWJCalleeParam = 1;  // EHABI only

static constexpr uint32_t kWJResultSlot = 64;
static constexpr uint32_t kWJThisSlot = 65;
// Traced scratch slots (within [0..kWJThisSlot] so WJTraceRoots relocates them
// on a moving GC) used to stage the construct callee + newTarget across the
// WJH_CONSTRUCT call window. Must be > kWJMaxArgs and not collide with
// result/this. The untraced gWJCallCallee/gWJConstructNewTarget globals were a
// stale-pointer source under nursery compaction.
static constexpr uint32_t kWJCalleeSlot = 62;
static constexpr uint32_t kWJNewTargetSlot = 63;
static constexpr uint32_t kWJScratchSlots = 72;
static constexpr uint32_t kWJResultOff = kWJResultSlot * 8;
static constexpr uint32_t kWJThisOff = kWJThisSlot * 8;

}  // namespace wasm
}  // namespace js

#endif  // wasm_WasmJitBackend_h
