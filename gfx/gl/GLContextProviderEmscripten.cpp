/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

// GLContextProvider for the wasm32-emscripten (headless) build. Gecko has no
// platform GL provider here (gl_provider="Null", EGL can't dlopen libEGL.so),
// so WebRender's RenderCompositorOGL had no GL context and the engine fell back
// to software RenderDocument. This provider creates a WebGL2 (GLES3) context via
// emscripten's html5_webgl API, backed by the page <canvas>, so WebRender can
// composite on the GPU and present directly to the canvas.
//
// Threading: WebRender creates this on its "Renderer" thread, which does NOT own
// the page canvas's OffscreenCanvas. We therefore use a context proxied to the
// main browser thread (proxyContextToMainThread=ALWAYS), which works from any
// pthread. GL commands run on the main thread; presentation is implicit when
// that thread yields. (A faster path transferring the OffscreenCanvas to the
// Renderer thread is a later optimization.) See gecko-wasm-webgl-canvas notes.

#include "GLContextProvider.h"
#include "GLContext.h"
#include "GLLibraryLoader.h"
#include "mozilla/widget/CompositorWidget.h"
#include "nsString.h"

#include <emscripten.h>
#include <emscripten/html5.h>
#include <emscripten/html5_webgl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <pthread.h>

// The page canvas the embedder renders into (embed-xul/index.html: <canvas id="screen">).
#define MOZ_EMSCRIPTEN_GL_CANVAS "#screen"

namespace mozilla {
namespace gl {

// Canvas passthrough (flag GECKO_GL_PASSTHROUGH): when content requests a
// WebGL/WebGL2 canvas, Gecko's WebGLContext asks GLContextProvider::CreateHeadless
// for an offscreen GL context. We give it a REAL host WebGL2 context on its own
// hidden host <canvas> (the compositor owns "#screen", so content can't share it).
// Content GL commands pass through emscripten to that host context; the result is
// read back to the compositor via SurfaceFactory_Basic (glReadPixels) and uploaded
// by WebRender. Off by default (content WebGL then simply fails to create, as
// before). See gecko-wasm canvas-passthrough notes.
extern "C" __attribute__((weak)) int gecko_gl_passthrough_enabled();
static bool ContentPassthroughEnabled() {
  // The flag is captured on the app thread (embedder main(), where getenv sees the
  // env) and exposed via this weak symbol; getenv() here on the content WebGL
  // worker thread is unreliable (ENV not propagated to all workers).
  return gecko_gl_passthrough_enabled && gecko_gl_passthrough_enabled() != 0;
}

// JSPI yield (gecko.js/build/gl-present.js, marked __async): suspends the calling
// Renderer thread for one macrotask so its worker event loop runs, which is when the
// browser implicit-presents the OffscreenCanvas to its #screen placeholder. Requires
// the final link's -sJSPI.
extern "C" void gl_present_yield(void);

// Host-GPU image passthrough: when the compositor GL context is created on the
// Renderer thread, record this thread's id + install the JS handler that uploads
// host-decoded VideoFrames into this context (see hostimg-bridge.js,
// RenderHostGpuTextureHost). Both are no-ops unless GECKO_IMG_PASSTHROUGH=gpu.
extern "C" {
void hostimg_set_renderer_tid(int aTid);
void hostimg_renderer_install(void);
}

// emscripten_webgl_get_proc_address is provided by emscripten's GL JS library at the
// final link, so it is undefined in this relocatable libxul.so. Taking its address
// directly emits a table-index relocation against an undefined symbol, which newer
// wasm-ld (emscripten 6.0.1) rejects; a CALL to it is a deferred relocation that is
// fine. So route the loader through this locally-defined shim.
static void* EmscriptenGLGetProcAddress(const char* name) {
  return emscripten_webgl_get_proc_address(name);
}

class GLContextEmscripten final : public GLContext {
 public:
  MOZ_DECLARE_REFCOUNTED_VIRTUAL_TYPENAME(GLContextEmscripten, override)

  // Shared creation. `target` is an emscripten canvas selector. `present` =
  // compositor context (explicit-swap offscreen back buffer, blitted to the
  // visible canvas on commit_frame); !present = content/offscreen context (no
  // swap -- WebGLContext drives its own framebuffer and we read it back).
  static RefPtr<GLContextEmscripten> CreateImpl(const GLContextDesc& desc,
                                                const char* target, bool present,
                                                nsACString* const out_failureId) {
    EmscriptenWebGLContextAttributes attrs;
    emscripten_webgl_init_context_attributes(&attrs);
    attrs.majorVersion = 2;  // WebGL2 / GLES3 (WebRender requires GLES3)
    attrs.minorVersion = 0;
    attrs.alpha = EM_TRUE;
    attrs.depth = EM_TRUE;
    attrs.stencil = EM_TRUE;
    attrs.antialias = EM_FALSE;
    attrs.premultipliedAlpha = EM_TRUE;
    attrs.preserveDrawingBuffer = EM_TRUE;
    attrs.failIfMajorPerformanceCaveat = EM_FALSE;
    attrs.enableExtensionsByDefault = EM_TRUE;
    // Compositor: present explicitly; with a proxied context (no OffscreenCanvas)
    // emscripten emulates the swap by blitting an offscreen back buffer to the
    // default framebuffer on emscripten_webgl_commit_frame(). This back buffer is
    // REQUIRED for presentation -- implicit present does NOT work for a proxied
    // context, so without it the canvas stays black. Content contexts render to
    // their own framebuffer (read back for compositing), so they don't swap.
    attrs.explicitSwapControl = present ? EM_TRUE : EM_FALSE;
    // No offscreen back buffer: when #screen is a real (transferred) OffscreenCanvas
    // on this thread, WebRender renders directly to its default framebuffer and the
    // implicit swap presents it to the visible canvas (commit_frame's back-buffer blit
    // would target a redundant FBO that never reaches the OffscreenCanvas). If we
    // instead fall back to proxying, emscripten proxies without the back buffer too
    // (OffscreenCanvas support is built in).
    attrs.renderViaOffscreenBackBuffer = EM_FALSE;
    // PROXY_FALLBACK (not ALWAYS): if the page canvas (#screen) was transferred to
    // THIS (Renderer) thread as an OffscreenCanvas (see RenderThread::Start + NSPR
    // _pr_emscripten_next_canvas), create the context LOCAL here -- no per-GL-call
    // proxy to the main thread. If it wasn't transferred (transfer raced, or a
    // content/offscreen context), fall back to proxying so rendering still works.
    attrs.proxyContextToMainThread = EMSCRIPTEN_WEBGL_CONTEXT_PROXY_FALLBACK;

    EMSCRIPTEN_WEBGL_CONTEXT_HANDLE ctx =
        emscripten_webgl_create_context(target, &attrs);
    if (ctx <= 0) {
      if (out_failureId) {
        *out_failureId = "FEATURE_FAILURE_EMSCRIPTEN_WEBGL_CREATE"_ns;
      }
      NS_WARNING("GLContextEmscripten: emscripten_webgl_create_context failed");
      return nullptr;
    }

    RefPtr<GLContextEmscripten> gl = new GLContextEmscripten(desc, ctx, present);
    if (!gl->Init()) {
      if (out_failureId) {
        *out_failureId = "FEATURE_FAILURE_EMSCRIPTEN_WEBGL_INIT"_ns;
      }
      return nullptr;
    }

    // Host-GPU image passthrough: this is the compositor context, created on the
    // Renderer thread (#screen transferred here). Register the thread + install
    // the VideoFrame upload handler so host-decoded images can be composited as
    // WebRender external textures with no CPU readback.
    const char* imgPass = getenv("GECKO_IMG_PASSTHROUGH");
    if (present && imgPass && !strcmp(imgPass, "gpu")) {
      gl->MakeCurrent();
      hostimg_set_renderer_tid(static_cast<int>(pthread_self()));
      hostimg_renderer_install();
    }
    return gl;
  }

  // Compositor context on the page's visible canvas ("#screen").
  static RefPtr<GLContextEmscripten> Create(const GLContextDesc& desc,
                                            nsACString* const out_failureId) {
    return CreateImpl(desc, MOZ_EMSCRIPTEN_GL_CANVAS, /* present */ true,
                      out_failureId);
  }

  // Content WebGL context (canvas passthrough). Created WORKER-LOCAL on the
  // calling (content) thread via a new OffscreenCanvas, NOT proxied to thread 0.
  // emscripten's GLctx/current-context is per-thread JS state; the compositor's
  // context is proxied to thread 0, so a content context registered on the content
  // thread uses THIS thread's own GLctx and can never clobber the compositor's
  // (which is what broke GPU-mode passthrough). It also runs without the per-call
  // proxy tax. The result is read back (CPU, SurfaceFactory_Basic) for compositing.
  static RefPtr<GLContextEmscripten> CreateContent(
      const GLContextDesc& desc, nsACString* const out_failureId) {
    // Honor the WebGL version the content asked for: WebGLContext sets PREFER_ES3 for
    // WebGL2 (GLES3) and leaves it off for WebGL1 (GLES2). Forcing WebGL2 broke real
    // WebGL1 content -- e.g. shaders that `#extension GL_EXT_frag_depth : require`,
    // which is a WebGL1 extension but core (and so not an exposable extension) in
    // WebGL2, so the require errors and the shader won't compile.
    int wantWebgl2 = bool(desc.flags & CreateContextFlags::PREFER_ES3) ? 1 : 0;
    // NOTE: no top-level commas inside EM_ASM_INT -- the C preprocessor splits
    // macro args on commas (only parens protect them, not braces), so the attrs
    // object is built with assignments rather than a comma-separated literal.
    EMSCRIPTEN_WEBGL_CONTEXT_HANDLE ctx =
        (EMSCRIPTEN_WEBGL_CONTEXT_HANDLE)EM_ASM_INT({
          try {
            var oc = new OffscreenCanvas(300, 150);
            var attrs = {};
            attrs.majorVersion = $0 ? 2 : 1;
            attrs.minorVersion = 0;
            attrs.alpha = true;
            attrs.depth = true;
            attrs.stencil = true;
            attrs.antialias = false;
            attrs.premultipliedAlpha = true;
            attrs.preserveDrawingBuffer = true;
            attrs.failIfMajorPerformanceCaveat = false;
            attrs.enableExtensionsByDefault = true;
            var glctx = oc.getContext($0 ? "webgl2" : "webgl", attrs);
            if (!glctx) return 0;
            // Register with emscripten's GL on THIS thread (owns it -> local, not
            // proxied). GL.registerContext stamps the current pthread as owner.
            return GL.registerContext(glctx, attrs);
          } catch (e) { return 0; }
        }, wantWebgl2);
    if (ctx <= 0) {
      if (out_failureId) {
        *out_failureId = "FEATURE_FAILURE_EMSCRIPTEN_OFFSCREENCANVAS"_ns;
      }
      NS_WARNING("GLContextEmscripten: content OffscreenCanvas context failed");
      return nullptr;
    }
    RefPtr<GLContextEmscripten> gl =
        new GLContextEmscripten(desc, ctx, /* present */ false);
    if (!gl->Init()) {
      if (out_failureId) {
        *out_failureId = "FEATURE_FAILURE_EMSCRIPTEN_WEBGL_INIT"_ns;
      }
      return nullptr;
    }
    return gl;
  }

  GLContextType GetContextType() const override {
    return GLContextType::Emscripten;
  }

  bool MakeCurrentImpl() const override {
    return emscripten_webgl_make_context_current(mContext) ==
           EMSCRIPTEN_RESULT_SUCCESS;
  }

  bool IsCurrentImpl() const override {
    return emscripten_webgl_get_current_context() == mContext;
  }

  Maybe<SymbolLoader> GetSymbolLoader() const override {
    // emscripten resolves GL entry points by name (requires
    // -sGL_ENABLE_GET_PROC_ADDRESS). Matches SymbolLoader's void*(const char*)
    // ctor; the base GLContext::InitImpl loads the whole symbol table through it.
    return Some(SymbolLoader(EmscriptenGLGetProcAddress));
  }

  bool IsDoubleBuffered() const override { return true; }

  bool SwapBuffers() override {
    if (!mPresent) return true;
    // This (Renderer) thread owns #screen's transferred OffscreenCanvas, and the
    // context has explicitSwapControl + no offscreen back buffer, so the browser
    // implicit-presents it to the #screen placeholder when this thread yields to its
    // worker event loop. A Gecko thread otherwise runs a blocking loop and never
    // yields, so yield here via JSPI: gl_present_yield suspends the wasm stack for one
    // macrotask (the present fires during it), then we resume. No
    // transferToImageBitmap / postMessage / #glout overlay -- #screen shows the frame.
    gl_present_yield();
    return true;
  }

  void GetWSIInfo(nsCString* const out) const override {
    out->AppendLiteral("emscripten WebGL2 (Renderer-thread OffscreenCanvas)");
  }

 private:
  GLContextEmscripten(const GLContextDesc& desc,
                      EMSCRIPTEN_WEBGL_CONTEXT_HANDLE ctx, bool present)
      : GLContext(desc, nullptr, false), mContext(ctx), mPresent(present) {}

  ~GLContextEmscripten() {
    MarkDestroyed();
    if (mContext > 0) {
      emscripten_webgl_destroy_context(mContext);
    }
  }

  EMSCRIPTEN_WEBGL_CONTEXT_HANDLE mContext;
  bool mPresent;
};

// -- Provider statics --------------------------------------------------------

already_AddRefed<GLContext> GLContextProviderEmscripten::CreateForCompositorWidget(
    widget::CompositorWidget* aCompositorWidget, bool /*aHardwareWebRender*/,
    bool /*aForceAccelerated*/) {
  GLContextDesc desc;
  desc.isOffscreen = false;
  nsCString failureId;
  RefPtr<GLContext> gl = GLContextEmscripten::Create(desc, &failureId);
  return gl.forget();
}

already_AddRefed<GLContext> GLContextProviderEmscripten::CreateHeadless(
    const GLContextCreateDesc& createDesc, nsACString* const out_failureId) {
  // Offscreen contexts here are content WebGL (canvas.getContext). Only honor them
  // when the canvas passthrough flag is set; otherwise return null (content WebGL
  // is unavailable) rather than fighting the compositor for "#screen".
  if (!ContentPassthroughEnabled()) {
    if (out_failureId) {
      *out_failureId = "FEATURE_FAILURE_EMSCRIPTEN_NO_PASSTHROUGH"_ns;
    }
    return nullptr;
  }
  GLContextDesc desc = {};
  static_cast<GLContextCreateDesc&>(desc) = createDesc;
  desc.isOffscreen = true;
  RefPtr<GLContext> gl = GLContextEmscripten::CreateContent(desc, out_failureId);
  return gl.forget();
}

GLContext* GLContextProviderEmscripten::GetGlobalContext() { return nullptr; }

void GLContextProviderEmscripten::Shutdown() {}

}  // namespace gl
}  // namespace mozilla
