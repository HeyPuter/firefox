/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include "RenderHostGpuTextureHost.h"

#include "mozilla/webrender/RenderThread.h"
#include "mozilla/webrender/WebRenderTypes.h"

#ifdef __EMSCRIPTEN__
#  include <emscripten.h>
#endif

#ifdef __EMSCRIPTEN__
extern "C" {
// The pthread id of WebRender's Renderer thread (where its GL context lives).
// Set by the Renderer thread at compositor-context creation; read by the main-
// thread bridge (hostimg-bridge.js) to find the worker to transfer VideoFrames
// to for GL upload.
static int g_hostimg_renderer_tid = 0;
int hostimg_renderer_tid() { return g_hostimg_renderer_tid; }
void hostimg_set_renderer_tid(int aTid) { g_hostimg_renderer_tid = aTid; }
}
#endif

namespace mozilla {
namespace wr {

RenderHostGpuTextureHost::RenderHostGpuTextureHost(uint64_t aExternalId,
                                                   gfx::IntSize aSize)
    : mExternalId(aExternalId), mSize(aSize) {}

RenderHostGpuTextureHost::~RenderHostGpuTextureHost() {
#ifdef __EMSCRIPTEN__
  // Drop the host-side GL texture + map entry when WebRender is done with the
  // external image (the host owns the GL object).
  uint32_t lo = uint32_t(mExternalId & 0xffffffff);
  uint32_t hi = uint32_t(mExternalId >> 32);
  // clang-format off
  EM_ASM({
    var id = $1 * 4294967296 + ($0 >>> 0);
    var m = globalThis.geckoHostImgTex;
    if (!m) { return; }
    var e = m.get(id);
    if (!e) { return; }
    try {
      if (typeof GL !== 'undefined' && GL.textures && e.tex) {
        var t = GL.textures[e.tex];
        if (t && typeof GLctx !== 'undefined') { GLctx.deleteTexture(t); }
        GL.textures[e.tex] = null;
      }
    } catch (_) {}
    m.delete(id);
  }, (int)lo, (int)hi);
  // clang-format on
#endif
}

wr::WrExternalImage RenderHostGpuTextureHost::Lock(uint8_t aChannelIndex,
                                                   gl::GLContext* aGL) {
#ifdef __EMSCRIPTEN__
  uint32_t lo = uint32_t(mExternalId & 0xffffffff);
  uint32_t hi = uint32_t(mExternalId >> 32);
  // Look up the integer GL texture id that the upload handler stashed for this
  // external image. Runs on the Renderer thread -- same GL.textures table that
  // WebRender's integer-handle GL calls use.
  // clang-format off
  int handle = EM_ASM_INT({
    var id = $1 * 4294967296 + ($0 >>> 0);
    var m = globalThis.geckoHostImgTex;
    var e = m ? m.get(id) : null;
    return (e && e.ready) ? e.tex : 0;
  }, (int)lo, (int)hi);
  // clang-format on
  if (handle <= 0) {
    return wr::InvalidToWrExternalImage();
  }
  return wr::NativeTextureToWrExternalImage(uint32_t(handle), 0, 0,
                                            float(mSize.width),
                                            float(mSize.height));
#else
  return wr::InvalidToWrExternalImage();
#endif
}

void RenderHostGpuTextureHost::Unlock() {}

}  // namespace wr
}  // namespace mozilla
