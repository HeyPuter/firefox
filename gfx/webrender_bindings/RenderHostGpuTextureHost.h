/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef MOZILLA_GFX_RENDERHOSTGPUTEXTUREHOST_H
#define MOZILLA_GFX_RENDERHOSTGPUTEXTUREHOST_H

#include "RenderTextureHost.h"
#include "mozilla/gfx/Point.h"

namespace mozilla {
namespace wr {

// A WebRender external image backed by a GL texture the HOST browser uploaded.
// gecko.js/lib/hostimg-bridge.js decodes a content image with the host's
// WebCodecs ImageDecoder, transfers the VideoFrame to the Renderer thread, and
// uploads it (texImage2D) into WebRender's GL context -- registering the texture
// in emscripten's GL.textures table by integer id and stashing
// {externalId -> {tex:id, w, h}} in a Renderer-thread-global map. Lock() runs on
// the Renderer thread (same GL.textures table) and returns that integer id as a
// native WR texture handle, so the decoded image composites with NO CPU readback
// or texture-cache upload. See [[gecko-wasm-webcodecs-passthrough]].
class RenderHostGpuTextureHost final : public RenderTextureHost {
 public:
  RenderHostGpuTextureHost(uint64_t aExternalId, gfx::IntSize aSize);

  wr::WrExternalImage Lock(uint8_t aChannelIndex, gl::GLContext* aGL) override;
  void Unlock() override;
  size_t Bytes() override {
    return size_t(mSize.width) * mSize.height * 4;
  }
  gfx::SurfaceFormat GetFormat() const override {
    return gfx::SurfaceFormat::R8G8B8A8;
  }

 private:
  virtual ~RenderHostGpuTextureHost();

  const uint64_t mExternalId;
  const gfx::IntSize mSize;
};

}  // namespace wr
}  // namespace mozilla

#endif  // MOZILLA_GFX_RENDERHOSTGPUTEXTUREHOST_H
