/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include "nsHostProxyImageDecoder.h"

#include "RasterImage.h"
#include "SurfacePipeFactory.h"
#include "imgFrame.h"

#ifdef __EMSCRIPTEN__
#  include <emscripten/threading.h>
#  include <stdlib.h>
#  include "mozilla/layers/AsyncImagePipelineManager.h"
#  include "mozilla/webrender/WebRenderTypes.h"
#endif

using namespace mozilla::gfx;

namespace mozilla {
namespace image {

#ifdef __EMSCRIPTEN__
extern "C" {
// Implemented in gecko.js/lib/hostimg-bridge.js (__proxy:'sync', runs on the
// browser main thread). Kicks off an async host ImageDecoder decode of the
// encoded bytes; when it finishes it mallocs a packed BGRA buffer, writes the
// result into the control block, and Atomics.notify(IMG_STATUS) to wake us.
void hostimg_decode(int32_t* aCtrl, const char* aMime, const uint8_t* aData,
                    int aLen);

// Host-GPU path: decode on the host, transfer the VideoFrame to the Renderer
// thread, and upload it into WebRender's GL context as the external image with
// id (aIdLo|aIdHi<<32). Writes status/w/h into the control block + wakes us.
void hostimg_gpu_decode(int32_t* aCtrl, int aIdLo, int aIdHi, const char* aMime,
                        const uint8_t* aData, int aLen);
}

// Single-shot control block shared with the bridge (atomic int32 on the wasm
// heap). MUST match the layout in hostimg-bridge.js.
#  define IMG_STATUS 0  // 0 = pending, 1 = done, 2 = error (also the futex word)
#  define IMG_W 1
#  define IMG_H 2
#  define IMG_STRIDE 3       // bytes per row of the BGRA buffer
#  define IMG_PTR 4          // malloc'd BGRA buffer (we free it)
#  define IMG_FRAMES 5       // frameCount (informational; v1 is first-frame only)
#  define IMG_CTRL_INTS 6

static const char* MimeForType(DecoderType aType) {
  switch (aType) {
    case DecoderType::PNG:
      return "image/png";
    case DecoderType::JPEG:
    case DecoderType::JPEG_PDF:
      return "image/jpeg";
    case DecoderType::WEBP:
      return "image/webp";
    case DecoderType::AVIF:
      return "image/avif";
    case DecoderType::GIF:
      return "image/gif";
    default:
      return nullptr;
  }
}
#endif  // __EMSCRIPTEN__

nsHostProxyImageDecoder::nsHostProxyImageDecoder(RasterImage* aImage,
                                                 DecoderType aType, bool aGpu)
    : Decoder(aImage),
      mType(aType),
      mGpu(aGpu),
      mFormat(SurfaceFormat::OS_RGBA) {}

nsHostProxyImageDecoder::~nsHostProxyImageDecoder() = default;

LexerResult nsHostProxyImageDecoder::DoDecode(SourceBufferIterator& aIterator,
                                              IResumable* aOnResume) {
#ifndef __EMSCRIPTEN__
  return LexerResult(TerminalState::FAILURE);
#else
  // 1. Accumulate the entire encoded stream (mirrors nsAVIFDecoder).
  while (!mReadComplete) {
    SourceBufferIterator::State state =
        aIterator.AdvanceOrScheduleResume(SIZE_MAX, aOnResume);
    switch (state) {
      case SourceBufferIterator::WAITING:
        return LexerResult(Yield::NEED_MORE_DATA);
      case SourceBufferIterator::COMPLETE:
        mReadComplete = true;
        break;
      case SourceBufferIterator::READY:
        if (!mEncoded.append(reinterpret_cast<const uint8_t*>(aIterator.Data()),
                             aIterator.Length())) {
          return LexerResult(TerminalState::FAILURE);
        }
        break;
      default:
        return LexerResult(TerminalState::FAILURE);
    }
  }

  if (mEncoded.length() == 0) {
    return LexerResult(TerminalState::FAILURE);
  }

  if (!mGpu) {
    return DoDecodeCpu();
  }

  // GPU fast path. If the host upload mechanism is unavailable (no Renderer
  // worker / transfer or decode failed) it fails BEFORE posting a size; fall back
  // to the CPU bridge path in that case (don't re-post size if GPU got that far).
  LexerResult r = DoDecodeGpu();
  if (r.is<TerminalState>() && r.as<TerminalState>() == TerminalState::FAILURE &&
      !HasSize()) {
    return DoDecodeCpu();
  }
  return r;
#endif  // __EMSCRIPTEN__
}

LexerResult nsHostProxyImageDecoder::DoDecodeCpu() {
#ifndef __EMSCRIPTEN__
  return LexerResult(TerminalState::FAILURE);
#else
  const char* mime = MimeForType(mType);
  if (!mime) {
    return LexerResult(TerminalState::FAILURE);
  }

  // 2. Hand the encoded bytes to the host and block until it posts a result.
  // The control block lives on our (shared) stack; the bridge reads/writes it
  // via HEAP32 and wakes us with Atomics.notify on IMG_STATUS.
  int32_t ctrl[IMG_CTRL_INTS] = {0};
  hostimg_decode(ctrl, mime, mEncoded.begin(),
                 static_cast<int>(mEncoded.length()));

  while (__atomic_load_n(&ctrl[IMG_STATUS], __ATOMIC_SEQ_CST) == 0) {
    emscripten_futex_wait(&ctrl[IMG_STATUS], 0, 10000.0);
  }

  if (__atomic_load_n(&ctrl[IMG_STATUS], __ATOMIC_SEQ_CST) != 1) {
    return LexerResult(TerminalState::FAILURE);
  }

  const int32_t w = ctrl[IMG_W];
  const int32_t h = ctrl[IMG_H];
  const int32_t stride = ctrl[IMG_STRIDE];
  uint8_t* pixels = reinterpret_cast<uint8_t*>(ctrl[IMG_PTR]);
  if (w <= 0 || h <= 0 || stride < w * 4 || !pixels) {
    if (pixels) {
      free(pixels);
    }
    return LexerResult(TerminalState::FAILURE);
  }

  // 3. Report the size.
  PostSize(w, h);
  if (HasError()) {
    free(pixels);
    return LexerResult(TerminalState::FAILURE);
  }

  // Metadata-only decode (shouldn't happen on the CreateDecoder path, but be
  // defensive): the dimensions are all that's needed.
  if (IsMetadataDecode()) {
    free(pixels);
    return LexerResult(TerminalState::SUCCESS);
  }

  // 4. Push the decoded BGRA rows through the SurfacePipe. The host gives us
  // straight (non-premultiplied) alpha; the pipe premultiplies + swizzles +
  // downscales as needed into the imgFrame surface.
  SurfacePipeFlags pipeFlags = SurfacePipeFlags();
  if (!(GetSurfaceFlags() & SurfaceFlags::NO_PREMULTIPLY_ALPHA)) {
    pipeFlags |= SurfacePipeFlags::PREMULTIPLY_ALPHA;
  }

  Maybe<SurfacePipe> pipe = SurfacePipeFactory::CreateSurfacePipe(
      this, Size(), OutputSize(), FullFrame(), SurfaceFormat::OS_RGBA, mFormat,
      Nothing(), nullptr, pipeFlags);
  if (!pipe) {
    free(pixels);
    return LexerResult(TerminalState::FAILURE);
  }
  mPipe = std::move(*pipe);

  for (int32_t y = 0; y < h; ++y) {
    uint32_t* row =
        reinterpret_cast<uint32_t*>(pixels + static_cast<size_t>(y) * stride);
    WriteState ws = mPipe.WriteBuffer(row);
    if (ws == WriteState::FAILURE) {
      free(pixels);
      return LexerResult(TerminalState::FAILURE);
    }
    if (ws == WriteState::FINISHED) {
      break;
    }
  }

  free(pixels);

  PostFrameStop(Opacity::SOME_TRANSPARENCY);
  PostDecodeDone();
  return LexerResult(TerminalState::SUCCESS);
#endif  // __EMSCRIPTEN__
}

LexerResult nsHostProxyImageDecoder::DoDecodeGpu() {
#ifndef __EMSCRIPTEN__
  return LexerResult(TerminalState::FAILURE);
#else
  const char* mime = MimeForType(mType);
  if (!mime) {
    return LexerResult(TerminalState::FAILURE);
  }

  // Allocate a WebRender external image id; the host decodes + uploads the GL
  // texture under this id (RenderHostGpuTextureHost looks it up at composite).
  uint64_t extId =
      wr::AsUint64(layers::AsyncImagePipelineManager::GetNextExternalImageId());

  int32_t ctrl[IMG_CTRL_INTS] = {0};
  hostimg_gpu_decode(ctrl, int(uint32_t(extId & 0xffffffff)),
                     int(uint32_t(extId >> 32)), mime, mEncoded.begin(),
                     static_cast<int>(mEncoded.length()));

  while (__atomic_load_n(&ctrl[IMG_STATUS], __ATOMIC_SEQ_CST) == 0) {
    emscripten_futex_wait(&ctrl[IMG_STATUS], 0, 10000.0);
  }
  if (__atomic_load_n(&ctrl[IMG_STATUS], __ATOMIC_SEQ_CST) != 1) {
    return LexerResult(TerminalState::FAILURE);
  }

  const int32_t w = ctrl[IMG_W];
  const int32_t h = ctrl[IMG_H];
  if (w <= 0 || h <= 0) {
    return LexerResult(TerminalState::FAILURE);
  }

  PostSize(w, h);
  if (HasError()) {
    return LexerResult(TerminalState::FAILURE);
  }
  if (IsMetadataDecode()) {
    return LexerResult(TerminalState::SUCCESS);
  }

  // Produce a valid frame that carries the external image id. The pixels live in
  // the host-uploaded GL texture; WebRender composites it via the external image
  // (DecodedSurfaceProvider::UpdateKey). The CPU surface is allocated but left
  // unfilled (no readback) -- only WebRender compositing is the fast path; other
  // consumers (drawImage/snapshot) would see a blank surface.
  nsresult rv =
      AllocateFrame(OutputSize().ToUnknownSize(), SurfaceFormat::OS_RGBA);
  if (NS_FAILED(rv)) {
    return LexerResult(TerminalState::FAILURE);
  }
  if (imgFrame* frame = GetCurrentFrame()) {
    frame->SetHostGpuExternalImage(extId);
  }
  PostFrameStop(Opacity::SOME_TRANSPARENCY);
  PostDecodeDone();
  return LexerResult(TerminalState::SUCCESS);
#endif  // __EMSCRIPTEN__
}

}  // namespace image
}  // namespace mozilla
