/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef mozilla_image_decoders_nsHostProxyImageDecoder_h
#define mozilla_image_decoders_nsHostProxyImageDecoder_h

#include "Decoder.h"
#include "ImageUtils.h"  // DecoderType
#include "StreamingLexer.h"
#include "SurfacePipe.h"
#include "mozilla/Vector.h"

namespace mozilla {
namespace image {

class RasterImage;

// A single-frame image decoder that routes decoding to the HOST browser's
// WebCodecs ImageDecoder (via the gecko.js/lib/hostimg-bridge.js js-library),
// instead of decoding in-process with the wasm-compiled libjpeg/libpng/etc.
// The win is offloading pixel decode to the host's native (often GPU-backed)
// decoders. Opt-in via GECKO_IMG_PASSTHROUGH; only the full single-frame decode
// path is intercepted (DecoderFactory::CreateDecoder) -- metadata and animated
// decodes still use the native decoders.
class nsHostProxyImageDecoder final : public Decoder {
 public:
  virtual ~nsHostProxyImageDecoder();

  DecoderType GetType() const override { return mType; }

 protected:
  LexerResult DoDecode(SourceBufferIterator& aIterator,
                       IResumable* aOnResume) override;

 private:
  friend class DecoderFactory;

  // aGpu: keep the decoded image on the GPU as a WebRender external image (no CPU
  // readback). Requires GPU mode (WebRender). Otherwise the CPU bridge path.
  nsHostProxyImageDecoder(RasterImage* aImage, DecoderType aType, bool aGpu);

  LexerResult DoDecodeCpu();
  LexerResult DoDecodeGpu();

  // The complete encoded image, accumulated from the SourceBufferIterator (the
  // host ImageDecoder wants the whole buffer up front).
  Vector<uint8_t> mEncoded;
  bool mReadComplete = false;

  DecoderType mType;
  bool mGpu;
  gfx::SurfaceFormat mFormat;
  SurfacePipe mPipe;
};

}  // namespace image
}  // namespace mozilla

#endif  // mozilla_image_decoders_nsHostProxyImageDecoder_h
