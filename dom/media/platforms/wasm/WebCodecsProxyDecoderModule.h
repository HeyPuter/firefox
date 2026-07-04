/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef DOM_MEDIA_PLATFORMS_WASM_WEBCODECSPROXYDECODERMODULE_H_
#define DOM_MEDIA_PLATFORMS_WASM_WEBCODECSPROXYDECODERMODULE_H_

#include "PlatformDecoderModule.h"

namespace mozilla {

// PlatformDecoderModule that proxies H.264 video decoding to the host browser's
// WebCodecs VideoDecoder (running on the emscripten main thread). Used in the
// wasm build, where no native or system decoder is available. The actual
// transport lives in the .cpp behind __EMSCRIPTEN__; on other platforms this
// module produces no decoders.
class WebCodecsProxyDecoderModule : public PlatformDecoderModule {
 public:
  static already_AddRefed<PlatformDecoderModule> Create();

  media::DecodeSupportSet SupportsMimeType(
      const nsACString& aMimeType,
      DecoderDoctorDiagnostics* aDiagnostics) const override;

  already_AddRefed<MediaDataDecoder> CreateVideoDecoder(
      const CreateDecoderParams& aParams) override;

  already_AddRefed<MediaDataDecoder> CreateAudioDecoder(
      const CreateDecoderParams& aParams) override;

  const char* Name() const override { return "WebCodecsProxy"; }

 protected:
  WebCodecsProxyDecoderModule() = default;
  ~WebCodecsProxyDecoderModule() override = default;
};

}  // namespace mozilla

#endif  // DOM_MEDIA_PLATFORMS_WASM_WEBCODECSPROXYDECODERMODULE_H_
