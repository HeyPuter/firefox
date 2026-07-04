/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include "WebCodecsProxyDecoderModule.h"

#include "MP4Decoder.h"

#ifdef __EMSCRIPTEN__
#  include <emscripten/threading.h>
#  include <cstdint>
#  include <cstdlib>
#  include <cstring>
#  include <map>

#  include "ImageContainer.h"
#  include "MediaData.h"
#  include "MediaInfo.h"
#  include "VideoUtils.h"
#  include "mozilla/Result.h"
#  include "mozilla/TaskQueue.h"
#  include "nsThreadUtils.h"

// Shared control block layout. MUST stay in sync with
// embed-xul/webcodecs-bridge.js. All accesses are atomic int32 on the wasm
// shared heap; the worker (decode TaskQueue thread) is the single consumer and
// the browser main thread is the single producer (SPSC ring).
#  define WC_WRITE 0
#  define WC_READ 1
#  define WC_FLUSH 2
#  define WC_ERROR 3
#  define WC_FUTEX 4
#  define WC_HDR 5
#  define WC_RING_N 64
#  define WC_SLOT_INTS 14
#  define WC_CTRL_INTS (WC_HDR + WC_RING_N * WC_SLOT_INTS)

#  define SL_PTR 0
#  define SL_LEN 1
#  define SL_W 2
#  define SL_H 3
#  define SL_SY 4
#  define SL_SU 5
#  define SL_SV 6
#  define SL_OY 7
#  define SL_OU 8
#  define SL_OV 9
#  define SL_TSLO 10
#  define SL_TSHI 11
#  define SL_MTX 12
#  define SL_RANGE 13

static_assert(WC_CTRL_INTS == 901, "control block size mismatch with JS");

// Audio control block: a separate per-decoder block reusing the same atomic
// header (WC_WRITE..WC_FUTEX) as the video path. Each ring slot describes one
// chunk of interleaved float32 PCM published by the host AudioDecoder. MUST stay
// in sync with embed-xul/webcodecs-bridge.js ($wcaOff).
#  define AUDIO_RING_N 64
#  define AUDIO_SLOT_INTS 7
#  define AUDIO_CTRL_INTS (WC_HDR + AUDIO_RING_N * AUDIO_SLOT_INTS)

#  define AL_PTR 0     // shared-heap pointer to interleaved f32 PCM (malloc'd on main)
#  define AL_BYTES 1   // buffer byte length
#  define AL_FRAMES 2  // frames per channel
#  define AL_CH 3      // channel count
#  define AL_RATE 4    // sample rate
#  define AL_TSLO 5    // timestamp (microseconds) low 32 bits
#  define AL_TSHI 6    // timestamp high 32 bits

static_assert(AUDIO_CTRL_INTS == 453,
              "audio control block size mismatch with JS");

extern "C" {
// Implemented in embed-xul/webcodecs-bridge.js (emscripten js-library,
// __proxy:'sync' so they run on the browser main thread).
int webcodecs_create(int32_t* aCtrl, const char* aCodec, const uint8_t* aDesc,
                     int aDescLen, int aWidth, int aHeight);
void webcodecs_feed(int aHandle, const uint8_t* aData, int aLen, int aIsKey,
                    int aTsLo, int aTsHi);
void webcodecs_flush(int aHandle);
void webcodecs_reset(int aHandle);
void webcodecs_destroy(int aHandle);

int audiodecoder_create(int32_t* aCtrl, const char* aCodec, const uint8_t* aDesc,
                        int aDescLen, int aRate, int aChannels);
void audiodecoder_feed(int aHandle, const uint8_t* aData, int aLen, int aIsKey,
                       int aTsLo, int aTsHi);
void audiodecoder_flush(int aHandle);
void audiodecoder_reset(int aHandle);
void audiodecoder_destroy(int aHandle);
}
#endif  // __EMSCRIPTEN__

namespace mozilla {

// The WebM demuxer reports these exact mime strings (no parameters) for VPx
// tracks; see dom/media/webm/WebMDemuxer.cpp.
static bool IsVP8Mime(const nsACString& aMimeType) {
  return aMimeType.EqualsLiteral("video/vp8");
}
static bool IsVP9Mime(const nsACString& aMimeType) {
  return aMimeType.EqualsLiteral("video/vp9");
}

/* static */
already_AddRefed<PlatformDecoderModule> WebCodecsProxyDecoderModule::Create() {
  RefPtr<PlatformDecoderModule> pdm = new WebCodecsProxyDecoderModule();
  return pdm.forget();
}

media::DecodeSupportSet WebCodecsProxyDecoderModule::SupportsMimeType(
    const nsACString& aMimeType, DecoderDoctorDiagnostics*) const {
  if (MP4Decoder::IsH264(aMimeType) || IsVP8Mime(aMimeType) ||
      IsVP9Mime(aMimeType) || MP4Decoder::IsAAC(aMimeType)) {
    return media::DecodeSupport::SoftwareDecode;
  }
  return media::DecodeSupportSet{};
}

#ifdef __EMSCRIPTEN__

using layers::ImageContainer;
using layers::KnowsCompositor;

DDLoggedTypeDeclNameAndBase(WebCodecsProxyDecoder, MediaDataDecoder);

class WebCodecsProxyDecoder final
    : public MediaDataDecoder,
      public DecoderDoctorLifeLogger<WebCodecsProxyDecoder> {
 public:
  NS_INLINE_DECL_THREADSAFE_REFCOUNTING(WebCodecsProxyDecoder, final);

  enum class Codec { H264, VP8, VP9 };
  static Codec CodecFromMime(const nsACString& aMimeType) {
    if (IsVP8Mime(aMimeType)) {
      return Codec::VP8;
    }
    if (IsVP9Mime(aMimeType)) {
      return Codec::VP9;
    }
    return Codec::H264;
  }

  explicit WebCodecsProxyDecoder(const CreateDecoderParams& aParams)
      : mInfo(aParams.VideoConfig()),
        mCodec(CodecFromMime(aParams.VideoConfig().mMimeType)),
        mTaskQueue(TaskQueue::Create(
            GetMediaThreadPool(MediaThreadType::PLATFORM_DECODER),
            "WebCodecsProxyDecoder")),
        mImageContainer(aParams.mImageContainer),
        mImageAllocator(aParams.mKnowsCompositor) {}

  RefPtr<InitPromise> Init() override {
    RefPtr<WebCodecsProxyDecoder> self = this;
    return InvokeAsync(mTaskQueue, __func__, [self, this]() {
      mCtrl = static_cast<int32_t*>(calloc(WC_CTRL_INTS, sizeof(int32_t)));
      if (!mCtrl) {
        return InitPromise::CreateAndReject(
            MediaResult(NS_ERROR_OUT_OF_MEMORY,
                        RESULT_DETAIL("calloc control block failed")),
            __func__);
      }
      // Build the WebCodecs codec string and optional description. VP8/VP9 are
      // scoped to 8-bit 4:2:0 (profile 0), matching the fixed COLOR_8 +
      // HALF_WIDTH_AND_HEIGHT output in DrainRing; the host VideoDecoder reads
      // the real profile/depth from the VPx bitstream regardless of the string.
      nsCString codec;
      const uint8_t* desc = nullptr;
      int descLen = 0;
      switch (mCodec) {
        case Codec::VP8:
          codec.AssignLiteral("vp8");
          break;
        case Codec::VP9:
          codec.AssignLiteral("vp09.00.10.08");
          break;
        default:
          if (mInfo.mExtraData && mInfo.mExtraData->Length() >= 4) {
            const uint8_t* e = mInfo.mExtraData->Elements();
            codec.AppendPrintf("avc1.%02x%02x%02x", e[1], e[2], e[3]);
            desc = e;
            descLen = static_cast<int>(mInfo.mExtraData->Length());
          } else {
            codec.AssignLiteral("avc1.42E01E");
          }
          break;
      }
      mHandle = webcodecs_create(mCtrl, codec.get(), desc, descLen,
                                 mInfo.mImage.width, mInfo.mImage.height);
      if (mHandle < 0) {
        free(mCtrl);
        mCtrl = nullptr;
        return InitPromise::CreateAndReject(
            MediaResult(NS_ERROR_DOM_MEDIA_FATAL_ERR,
                        RESULT_DETAIL("WebCodecs VideoDecoder unavailable")),
            __func__);
      }
      return InitPromise::CreateAndResolve(TrackInfo::kVideoTrack, __func__);
    });
  }

  RefPtr<DecodePromise> Decode(MediaRawData* aSample) override {
    RefPtr<WebCodecsProxyDecoder> self = this;
    RefPtr<MediaRawData> sample = aSample;
    return InvokeAsync(mTaskQueue, __func__,
                       [self, this, sample]() { return ProcessDecode(sample); });
  }

  RefPtr<DecodePromise> Drain() override {
    RefPtr<WebCodecsProxyDecoder> self = this;
    return InvokeAsync(mTaskQueue, __func__, [self, this]() {
      DecodedData results;
      if (mHandle < 0) {
        return DecodePromise::CreateAndResolve(std::move(results), __func__);
      }
      uint32_t fx = static_cast<uint32_t>(CtrlLoad(WC_FUTEX));
      const int32_t flushBase = CtrlLoad(WC_FLUSH);
      webcodecs_flush(mHandle);
      while (true) {
        MediaResult r = DrainRing(results);
        if (NS_FAILED(r.Code())) {
          return DecodePromise::CreateAndReject(r, __func__);
        }
        if (CtrlLoad(WC_ERROR)) {
          return DecodePromise::CreateAndReject(
              MediaResult(NS_ERROR_DOM_MEDIA_DECODE_ERR,
                          RESULT_DETAIL("WebCodecs error during drain")),
              __func__);
        }
        if (CtrlLoad(WC_FLUSH) != flushBase) {
          break;
        }
        emscripten_futex_wait(&mCtrl[WC_FUTEX], fx, 1000.0);
        fx = static_cast<uint32_t>(CtrlLoad(WC_FUTEX));
      }
      MediaResult r = DrainRing(results);
      if (NS_FAILED(r.Code())) {
        return DecodePromise::CreateAndReject(r, __func__);
      }
      return DecodePromise::CreateAndResolve(std::move(results), __func__);
    });
  }

  RefPtr<FlushPromise> Flush() override {
    RefPtr<WebCodecsProxyDecoder> self = this;
    return InvokeAsync(mTaskQueue, __func__, [self, this]() {
      if (mHandle >= 0) {
        webcodecs_reset(mHandle);
      }
      mReadCount = 0;
      mPending.clear();
      return FlushPromise::CreateAndResolve(true, __func__);
    });
  }

  RefPtr<ShutdownPromise> Shutdown() override {
    RefPtr<WebCodecsProxyDecoder> self = this;
    return InvokeAsync(mTaskQueue, __func__, [self, this]() {
      if (mHandle >= 0) {
        webcodecs_destroy(mHandle);
        mHandle = -1;
      }
      if (mCtrl) {
        free(mCtrl);
        mCtrl = nullptr;
      }
      mPending.clear();
      return self->mTaskQueue->BeginShutdown();
    });
  }

  nsCString GetDescriptionName() const override {
    return "webcodecs proxy video decoder"_ns;
  }

  nsCString GetCodecName() const override {
    switch (mCodec) {
      case Codec::VP8:
        return "vp8"_ns;
      case Codec::VP9:
        return "vp9"_ns;
      default:
        return "h264"_ns;
    }
  }

  // H.264 samples arrive as Annex B and must be converted to AVCC (length-
  // prefixed) for EncodedVideoChunk; VP8/VP9 WebM samples are already raw
  // frames and need no conversion.
  ConversionRequired NeedsConversion() const override {
    return mCodec == Codec::H264 ? ConversionRequired::kNeedAVCC
                                 : ConversionRequired::kNeedNone;
  }

 private:
  ~WebCodecsProxyDecoder() = default;

  struct FrameMeta {
    media::TimeUnit mDuration;
    int64_t mOffset = 0;
    bool mKeyframe = false;
  };

  int32_t CtrlLoad(int aIndex) const {
    return __atomic_load_n(&mCtrl[aIndex], __ATOMIC_SEQ_CST);
  }
  void CtrlStore(int aIndex, int32_t aValue) {
    __atomic_store_n(&mCtrl[aIndex], aValue, __ATOMIC_SEQ_CST);
  }

  RefPtr<DecodePromise> ProcessDecode(MediaRawData* aSample) {
    MOZ_ASSERT(mTaskQueue->IsCurrentThreadIn());
    if (mHandle < 0) {
      return DecodePromise::CreateAndReject(
          MediaResult(NS_ERROR_DOM_MEDIA_FATAL_ERR,
                      RESULT_DETAIL("decoder not initialized")),
          __func__);
    }
    const int64_t ts = aSample->mTime.ToMicroseconds();
    mPending[ts] =
        FrameMeta{aSample->mDuration, aSample->mOffset, aSample->mKeyframe};
    const int32_t tsLo = static_cast<int32_t>(static_cast<uint32_t>(ts));
    const int32_t tsHi = static_cast<int32_t>(
        static_cast<uint32_t>(static_cast<uint64_t>(ts) >> 32));
    webcodecs_feed(mHandle, aSample->Data(), static_cast<int>(aSample->Size()),
                   aSample->mKeyframe ? 1 : 0, tsLo, tsHi);

    DecodedData results;
    MediaResult r = DrainRing(results);
    if (NS_FAILED(r.Code())) {
      return DecodePromise::CreateAndReject(r, __func__);
    }
    if (CtrlLoad(WC_ERROR)) {
      return DecodePromise::CreateAndReject(
          MediaResult(NS_ERROR_DOM_MEDIA_DECODE_ERR,
                      RESULT_DETAIL("WebCodecs decode error")),
          __func__);
    }
    return DecodePromise::CreateAndResolve(std::move(results), __func__);
  }

  MediaResult DrainRing(DecodedData& aResults) {
    const int32_t writeCount = CtrlLoad(WC_WRITE);
    while (mReadCount != writeCount) {
      const int32_t* slot =
          &mCtrl[WC_HDR + (mReadCount % WC_RING_N) * WC_SLOT_INTS];
      uint8_t* base = reinterpret_cast<uint8_t*>(
          static_cast<uintptr_t>(static_cast<uint32_t>(slot[SL_PTR])));
      const int32_t w = slot[SL_W];
      const int32_t h = slot[SL_H];
      const int32_t cw = (w + 1) / 2;
      const int32_t ch = (h + 1) / 2;

      VideoData::YCbCrBuffer b;
      b.mPlanes[0].mData = base + slot[SL_OY];
      b.mPlanes[0].mWidth = w;
      b.mPlanes[0].mHeight = h;
      b.mPlanes[0].mStride = slot[SL_SY];
      b.mPlanes[0].mSkip = 0;
      b.mPlanes[1].mData = base + slot[SL_OU];
      b.mPlanes[1].mWidth = cw;
      b.mPlanes[1].mHeight = ch;
      b.mPlanes[1].mStride = slot[SL_SU];
      b.mPlanes[1].mSkip = 0;
      b.mPlanes[2].mData = base + slot[SL_OV];
      b.mPlanes[2].mWidth = cw;
      b.mPlanes[2].mHeight = ch;
      b.mPlanes[2].mStride = slot[SL_SV];
      b.mPlanes[2].mSkip = 0;

      b.mYUVColorSpace = MatrixToColorSpace(slot[SL_MTX]);
      b.mColorPrimaries = gfx::ToColorSpace2(b.mYUVColorSpace);
      b.mColorDepth = gfx::ColorDepth::COLOR_8;
      b.mColorRange =
          slot[SL_RANGE] ? gfx::ColorRange::FULL : gfx::ColorRange::LIMITED;
      b.mChromaSubsampling = gfx::ChromaSubsampling::HALF_WIDTH_AND_HEIGHT;

      const int64_t ts = static_cast<int64_t>(
          (static_cast<uint64_t>(static_cast<uint32_t>(slot[SL_TSHI])) << 32) |
          static_cast<uint32_t>(slot[SL_TSLO]));
      const media::TimeUnit time = media::TimeUnit::FromMicroseconds(ts);
      media::TimeUnit duration = media::TimeUnit::Zero();
      int64_t offset = 0;
      bool keyframe = false;
      auto it = mPending.find(ts);
      if (it != mPending.end()) {
        duration = it->second.mDuration;
        offset = it->second.mOffset;
        keyframe = it->second.mKeyframe;
        mPending.erase(it);
      }

      Result<already_AddRefed<VideoData>, MediaResult> result =
          VideoData::CreateAndCopyData(mInfo, mImageContainer, offset, time,
                                       duration, b, keyframe, time,
                                       mInfo.ScaledImageRect(w, h),
                                       mImageAllocator);

      // The host worker handed ownership of this buffer to us; it is no longer
      // referenced by the ring once consumed, so free it after the deep copy.
      free(base);
      mReadCount++;
      CtrlStore(WC_READ, mReadCount);

      if (result.isErr()) {
        return result.unwrapErr();
      }
      RefPtr<VideoData> v = result.unwrap();
      if (v) {
        aResults.AppendElement(std::move(v));
      }
    }
    return MediaResult(NS_OK);
  }

  static gfx::YUVColorSpace MatrixToColorSpace(int32_t aMatrix) {
    switch (aMatrix) {
      case 0:
        return gfx::YUVColorSpace::BT601;
      case 2:
        return gfx::YUVColorSpace::BT2020;
      case 3:
        return gfx::YUVColorSpace::Identity;
      default:
        return gfx::YUVColorSpace::BT709;
    }
  }

  const VideoInfo mInfo;
  const Codec mCodec;
  const RefPtr<TaskQueue> mTaskQueue;
  const RefPtr<ImageContainer> mImageContainer;
  const RefPtr<KnowsCompositor> mImageAllocator;
  int mHandle = -1;
  int32_t* mCtrl = nullptr;
  int32_t mReadCount = 0;
  std::map<int64_t, FrameMeta> mPending;
};

already_AddRefed<MediaDataDecoder>
WebCodecsProxyDecoderModule::CreateVideoDecoder(
    const CreateDecoderParams& aParams) {
  const nsACString& mime = aParams.mConfig.mMimeType;
  if (!MP4Decoder::IsH264(mime) && !IsVP8Mime(mime) && !IsVP9Mime(mime)) {
    return nullptr;
  }
  RefPtr<MediaDataDecoder> decoder = new WebCodecsProxyDecoder(aParams);
  return decoder.forget();
}

DDLoggedTypeDeclNameAndBase(WebCodecsProxyAudioDecoder, MediaDataDecoder);

// Proxies AAC decoding to the host browser's WebCodecs AudioDecoder. Mirrors
// WebCodecsProxyDecoder but its ring carries interleaved float32 PCM and the
// host output callback is synchronous (AudioData.copyTo with format 'f32' is a
// sync, format-converting copy), so there is no gen/straggler race to guard.
class WebCodecsProxyAudioDecoder final
    : public MediaDataDecoder,
      public DecoderDoctorLifeLogger<WebCodecsProxyAudioDecoder> {
 public:
  NS_INLINE_DECL_THREADSAFE_REFCOUNTING(WebCodecsProxyAudioDecoder, final);

  explicit WebCodecsProxyAudioDecoder(const CreateDecoderParams& aParams)
      : mInfo(aParams.AudioConfig()),
        mTaskQueue(TaskQueue::Create(
            GetMediaThreadPool(MediaThreadType::PLATFORM_DECODER),
            "WebCodecsProxyAudioDecoder")) {}

  RefPtr<InitPromise> Init() override {
    RefPtr<WebCodecsProxyAudioDecoder> self = this;
    return InvokeAsync(mTaskQueue, __func__, [self, this]() {
      mCtrl = static_cast<int32_t*>(calloc(AUDIO_CTRL_INTS, sizeof(int32_t)));
      if (!mCtrl) {
        return InitPromise::CreateAndReject(
            MediaResult(NS_ERROR_OUT_OF_MEMORY,
                        RESULT_DETAIL("calloc audio control block failed")),
            __func__);
      }
      // Build the WebCodecs codec string "mp4a.40.<AOT>" and hand the host the
      // AudioSpecificConfig as the decoder description: MP4 AAC samples are raw
      // access units (no ADTS header), so the host needs the ASC to learn
      // rate/channels/SBR. The ASC (the DECODER_SPECIFIC_TAG payload) lives,
      // despite its name, in mDecoderConfigDescriptorBinaryBlob -- the same blob
      // ffmpeg consumes as AAC extradata. The object type is read from the ASC
      // first byte (with the 5-bit escape), falling back to the esds profile.
      int aot = mInfo.mExtendedProfile > 0 ? mInfo.mExtendedProfile
                : mInfo.mProfile > 0       ? mInfo.mProfile
                                           : 2;
      RefPtr<MediaByteBuffer> asc;
      if (mInfo.mCodecSpecificConfig.is<AacCodecSpecificData>()) {
        asc = mInfo.mCodecSpecificConfig.as<AacCodecSpecificData>()
                  .mDecoderConfigDescriptorBinaryBlob;
      }
      const uint8_t* desc = nullptr;
      int descLen = 0;
      if (asc && asc->Length() >= 2) {
        const uint8_t* e = asc->Elements();
        int a = e[0] >> 3;
        if (a == 31) {
          a = 32 + (((e[0] & 0x7) << 3) | (e[1] >> 5));
        }
        if (a > 0) {
          aot = a;
        }
        desc = e;
        descLen = static_cast<int>(asc->Length());
      }
      nsCString codec;
      codec.AppendPrintf("mp4a.40.%d", aot);
      mHandle = audiodecoder_create(mCtrl, codec.get(), desc, descLen,
                                    static_cast<int>(mInfo.mRate),
                                    static_cast<int>(mInfo.mChannels));
      if (mHandle < 0) {
        free(mCtrl);
        mCtrl = nullptr;
        return InitPromise::CreateAndReject(
            MediaResult(NS_ERROR_DOM_MEDIA_FATAL_ERR,
                        RESULT_DETAIL("WebCodecs AudioDecoder unavailable")),
            __func__);
      }
      return InitPromise::CreateAndResolve(TrackInfo::kAudioTrack, __func__);
    });
  }

  RefPtr<DecodePromise> Decode(MediaRawData* aSample) override {
    RefPtr<WebCodecsProxyAudioDecoder> self = this;
    RefPtr<MediaRawData> sample = aSample;
    return InvokeAsync(mTaskQueue, __func__,
                       [self, this, sample]() { return ProcessDecode(sample); });
  }

  RefPtr<DecodePromise> Drain() override {
    RefPtr<WebCodecsProxyAudioDecoder> self = this;
    return InvokeAsync(mTaskQueue, __func__, [self, this]() {
      DecodedData results;
      if (mHandle < 0) {
        return DecodePromise::CreateAndResolve(std::move(results), __func__);
      }
      uint32_t fx = static_cast<uint32_t>(CtrlLoad(WC_FUTEX));
      const int32_t flushBase = CtrlLoad(WC_FLUSH);
      audiodecoder_flush(mHandle);
      while (true) {
        MediaResult r = DrainRing(results);
        if (NS_FAILED(r.Code())) {
          return DecodePromise::CreateAndReject(r, __func__);
        }
        if (CtrlLoad(WC_ERROR)) {
          return DecodePromise::CreateAndReject(
              MediaResult(NS_ERROR_DOM_MEDIA_DECODE_ERR,
                          RESULT_DETAIL("WebCodecs error during drain")),
              __func__);
        }
        if (CtrlLoad(WC_FLUSH) != flushBase) {
          break;
        }
        emscripten_futex_wait(&mCtrl[WC_FUTEX], fx, 1000.0);
        fx = static_cast<uint32_t>(CtrlLoad(WC_FUTEX));
      }
      MediaResult r = DrainRing(results);
      if (NS_FAILED(r.Code())) {
        return DecodePromise::CreateAndReject(r, __func__);
      }
      return DecodePromise::CreateAndResolve(std::move(results), __func__);
    });
  }

  RefPtr<FlushPromise> Flush() override {
    RefPtr<WebCodecsProxyAudioDecoder> self = this;
    return InvokeAsync(mTaskQueue, __func__, [self, this]() {
      if (mHandle >= 0) {
        audiodecoder_reset(mHandle);
      }
      mReadCount = 0;
      mPending.clear();
      return FlushPromise::CreateAndResolve(true, __func__);
    });
  }

  RefPtr<ShutdownPromise> Shutdown() override {
    RefPtr<WebCodecsProxyAudioDecoder> self = this;
    return InvokeAsync(mTaskQueue, __func__, [self, this]() {
      if (mHandle >= 0) {
        audiodecoder_destroy(mHandle);
        mHandle = -1;
      }
      if (mCtrl) {
        free(mCtrl);
        mCtrl = nullptr;
      }
      mPending.clear();
      return self->mTaskQueue->BeginShutdown();
    });
  }

  nsCString GetDescriptionName() const override {
    return "webcodecs proxy audio decoder"_ns;
  }

  nsCString GetCodecName() const override { return "aac"_ns; }

  // MP4 AAC samples are already raw access units, exactly what
  // EncodedAudioChunk wants alongside the ASC description.
  ConversionRequired NeedsConversion() const override {
    return ConversionRequired::kNeedNone;
  }

 private:
  ~WebCodecsProxyAudioDecoder() = default;

  int32_t CtrlLoad(int aIndex) const {
    return __atomic_load_n(&mCtrl[aIndex], __ATOMIC_SEQ_CST);
  }
  void CtrlStore(int aIndex, int32_t aValue) {
    __atomic_store_n(&mCtrl[aIndex], aValue, __ATOMIC_SEQ_CST);
  }

  RefPtr<DecodePromise> ProcessDecode(MediaRawData* aSample) {
    MOZ_ASSERT(mTaskQueue->IsCurrentThreadIn());
    if (mHandle < 0) {
      return DecodePromise::CreateAndReject(
          MediaResult(NS_ERROR_DOM_MEDIA_FATAL_ERR,
                      RESULT_DETAIL("decoder not initialized")),
          __func__);
    }
    const int64_t ts = aSample->mTime.ToMicroseconds();
    mPending[ts] = aSample->mOffset;
    const int32_t tsLo = static_cast<int32_t>(static_cast<uint32_t>(ts));
    const int32_t tsHi = static_cast<int32_t>(
        static_cast<uint32_t>(static_cast<uint64_t>(ts) >> 32));
    audiodecoder_feed(mHandle, aSample->Data(),
                      static_cast<int>(aSample->Size()),
                      aSample->mKeyframe ? 1 : 0, tsLo, tsHi);

    DecodedData results;
    MediaResult r = DrainRing(results);
    if (NS_FAILED(r.Code())) {
      return DecodePromise::CreateAndReject(r, __func__);
    }
    if (CtrlLoad(WC_ERROR)) {
      return DecodePromise::CreateAndReject(
          MediaResult(NS_ERROR_DOM_MEDIA_DECODE_ERR,
                      RESULT_DETAIL("WebCodecs decode error")),
          __func__);
    }
    return DecodePromise::CreateAndResolve(std::move(results), __func__);
  }

  MediaResult DrainRing(DecodedData& aResults) {
    const int32_t writeCount = CtrlLoad(WC_WRITE);
    while (mReadCount != writeCount) {
      const int32_t* slot =
          &mCtrl[WC_HDR + (mReadCount % AUDIO_RING_N) * AUDIO_SLOT_INTS];
      uint8_t* base = reinterpret_cast<uint8_t*>(
          static_cast<uintptr_t>(static_cast<uint32_t>(slot[AL_PTR])));
      const int32_t frames = slot[AL_FRAMES];
      const int32_t channels = slot[AL_CH];
      const int32_t rate = slot[AL_RATE];
      const int64_t ts = static_cast<int64_t>(
          (static_cast<uint64_t>(static_cast<uint32_t>(slot[AL_TSHI])) << 32) |
          static_cast<uint32_t>(slot[AL_TSLO]));

      int64_t offset = 0;
      auto it = mPending.find(ts);
      if (it != mPending.end()) {
        offset = it->second;
        mPending.erase(it);
      }

      const size_t samples = static_cast<size_t>(frames) * channels;
      AlignedAudioBuffer audio(samples);
      if (!audio) {
        free(base);
        mReadCount++;
        CtrlStore(WC_READ, mReadCount);
        return MediaResult(NS_ERROR_OUT_OF_MEMORY,
                           RESULT_DETAIL("audio buffer alloc failed"));
      }
      memcpy(audio.get(), base, samples * sizeof(AudioDataValue));

      // The host handed us ownership of this buffer; free it after the copy.
      free(base);
      mReadCount++;
      CtrlStore(WC_READ, mReadCount);

      RefPtr<AudioData> data = new AudioData(
          offset, media::TimeUnit::FromMicroseconds(ts), std::move(audio),
          static_cast<uint32_t>(channels), static_cast<uint32_t>(rate),
          mInfo.mChannelMap);
      aResults.AppendElement(std::move(data));
    }
    return MediaResult(NS_OK);
  }

  const AudioInfo mInfo;
  const RefPtr<TaskQueue> mTaskQueue;
  int mHandle = -1;
  int32_t* mCtrl = nullptr;
  int32_t mReadCount = 0;
  std::map<int64_t, int64_t> mPending;
};

already_AddRefed<MediaDataDecoder>
WebCodecsProxyDecoderModule::CreateAudioDecoder(
    const CreateDecoderParams& aParams) {
  if (!MP4Decoder::IsAAC(aParams.mConfig.mMimeType)) {
    return nullptr;
  }
  RefPtr<MediaDataDecoder> decoder = new WebCodecsProxyAudioDecoder(aParams);
  return decoder.forget();
}

#else  // __EMSCRIPTEN__

already_AddRefed<MediaDataDecoder>
WebCodecsProxyDecoderModule::CreateVideoDecoder(const CreateDecoderParams&) {
  return nullptr;
}

already_AddRefed<MediaDataDecoder>
WebCodecsProxyDecoderModule::CreateAudioDecoder(const CreateDecoderParams&) {
  return nullptr;
}

#endif  // __EMSCRIPTEN__

}  // namespace mozilla
