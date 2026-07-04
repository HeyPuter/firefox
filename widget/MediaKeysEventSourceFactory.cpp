/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this file,
 * You can obtain one at http://mozilla.org/MPL/2.0/. */

#include "mozilla/widget/MediaKeysEventSourceFactory.h"

#include "mozilla/dom/MediaControlKeySource.h"

namespace mozilla::widget {

#ifdef __EMSCRIPTEN__
class EmscriptenMediaControlKeySource final
    : public mozilla::dom::MediaControlKeySource {
 public:
  NS_INLINE_DECL_REFCOUNTING(EmscriptenMediaControlKeySource, override)

  bool Open() override { return false; }
  bool IsOpened() const override { return false; }
  void SetSupportedMediaKeys(const MediaKeysArray&) override {}

 private:
  ~EmscriptenMediaControlKeySource() override = default;
};

mozilla::dom::MediaControlKeySource* CreateMediaControlKeySource() {
  return new EmscriptenMediaControlKeySource();
}
#endif

}  // namespace mozilla::widget
