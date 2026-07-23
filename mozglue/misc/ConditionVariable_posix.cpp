/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include "mozilla/Assertions.h"
#include "mozilla/CheckedInt.h"

#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <time.h>

#include "mozilla/PlatformConditionVariable.h"
#include "mozilla/PlatformMutex.h"

using mozilla::CheckedInt;
using mozilla::TimeDuration;

static const long NanoSecPerSec = 1000000000;

// macOS has the clock functions, but not pthread_condattr_setclock.
#if defined(HAVE_CLOCK_MONOTONIC) && !defined(__APPLE__)
#  define CV_USE_CLOCK_API
#endif

#ifdef CV_USE_CLOCK_API
// The C++ specification defines std::condition_variable::wait_for in terms of
// std::chrono::steady_clock, which is closest to CLOCK_MONOTONIC.
static const clockid_t WhichClock = CLOCK_MONOTONIC;

// While timevaladd is widely available to work with timevals, the newer
// timespec structure is largely lacking such conveniences. Thankfully, the
// utilities available in MFBT make implementing our own quite easy.
static void moz_timespecadd(struct timespec* lhs, struct timespec* rhs,
                            struct timespec* result) {
  // Add nanoseconds. This may wrap, but not above 2 billion.
  MOZ_RELEASE_ASSERT(lhs->tv_nsec < NanoSecPerSec);
  MOZ_RELEASE_ASSERT(rhs->tv_nsec < NanoSecPerSec);
  result->tv_nsec = lhs->tv_nsec + rhs->tv_nsec;

  // Add seconds, checking for overflow in the platform specific time_t type.
  CheckedInt<time_t> sec = CheckedInt<time_t>(lhs->tv_sec) + rhs->tv_sec;

  // If nanoseconds overflowed, carry the result over into seconds.
  if (result->tv_nsec >= NanoSecPerSec) {
    MOZ_RELEASE_ASSERT(result->tv_nsec < 2 * NanoSecPerSec);
    result->tv_nsec -= NanoSecPerSec;
    sec += 1;
  }

  // Extracting the value asserts that there was no overflow.
  MOZ_RELEASE_ASSERT(sec.isValid());
  result->tv_sec = sec.value();
}
#endif

#if defined(__EMSCRIPTEN__) && !defined(__EMSCRIPTEN_PTHREADS__)
// Single-threaded wasm: there are no OS threads, so a condvar wait can never
// be satisfied by another thread. Emscripten's stub pthread_cond_(timed)wait
// returns 0 immediately (a permitted spurious wakeup), which keeps wait loops
// from deadlocking but makes no forward progress. This hook lets xpcom pump
// the virtual-thread scheduler (drain cooperatively-scheduled event queues,
// fire due timers) on every wait so the condition being waited on can actually
// become true. Returns true if any work was done. The idle hook is invoked
// when a wait made no progress (the embedder may yield to the JS event loop
// there, briefly).
extern "C" {
bool (*gecko_st_wait_hook)(void) = nullptr;
void (*gecko_st_idle_hook)(void) = nullptr;
}
static void GeckoSTOnWait() {
  static unsigned long long sNoProgress = 0;
  bool progress = gecko_st_wait_hook ? gecko_st_wait_hook() : false;
  if (!progress && gecko_st_idle_hook) {
    gecko_st_idle_hook();
  }
  // Diagnostic: millions of consecutive no-progress waits = wedged on a
  // condition nothing can satisfy; abort() so the trap shows the waiter.
  if (progress) {
    sNoProgress = 0;
  } else if (sNoProgress++ % 5000000ULL == 4999999ULL) {
    fprintf(stderr, "GeckoSTOnWait: WARNING %llu consecutive no-progress waits\n",
            (unsigned long long)sNoProgress);
  }
}
#  define GECKO_ST_ON_WAIT() GeckoSTOnWait()
#else
#  define GECKO_ST_ON_WAIT() \
    do {                     \
    } while (0)
#endif

mozilla::detail::ConditionVariableImpl::ConditionVariableImpl() {
#ifdef CV_USE_CLOCK_API
  pthread_condattr_t attr;
  int r0 = pthread_condattr_init(&attr);
  MOZ_RELEASE_ASSERT(!r0);

  int r1 = pthread_condattr_setclock(&attr, WhichClock);
  MOZ_RELEASE_ASSERT(!r1);

  int r2 = pthread_cond_init(&mCond, &attr);
  MOZ_RELEASE_ASSERT(!r2);

  int r3 = pthread_condattr_destroy(&attr);
  MOZ_RELEASE_ASSERT(!r3);
#else
  int r = pthread_cond_init(&mCond, NULL);
  MOZ_RELEASE_ASSERT(!r);
#endif
}

mozilla::detail::ConditionVariableImpl::~ConditionVariableImpl() {
  int r = pthread_cond_destroy(&mCond);
  MOZ_RELEASE_ASSERT(r == 0);
}

void mozilla::detail::ConditionVariableImpl::notify_one() {
  int r = pthread_cond_signal(&mCond);
  MOZ_RELEASE_ASSERT(r == 0);
}

void mozilla::detail::ConditionVariableImpl::notify_all() {
  int r = pthread_cond_broadcast(&mCond);
  MOZ_RELEASE_ASSERT(r == 0);
}

void mozilla::detail::ConditionVariableImpl::wait(MutexImpl& lock) {
  GECKO_ST_ON_WAIT();
  int r = pthread_cond_wait(&mCond, &lock.mMutex);
  MOZ_RELEASE_ASSERT(r == 0);
}

mozilla::CVStatus mozilla::detail::ConditionVariableImpl::wait_for(
    MutexImpl& lock, const TimeDuration& a_rel_time) {
  if (a_rel_time == TimeDuration::Forever()) {
    wait(lock);
    return CVStatus::NoTimeout;
  }

#if defined(__EMSCRIPTEN__) && !defined(__EMSCRIPTEN_PTHREADS__)
  // Pump instead of blocking; report Timeout only when the deadline really
  // elapsed so `while (wait_for(..) != Timeout)` loops stay time-bounded and
  // periodic waits don't fire early.
  {
    struct timespec st_start;
    clock_gettime(CLOCK_MONOTONIC, &st_start);
    GECKO_ST_ON_WAIT();
    struct timespec st_now;
    clock_gettime(CLOCK_MONOTONIC, &st_now);
    double elapsed_ms = (st_now.tv_sec - st_start.tv_sec) * 1000.0 +
                        (st_now.tv_nsec - st_start.tv_nsec) / 1e6;
    return elapsed_ms >= a_rel_time.ToMilliseconds() ? CVStatus::Timeout
                                                     : CVStatus::NoTimeout;
  }
#endif

  int r;

  // Clamp to 0, as time_t is unsigned.
  TimeDuration rel_time = a_rel_time < TimeDuration::FromSeconds(0)
                              ? TimeDuration::FromSeconds(0)
                              : a_rel_time;

  // Convert the duration to a timespec.
  struct timespec rel_ts;
  rel_ts.tv_sec = static_cast<time_t>(rel_time.ToSeconds());
  rel_ts.tv_nsec =
      static_cast<uint64_t>(rel_time.ToMicroseconds() * 1000.0) % NanoSecPerSec;

#ifdef CV_USE_CLOCK_API
  struct timespec now_ts;
  r = clock_gettime(WhichClock, &now_ts);
  MOZ_RELEASE_ASSERT(!r);

  struct timespec abs_ts;
  moz_timespecadd(&now_ts, &rel_ts, &abs_ts);

  r = pthread_cond_timedwait(&mCond, &lock.mMutex, &abs_ts);
#else
  // Our non-clock-supporting platforms, OS X and Android, do support waiting
  // on a condition variable with a relative timeout.
  r = pthread_cond_timedwait_relative_np(&mCond, &lock.mMutex, &rel_ts);
#endif

  if (r == 0) {
    return CVStatus::NoTimeout;
  }
  MOZ_RELEASE_ASSERT(r == ETIMEDOUT);
  return CVStatus::Timeout;
}
