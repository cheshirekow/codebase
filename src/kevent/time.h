#ifndef KEVENT_TIME_H_
#define KEVENT_TIME_H_

#include <cstdint>
#include <cpp_nix/clock.h>

namespace kevent {

/// In microseconds, not nanoseconds. One second is 1e6.
typedef int64_t TimeDuration;

/// Abstract interface for various clocks
class Clock {
 public:
  /// Return the 'current' time according to the clock
  virtual TimeDuration GetTime() = 0;

  /// Return the 'current' time in milliseconds according to the clock
  int GetTimeMilliseconds() {
    return GetTime() / 1e3;
  }

  /// Return the resolution of the clock
  /**
   *  The resolution of the clock is the minimum interval it is capable of
   *  resolving.
   */
  virtual TimeDuration GetResolution() = 0;

 protected:
  virtual ~Clock(){}
};

/// A clock interface into the posix clock_gettime calls
class PosixClock : public Clock {
 public:
  PosixClock(clockid_t clock_id) : nix_clock_(clock_id) {}
  virtual ~PosixClock(){}

  TimeDuration GetTime() override;
  TimeDuration GetResolution() override;

 private:
  nix::Clock nix_clock_;
};

}  // namespace kevent

#endif // KEVENT_TIME_H_

