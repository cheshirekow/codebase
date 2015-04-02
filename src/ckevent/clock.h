/*
 * clock.h
 *
 *  Created on: Apr 2, 2015
 *      Author: josh
 */

#ifndef CKEVENT_CLOCK_H_
#define CKEVENT_CLOCK_H_

namespace ckevent {

typedef int64_t TimeDuration;

/// Abstract interface for various clocks
class Clock {
 public:
  /// Return the 'current' time according to the clock
  virtual TimeDuration GetTime() = 0;

  /// Return the resolution of the clock
  /**
   *  The resolution of the clock is the minimum interval it is capable of
   *  resolving.
   */
  virtual TimeDuration GetResolution() = 0;

 private:
  virtual ~Clock(){}
};

/// A clock interface into the posix clock_gettime calls
class PosixClock : public Clock {
 public:
  PosixClock(clockid_t clock_id) : nix_clock_(clock_id) {}

  TimeDuration GetTime() override {
    nix::Timespec dt = nix_clock_.GetTime();
    return dt.tv_sec * 1000 + dt.tv_nsec / 1000;
  }

  TimeDuration GetResolution() override {
    nix::Timespec dt = nix_clock_.GetRes();
    return dt.tv_sec * 1000 + dt.tv_nsec / 1000;
  }

 private:
  nix::Clock nix_clock_;
};

}  // namespace ckevent

#endif // CKEVENT_CLOCK_H_
