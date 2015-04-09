/**
 *  @file
 *  @date Apr 6, 2015
 *  @author Josh Bialkowski <josh.bialkowski@gmail.com>
 */

#include <kevent/time.h>

namespace kevent {

TimeDuration PosixClock::GetTime() {
  nix::Timespec dt = nix_clock_.GetTime();
  return dt.tv_sec * 1e6 + dt.tv_nsec / 1e3;
}

TimeDuration PosixClock::GetResolution() {
  nix::Timespec dt = nix_clock_.GetRes();
  return dt.tv_sec * 1e6 + dt.tv_nsec / 1e3;
}

}  // namespace kevent

