#ifndef KEVENT_EVENTLOOP_H_
#define KEVENT_EVENTLOOP_H_

#include <queue>
#include <functional>
#include <kevent/time.h>

namespace kevent {

typedef std::function<void (TimeDuration)> TimerCallbackFn;
typedef std::function<void (int)> FileDescriptorCallbackFn;


enum class TimerPolicy : uint8_t {
  kOneShot = 0x01,
  kRelative = 0x02,
  kAbsolute = 0x03
};

class TimerRegistration {
 public:

 private:
  TimerCallbackFn fn_;  ///< callback to execute on the timeout
  TimeDuration  period_;  ///< period at which the callback should be called
  TimerPolicy   policy_;  ///< reschedule policy
};

class TimerQueueNode {
 public:
  bool operator<(const TimerQueueNode& other) {
    return due_ == other.due_ ? timer_ < other.timer_ : due_ < other.due_;
  }

 private:
  TimeDuration due_;  ///< timestamp when next to fire this timer
  TimerRegistration*  timer_; ///< the timer that is registered
};


/// A class for multiplexing responses to event sources, in particular:
/// file descriptors and timers
class EventLoop {
 public:
  void AddTimer(TimerCallbackFn fn, TimeDuration period, TimerPolicy policy =
                    TimerPolicy::kRelative);

 private:
  std::priority_queue<TimerQueueNode> timer_queue_;  ///< priority queue of timers

};


}  // namespace kevent

#endif // KEVENT_EVENTLOOP_H_
