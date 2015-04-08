#ifndef KEVENT_EVENTLOOP_H_
#define KEVENT_EVENTLOOP_H_

#include <atomic>
#include <functional>
#include <list>
#include <memory>
#include <queue>

#include <kevent/time.h>

namespace kevent {

/// Callback for a timer watch.
typedef std::function<void (TimeDuration)> TimerCallbackFn;

/// Callback for a FdWatch, the argument supplied to the callback is a bitmask
/// of which events are ready on the file descriptor
typedef std::function<void (int)> FDCallbackFn;

enum class TimerPolicy : uint8_t {
  kOneShot,   ///< fire the timer once and then cleanup
  kRelative,  ///< schedule timer based on previous fired time + period
  kAbsolute,  ///< schedule timer based on registration time + n*period
  kCleanup,   ///< timer has been unsubscribed, remove before firing
};

struct TimerWatch {
  TimerWatch(TimerCallbackFn fn, TimeDuration current_time,
                    TimeDuration period, TimerPolicy policy);

  TimerCallbackFn fn;  ///< callback to execute on the timeout
  TimeDuration start_time;  ///< time when registration occured
  TimeDuration period;  ///< period at which the callback should be called
  TimerPolicy policy;  ///< reschedule policy
  int n_fired;  ///< number of times the callback has been fired
};

struct TimerQueueNode {
  TimerWatch*  timer;
  TimeDuration due;

  TimerQueueNode(TimerWatch* timer, TimeDuration due) :
    timer(timer),
    due(due) {}

  bool IsReady(const TimeDuration now) const {
    return due <= now;
  }

  bool operator<(const TimerQueueNode& other) const {
    return due == other.due ? timer < other.timer : due < other.due;
  }
};


/// Bitfields indicating epoll event types
enum FdEvent {
  kCanRead  = 0x01 << 0,
  kCanWrite = 0x01 << 1,
  kError    = 0x01 << 2,
  kHangup   = 0x01 << 3
};



struct FDWatch {
  FDCallbackFn  fn;
};



/// A class for multiplexing responses to event sources, in particular:
/// file descriptors and timers
class EventLoop {
 public:
  EventLoop(const std::shared_ptr<Clock>& clock);
  void AddTimer(TimerCallbackFn fn, TimeDuration period, TimerPolicy policy =
                    TimerPolicy::kRelative);
  void ExecuteTimers();
  int Run();
  void Reset();
  void Quit();

 private:
  std::shared_ptr<Clock> clock_;
  std::priority_queue<TimerQueueNode> timer_queue_;  ///< priority queue of timers
  std::atomic<bool> should_quit_; ///< set to true if the event loop should terminate
  int epoll_fd_;
};



}  // namespace kevent

#endif // KEVENT_EVENTLOOP_H_
