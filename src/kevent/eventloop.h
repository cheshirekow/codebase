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

class TimerWatch;
class FDWatch;

struct TimerQueueNode {
  TimerWatch*  timer;
  int due_ms;

  TimerQueueNode(TimerWatch* timer, TimeDuration due) :
    timer(timer),
    due_ms(due) {}

  bool IsReady(const int now_ms) const {
    return due_ms <= now_ms;
  }

  bool operator<(const TimerQueueNode& other) const {
    return due_ms == other.due_ms ? timer < other.timer : due_ms < other.due_ms;
  }
};

/// Bitfields indicating epoll event types
enum FdEvent {
  kCanRead  = 0x01 << 0,
  kCanWrite = 0x01 << 1,
  kError    = 0x01 << 2,
  kHangup   = 0x01 << 3
};

/// A class for multiplexing responses to event sources, in particular:
/// file descriptors and timers
class EventLoop {
 public:
  EventLoop(const std::shared_ptr<Clock>& clock);
  TimerWatch* AddTimer(TimerCallbackFn fn, int period_ms, TimerPolicy policy =
                    TimerPolicy::kRelative);
  void RemoveTimer(TimerWatch* watch);
  FDWatch* AddFileDescriptor(int fd, FDCallbackFn fn, int events);
  void RemoveFileDescriptor(FDWatch* watch);

  void Run();
  void Reset();
  void Quit();

 private:
  std::shared_ptr<Clock> clock_;
  std::priority_queue<TimerQueueNode> timer_queue_;  ///< priority queue of timers
  std::atomic<bool> should_quit_; ///< set to true if the event loop should terminate
  int epoll_fd_;
  int pipe_read_fd_;
  int pipe_write_fd_;
  FDWatch* pipe_watch_;
};

}  // namespace kevent

#endif // KEVENT_EVENTLOOP_H_
