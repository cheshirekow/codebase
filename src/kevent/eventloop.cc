/**
 *  @file
 *  @date Apr 6, 2015
 *  @author Josh Bialkowski <josh.bialkowski@gmail.com>
 */

#include <sys/epoll.h>

#include <glog/logging.h>
#include <kevent/eventloop.h>

namespace kevent {

/// Maps field masks for our FdEvent bitvector to field masks for the epoll
/// bitvector.
std::vector<std::pair<int,int>> kBitMap = {
    {EPOLLIN, kCanRead},
    {EPOLLOUT, kCanWrite},
    {EPOLLERR, kError},
    {EPOLLHUP, kHangup}
};

static inline int FromEpollMask(int mask) {
  int kevent_mask = 0x00;
  for( auto& pair: kBitMap) {
    if(mask & pair.first) {
      kevent_mask |= pair.second;
    }
  }
  return kevent_mask;
}

static inline int ToEpollMask(int kevent_mask) {
  int mask = 0x00;
  for (auto& pair : kBitMap) {
    if (kevent_mask & pair.second) {
      mask |= pair.first;
    }
  }
  return mask;
}

struct TimerWatch {
  TimerWatch(TimerCallbackFn fn, TimeDuration current_time, TimeDuration period,
             TimerPolicy policy)
      : fn(fn),
        start_time(current_time),
        period(period),
        policy(policy),
        n_fired(0) {
  }

  TimerCallbackFn fn;  ///< callback to execute on the timeout
  TimeDuration start_time;  ///< time when registration occured
  TimeDuration period;  ///< period at which the callback should be called
  TimerPolicy policy;  ///< reschedule policy
  int n_fired;  ///< number of times the callback has been fired
};


struct FDWatch {
  FDWatch(int fd, FDCallbackFn fn, int events):
    fd(fd),
    fn(fn) {
    event.events = ToEpollMask(events);
    event.data.ptr = this;
  }

  int           fd;
  FDCallbackFn  fn;
  epoll_event   event;
};

EventLoop::EventLoop(const std::shared_ptr<Clock>& clock) :
  clock_(clock),
  epoll_fd_(0) {
  should_quit_.store(false);
  epoll_fd_ = epoll_create(1);
}

int EventLoop::Run() {
  while(!should_quit_.load()) {
    // fire ready timers
    ExecuteTimers();

    // Set timeout for epoll wait. If there are no registered timers, then
    // we wait indefinitely for file descriptor events.
    int timeout_ms = -1;
    if (timer_queue_.size() > 0) {
      timeout_ms = static_cast<int>(
          (timer_queue_.top().due - clock_->GetTime()) / 1000);
    }

    static const int kNumEvents = 10;
    epoll_event events[kNumEvents];
    int epoll_result = epoll_wait(epoll_fd_, events, kNumEvents, timeout_ms);
    if(epoll_result == -1) {
      PLOG(WARNING) << "epoll_wait returned -1 exiting event-loop. ";
      return epoll_result;
    }
    for(int i=0; i < epoll_result; i++) {
      static_cast<FDWatch*>(events[i].data.ptr)->fn(
          FromEpollMask(events[i].events));
    }
  }
  return 0;
}

void EventLoop::Reset() {
  should_quit_.store(false);
}

void EventLoop::Quit() {
  should_quit_.store(true);
}

void EventLoop::AddTimer(TimerCallbackFn fn, TimeDuration period,
                         TimerPolicy policy) {
  TimeDuration now = clock_->GetTime();
  timer_queue_.emplace(new TimerWatch(fn, now, period, policy), now + period);
}

FDWatch* EventLoop::AddFileDescriptor(int fd, FDCallbackFn fn, int events) {
  FDWatch* watch = new FDWatch(fd, fn, events);
  int epoll_result = epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, fd, &watch->event);
  PLOG_IF(WARNING, epoll_result == -1)
      << "epoll_ctl returned -1 while trying to add a watch.";
  return watch;
}

void EventLoop::RemoveFileDescriptor(FDWatch* watch) {
  int epoll_result = epoll_ctl(epoll_fd_, EPOLL_CTL_DEL, watch->fd,
                               &watch->event);
  PLOG_IF(WARNING, epoll_result == -1)
      << "epoll_ctl returned -1 while trying to remove a watch.";
}

void EventLoop::ExecuteTimers() {
  while (timer_queue_.size() > 0
      && timer_queue_.top().IsReady(clock_->GetTime())) {
    TimeDuration now = clock_->GetTime();
    TimerWatch* timer = timer_queue_.top().timer;
    timer_queue_.pop();
    timer->fn(now);
    timer->n_fired++;

    switch (timer->policy) {
      case TimerPolicy::kRelative:
        timer_queue_.emplace(timer, now + timer->period);
        break;

      case TimerPolicy::kAbsolute:
        timer_queue_.emplace(
            timer, timer->start_time + timer->n_fired * timer->period);
        break;

      default:
        break;
    }

  }
}

}  // namespace kevent
