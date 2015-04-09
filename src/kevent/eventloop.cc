/**
 *  @file
 *  @date Apr 6, 2015
 *  @author Josh Bialkowski <josh.bialkowski@gmail.com>
 */

#include <sys/epoll.h>

#include <cppformat/format.h>
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
        n_fired(0),
        is_queued(false),
        is_freed(false) {
  }

  TimerCallbackFn fn;  ///< callback to execute on the timeout
  TimeDuration start_time;  ///< time when registration occured
  TimeDuration period;  ///< period at which the callback should be called
  TimerPolicy policy;  ///< reschedule policy
  int n_fired;  ///< number of times the callback has been fired
  bool is_queued; ///< true if the timer is in the timer queue
  bool is_freed;  ///< true if the priority loop should destroy the watch
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

TimerWatch* EventLoop::AddTimer(TimerCallbackFn fn, TimeDuration period,
                                TimerPolicy policy) {
  TimeDuration now = clock_->GetTime();
  TimerWatch* watch = new TimerWatch(fn, now, period, policy);
  timer_queue_.emplace(watch, now + period);
  watch->is_queued = true;
  return watch;
}

void EventLoop::RemoveTimer(TimerWatch* watch) {
  if(watch->is_queued) {
    watch->is_freed = true;
  } else {
    delete watch;
  }
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
  delete watch;
}

void EventLoop::Run() {
  while(!should_quit_.load()) {
    TimeDuration now = clock_->GetTime();

    // fire ready timers
    while (timer_queue_.size() > 0
        && timer_queue_.top().IsReady(now)) {
      TimerWatch* timer = timer_queue_.top().timer;
      if(timer->is_freed) {
        delete timer;
        continue;
      }
      timer_queue_.pop();
      timer->fn(now);
      timer->n_fired++;
      timer->is_queued = false;

      switch (timer->policy) {
        case TimerPolicy::kRelative: {
          TimeDuration due1 = now + timer->period;
          TimeDuration due2 = timer->start_time
              + (timer->n_fired + 1) * timer->period;
          LOG(INFO)<< fmt::format(
              "\nstart + n * dt = {:d} * {:d} = {:d}"
              "\now + dt = {:d} + {:d} = {:d}"
              "\n",
              timer->n_fired, timer->period, due2 - timer->start_time,
              now - timer->start_time, timer->period, due1 - timer->start_time);

          timer_queue_.emplace(timer, due2);
          timer->is_queued = true;
          break;
        }

        case TimerPolicy::kAbsolute:
          timer_queue_.emplace(
              timer, timer->start_time + (timer->n_fired + 1) * timer->period);
          timer->is_queued = true;
          break;

        default:
          break;
      }
    }

    // Set timeout for epoll wait. If there are no registered timers, then
    // we wait indefinitely for file descriptor events.
    int timeout_ms = -1;
    if (timer_queue_.size() > 0) {
      timeout_ms = static_cast<int>(
          (timer_queue_.top().due - now) / 1e3);
    }

//    static const int kNumEvents = 10;
//    epoll_event events[kNumEvents];
//    int epoll_result = epoll_wait(epoll_fd_, events, kNumEvents, timeout_ms);
//    if(epoll_result == -1) {
//      PLOG(WARNING) << "epoll_wait returned -1 exiting event-loop. ";
//      return;
//    }
//    for(int i=0; i < epoll_result; i++) {
//      static_cast<FDWatch*>(events[i].data.ptr)->fn(
//          FromEpollMask(events[i].events));
//    }
  }
}

void EventLoop::Reset() {
  should_quit_.store(false);
}

void EventLoop::Quit() {
  should_quit_.store(true);
}


}  // namespace kevent
