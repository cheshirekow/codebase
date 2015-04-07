/**
 *  @file
 *  @date Apr 6, 2015
 *  @author Josh Bialkowski <josh.bialkowski@gmail.com>
 */

#include <kevent/eventloop.h>

namespace kevent {

Timer::Timer(TimerCallbackFn fn, TimeDuration current_time, TimeDuration period,
             TimerPolicy policy)
    : fn(fn),
      start_time(current_time),
      period(period),
      policy(policy),
      n_fired(0) {
}


EventLoop::EventLoop(const std::shared_ptr<Clock>& clock) :
  clock_(clock),
  epoll_fd_(0) {
  should_quit_.store(false);
  epoll_fd_ = epoll_create(1);
}

void EventLoop::Run() {
  while(!should_quit_.load()) {
    // fire ready timers
    ExecuteTimers();

    // set timeout for epoll wait
    int timeout_ms = -1;
    if (timer_queue_.size() > 0) {
      timeout_ms = static_cast<int>(
          (timer_queue_.top().due - clock_->GetTime()) / 1000);
    }

    static const int kNumEvents = 10;
    epoll_event events[kNumEvents];
    epoll_wait(epoll_fd_, events, kNumEvents, timeout_ms);
  }
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
  timer_queue_.emplace(new Timer(fn, now, period, policy), now + period);
}

void EventLoop::ExecuteTimers() {
  while (timer_queue_.size() > 0
      && timer_queue_.top().IsReady(clock_->GetTime())) {
    TimeDuration now = clock_->GetTime();
    Timer* timer = timer_queue_.top().timer;
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
