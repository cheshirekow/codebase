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
  clock_(clock) {

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
