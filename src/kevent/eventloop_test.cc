/**
 *  @file
 *  @date Apr 7, 2015
 *  @author Josh Bialkowski <josh.bialkowski@gmail.com>
 */

#include <cppformat/format.h>
#include <gtest/gtest.h>
#include <kevent/eventloop.h>

TEST(EventloopTest, AbsoluteTimerTest) {
  int n_timer_calls = 0;

  kevent::EventLoop event_loop(
      std::make_shared < kevent::PosixClock > (CLOCK_MONOTONIC));

  kevent::TimerCallbackFn timer_callback =
      [&n_timer_calls](kevent::TimeDuration now) {
        n_timer_calls++;
      };

  kevent::TimerCallbackFn quit_callback =
      [&event_loop](kevent::TimeDuration now) {
        event_loop.Quit();
      };

  // Note(josh): You might think that the timeout doesn't matter since
  // absolute timers are keyed in the priority queue off of how many
  // times they've been fired from the start. However, because each loop
  // of Run() processes timers en-mass it is possible to get an extra timer
  // callback if we set the timeouts too low.
  kevent::TimerWatch* timer_watch = event_loop.AddTimer(
      timer_callback, 10e3, kevent::TimerPolicy::kAbsolute);

  kevent::TimerWatch* quit_watch = event_loop.AddTimer(
      quit_callback, 101e3, kevent::TimerPolicy::kOneShot);

  event_loop.Run();
  EXPECT_EQ(10, n_timer_calls);
}


TEST(EventloopTest, RelativeTimerTest) {
  int n_timer_calls = 0;

  kevent::EventLoop event_loop(
      std::make_shared < kevent::PosixClock > (CLOCK_MONOTONIC));

  kevent::TimerCallbackFn timer_callback =
      [&n_timer_calls](kevent::TimeDuration now) {
        n_timer_calls++;
      };

  kevent::TimerCallbackFn quit_callback =
      [&event_loop](kevent::TimeDuration now) {
        event_loop.Quit();
      };

  kevent::TimerWatch* timer_watch = event_loop.AddTimer(
      timer_callback, 10e3, kevent::TimerPolicy::kRelative);

  kevent::TimerWatch* quit_watch = event_loop.AddTimer(
      quit_callback, 101e3, kevent::TimerPolicy::kOneShot);

  event_loop.Run();
  EXPECT_EQ(10, n_timer_calls);
}
