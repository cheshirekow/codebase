/**
 *  @file
 *  @date Apr 7, 2015
 *  @author Josh Bialkowski <josh.bialkowski@gmail.com>
 */

#include <cppformat/format.h>
#include <gtest/gtest.h>
#include <kevent/eventloop.h>

TEST(EventloopTest, SimpleTest) {
  std::shared_ptr<kevent::Clock> clock =
      std::make_shared < kevent::PosixClock > (CLOCK_REALTIME);
  kevent::EventLoop event_loop(clock);
  event_loop.AddTimer([](kevent::TimeDuration) {fmt::print("Hello world\n");},
                      0, kevent::TimerPolicy::kOneShot);
  usleep(100);
  event_loop.ExecuteTimers();
}
