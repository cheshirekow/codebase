/*
 *  Copyright (C) 2012 Josh Bialkowski (josh.bialkowski@gmail.com)
 *
 *  This file is part of gltk.
 *
 *  gltk is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  gltk is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with gltk.  If not, see <http://www.gnu.org/licenses/>.
 */
/**
 *  @file
 *  @date   May 1, 2015
 *  @author Josh Bialkowski (josh.bialkowski@gmail.com)
 *  @brief
 */

#include <sys/timerfd.h>
#include <gtest/gtest.h>
#include <gltk/eventloop.h>

namespace gltk {

TEST(EventLoopTest, SimpleTest) {
  std::shared_ptr<nix::Epoll> epoll_ptr(new nix::Epoll());
  std::shared_ptr<Pipeline> pipeline_ptr(new Pipeline());
  EventLoop loop(epoll_ptr, pipeline_ptr);

  int timer_fd = timerfd_create(CLOCK_MONOTONIC, 0);
  ::itimerspec spec{{1, 0}, {1, 0}};  // 1 second timeout
  timerfd_settime(timer_fd, 0, &spec, NULL);

  auto on_timer_expiration = [&loop]() { loop.Quit(); };
  epoll_ptr->Add(timer_fd, {{EPOLLIN, on_timer_expiration}});

  loop.Run();
}

}  // namespace gltk
