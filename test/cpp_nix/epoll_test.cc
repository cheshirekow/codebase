/*
 *  Copyright (C) 2014 Josh Bialkowski (josh.bialkowski@gmail.com)
 *
 *  This file is part of cpp-nix.
 *
 *  cpp-nix is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  cpp-nix is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with cpp-nix.  If not, see <http://www.gnu.org/licenses/>.
 */
/**
 *  @file
 *  @date   Jun 22, 2014
 *  @author Josh Bialkowski (josh.bialkowski@gmail.com)
 *  @brief
 */

#include <gtest/gtest.h>
#include <sys/types.h>

#include <cpp_nix/epoll.h>
#include <cpp_nix/fd_set.h>
#include <cpp_nix/notify_pipe.h>
#include <cpp_nix/timeval.h>

TEST(EpollTest,NotifyInOrderAfterFork) {
  const int sleep_time = 10;  // (milliseconds)
  const int num_spawn = 5;

  std::vector<nix::NotifyPipe> condition(num_spawn);
  std::vector<int> child_pid(num_spawn, 0);

  bool is_parent_process = true;
  int i_am_child_number = -1;
  for (int i = 0; i < num_spawn; i++) {
    child_pid[i] = fork();
    is_parent_process = (child_pid[i] > 0);
    if (!is_parent_process) {
      i_am_child_number = i;
      break;
    }
  }

  if (is_parent_process) {
    nix::Epoll epoll;
    std::vector<epoll_event> events_in(num_spawn);

    for (int i = 0; i < num_spawn; i++) {
      epoll_event& event = events_in[i];
      event.events = EPOLLIN;
      event.data.u32 = i;
      epoll.Add(condition[i].GetReadFd(), &event);
    }

    std::vector<epoll_event> events_out(num_spawn);
    for (int i = 0; i < num_spawn; i++) {
      int num_events = epoll.Wait(&events_out[0], events_out.size(),
                                  2 * sleep_time);
      EXPECT_EQ(num_events, 1);
      EXPECT_EQ(events_out[0].data.u32, i);
      EXPECT_EQ(condition[events_out[0].data.u32].Clear(), 1);
    }

    int child_status;
    for (int i = 0; i < num_spawn; i++) {
      pid_t dead_pid = waitpid(child_pid[i], &child_status, 0);
      EXPECT_EQ(dead_pid, child_pid[i]);
      EXPECT_EQ(WEXITSTATUS(child_status), 0);
    }
  } else {
    usleep(1000 * sleep_time * i_am_child_number);
    EXPECT_EQ(condition[i_am_child_number].Notify(), 1);
    exit(0);
  }
}

