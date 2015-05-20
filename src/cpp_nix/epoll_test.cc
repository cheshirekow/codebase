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

#include <sys/types.h>
#include <vector>

#include <gtest/gtest.h>
#include <glog/logging.h>
#include <cpp_nix/epoll.h>
#include <cpp_nix/fd_set.h>
#include <cpp_nix/notify_pipe.h>
#include <cpp_nix/timeval.h>

static const int kSleepTime = 10;  // (milliseconds)
static const int kNumSpawn = 5;

TEST(EpollTest, NotifyInOrderAfterFork) {
  std::vector<nix::NotifyPipe> condition(kNumSpawn);
  std::vector<int> child_pid(kNumSpawn, 0);

  bool is_parent_process = true;
  int i_am_child_number = -1;
  for (int i = 0; i < kNumSpawn; i++) {
    child_pid[i] = fork();
    is_parent_process = (child_pid[i] > 0);
    if (!is_parent_process) {
      i_am_child_number = i;
      break;
    }
  }

  if (is_parent_process) {
    nix::Epoll epoll;
    std::vector<epoll_event> events_in(kNumSpawn);

    bool notified[kNumSpawn];
    for (int i = 0; i < kNumSpawn; i++) {
      bool& bool_ref = notified[i];
      bool_ref = false;
      auto callback = [&bool_ref, i](){bool_ref = true;};
      epoll.Add(condition[i].GetReadFd(), {{EPOLLIN, callback}});
    }

    for (int i = 0; i < kNumSpawn; i++) {
      int num_events = epoll.Wait(2 * kSleepTime);
      condition[i].Clear();
      EXPECT_EQ(num_events, 1);
    }

    for (int i=0; i < kNumSpawn; i++) {
      EXPECT_TRUE(notified[i]) << " for child " << i;
    }

    int child_status;
    for (int i = 0; i < kNumSpawn; i++) {
      pid_t dead_pid = waitpid(child_pid[i], &child_status, 0);
      EXPECT_EQ(dead_pid, child_pid[i]);
      EXPECT_EQ(WEXITSTATUS(child_status), 0);
    }
  } else {
    usleep(1000 * kSleepTime * i_am_child_number);
    EXPECT_EQ(condition[i_am_child_number].Notify(), 1);
    exit(0);
  }
}

