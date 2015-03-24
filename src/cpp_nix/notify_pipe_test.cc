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
#include <sys/wait.h>

#include <cpp_nix/fd_set.h>
#include <cpp_nix/notify_pipe.h>
#include <cpp_nix/select_set.h>
#include <cpp_nix/timeval.h>

TEST(NotifyPipe,NotifyAfterFork) {
  const int32_t sleep_time = 1e5;
  const int num_conditions = 3;

  nix::NotifyPipe condition;
  int child_pid = fork();
  bool is_parent_process = (child_pid != 0);
  if (is_parent_process) {
    nix::SelectSet select_set;
    select_set[condition.GetReadFd()] = nix::SELECT_READ;

    int num_received = 0;
    for (int i = 0; i < num_conditions; i++) {
      nix::FdSet fd_set;
      timeval timeout { 0, 2 * sleep_time };
      int result = select_set.Select(&fd_set, NULL, NULL,&timeout);

      // we expect that a timeout did not occur
      EXPECT_GT(result,0);
      EXPECT_GT(timeout,(timeval{0,0}));

      // we expect that the read FD is readable
      EXPECT_TRUE(fd_set[condition.GetReadFd()]);
      EXPECT_EQ(condition.Clear(),1);
    }
    int child_status;
    pid_t term_pid = wait(&child_status);
    EXPECT_EQ(child_pid,term_pid);
    EXPECT_EQ(WEXITSTATUS(child_status),0);
  } else {
    for (int i = 0; i < num_conditions; i++) {
      EXPECT_EQ(condition.Notify(),1);
      usleep(sleep_time);
    }
    exit(0);
  }
}

