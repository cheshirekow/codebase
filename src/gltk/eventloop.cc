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

#include <unistd.h>
#include <glog/logging.h>
#include <gltk/eventloop.h>

namespace gltk {

EventLoop::EventLoop(const std::shared_ptr<nix::Epoll>& epoll,
                     const std::shared_ptr<Pipeline>& pipeline)
    : epoll_(epoll), pipeline_(pipeline), should_quit_(false) {
  int pipe_fds[2];
  if (pipe(pipe_fds) != 0) {
    PLOG(FATAL) << "Failed to create quit pipe";
  }

  pipe_read_fd_ = pipe_fds[0];
  pipe_write_fd_ = pipe_fds[1];

  nix::epoll::Callback callback = [this]() { this->should_quit_ = true; };
  epoll_->Add(pipe_read_fd_, {{EPOLLIN, callback}});
}

void EventLoop::Run() {
  while (!should_quit_) {
    epoll_->Wait(1000);
    pipeline_->DoFrame();
  }
;}

void EventLoop::Quit() {
  write(pipe_write_fd_, "goodbye", 7);
}

}  // namespace gltk
