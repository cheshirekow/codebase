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

#ifndef GLTK_EVENTLOOP_H_
#define GLTK_EVENTLOOP_H_

#include <atomic>
#include <memory>
#include <cpp_nix/epoll.h>
#include <gltk/pipeline.h>

namespace gltk {

/// Manages the top-level event loop
/**
 *  At the top level the event loop has two passes:
 *    1) Process all inbound events
 *    2) Render a frame if necessary
 */
class EventLoop {
 public:
  EventLoop(const std::shared_ptr<nix::Epoll>& epoll,
            const std::shared_ptr<Pipeline>& pipeline);

  void Run();
  void Quit();

 private:
  std::shared_ptr<nix::Epoll> epoll_;
  std::shared_ptr<Pipeline> pipeline_;
  std::atomic<bool> should_quit_;
  int pipe_read_fd_;
  int pipe_write_fd_;
};

}  // namespace gltk

#endif  // GLTK_EVENTLOOP_H_
