/*
 *  Copyright (C) 2012 Josh Bialkowski (josh.bialkowski@gmail.com)
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
 *  @date Jun 22, 2014
 *  @author Josh Bialkowski
 */

#ifndef CPP_NIX_EPOLL_H_
#define CPP_NIX_EPOLL_H_

#include <sys/epoll.h>

namespace nix {
namespace epoll {

struct Flags {
 public:
  Flags(int flags)
      : m_flags(flags) {
  }

  int Get() {
    return m_flags;
  }

 private:
  int m_flags;
};

}

class Epoll {
 public:
  Epoll() {
    m_epfd = epoll_create(1);
  }

  Epoll(epoll::Flags flags) {
    m_epfd = epoll_create1(flags.Get());
  }

  ~Epoll() {
    if (m_epfd > 0) {
      close(m_epfd);
    }
  }

  int GetFd() {
    return m_epfd;
  }

  int Add(int fd, epoll_event* event) {
    return epoll_ctl(m_epfd, EPOLL_CTL_ADD, fd, event);
  }

  int Modify(int fd, epoll_event* event) {
    return epoll_ctl(m_epfd, EPOLL_CTL_ADD, fd, event);
  }

  int Delete(int fd) {
    return epoll_ctl(m_epfd, EPOLL_CTL_DEL, fd, NULL);
  }

  int Wait(struct epoll_event *events, int maxevents, int timeout) {
    return epoll_wait(m_epfd, events, maxevents, timeout);
  }

  int Pwait(epoll_event *events, int maxevents, int timeout,
            const sigset_t *sigmask) {
    return epoll_pwait(m_epfd, events, maxevents, timeout, sigmask);
  }

 private:
  int m_epfd;
};

}  // namespace nix

#endif  // CPP_NIX_EPOLL_H_
