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
 *
 *  @date   Jun 22, 2013
 *  @author Josh Bialkowski (josh.bialkowski@gmail.com)
 *  @brief  
 */

#ifndef CPP_NIX_NOTIFYPIPE_H_
#define CPP_NIX_NOTIFYPIPE_H_

namespace nix {

/// Manages a pipe which can be used as a condition variable between two
/// threads. This is in contrast to a pthreads condition variable which
/// can't be `selected`-ed or `epoll`-ed.
class NotifyPipe {
 private:
  int m_fd[2];

 public:
  /// opens pipe
  NotifyPipe();

  /// closes pipe
  ~NotifyPipe();

  /// Return the file descriptor for the write end of the pipe
  int GetWriteFd();

  /// Return the file descriptor for the read end of the pipe
  int GetReadFd();

  /// Write a single byte to the pipe to signal all listeners
  int Notify();

  /// Read from the pipe until it's empty
  int Clear();
};

}  // namespace nix

#endif // NOTIFYPIPE_H_
