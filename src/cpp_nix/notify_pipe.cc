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
 *  @date   Jun 22, 2013
 *  @author Josh Bialkowski (josh.bialkowski@gmail.com)
 *  @brief  
 */
#include <cpp_nix/notify_pipe.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>

namespace nix {

NotifyPipe::NotifyPipe() {
  if (pipe(m_fd)){
    //ex()() << "Failed to open a pipe\n";
  }

  // make the read end nonblocking
  if (fcntl(GetReadFd(), F_SETFL, O_NONBLOCK)){
    //ex()() << "Failed to make pipe nonblocking";
  }
}

NotifyPipe::~NotifyPipe() {
  for (int i = 0; i < 2; i++)
    close(m_fd[i]);
}

int NotifyPipe::GetWriteFd() {
  return m_fd[1];
}

int NotifyPipe::GetReadFd() {
  return m_fd[0];
}

int NotifyPipe::Notify() {
  return write(GetWriteFd(), "x", 1);
}

int NotifyPipe::Clear() {
  char c;
  int result = 1;

  // read until there's nothing left
  int bytes_read = 0;
  while (result > 0){
    result = read(GetReadFd(), &c, 1);
    if(result >= 0){
      bytes_read += result;
    }
  }

  if (result < 0 && errno != EWOULDBLOCK) {
    return result;
  } else {
    return bytes_read;
  }
}

}  // namespace nix
