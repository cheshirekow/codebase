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
 *  @date   Feb 14, 2013
 *  @author Josh Bialkowski (josh.bialkowski@gmail.com)
 *  @brief  
 */

#include <cpp_nix/fd_set.h>

namespace nix {

FdSet::Ref::Ref(fd_set* set, int fd)
    : m_set(set),
      m_fd(fd) {
}

FdSet::Ref::operator bool() const {
  return FD_ISSET(m_fd, m_set);
}

FdSet::Ref& FdSet::Ref::operator=(bool x) {
  if (x)
    FD_SET(m_fd, m_set);
  else
    FD_CLR(m_fd, m_set);
  return *this;
}

FdSet::Ref& FdSet::Ref::flip() {
  if (*this)
    *this = false;
  else
    *this = true;

  return *this;
}

FdSet::ConstRef::ConstRef(const fd_set* set, int fd)
    : m_set(set),
      m_fd(fd) {
}

FdSet::ConstRef::operator bool() const {
  return FD_ISSET(m_fd, m_set);
}

void FdSet::clear() {
  FD_ZERO(this);
}

FdSet::Ref FdSet::operator[](unsigned int fd) {
  return Ref(this, fd);
}

const FdSet::ConstRef FdSet::operator[](unsigned int fd) const {
  return ConstRef(this, fd);
}

}  // namespace nix

