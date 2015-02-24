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

#ifndef CPP_NIX_FDSET_H_
#define CPP_NIX_FDSET_H_

#include <vector>
#include <cstdlib>
#include <bitset>
#include <unistd.h>

namespace nix {

/// an fd_set with array accessors
class FdSet : public fd_set {
 public:
  /// reference to an element in a bitset
  class Ref {
   private:
    fd_set* m_set;
    int m_fd;

    friend class FdSet;
    Ref(fd_set* set, int fd);

   public:
    operator bool() const;   ///< convert to bool
    Ref& operator=(bool x);  ///< assign from bool
    Ref& flip();             ///< flip the bit
  };

  /// reference to an element in a bitset
  class ConstRef {
   private:
    const fd_set* m_set;
    int m_fd;

    friend class FdSet;
    ConstRef(const fd_set* set, int fd);

   public:
    operator bool() const;          ///< convert to bool
  };

 public:
  /// clears (zeros) the set
  void clear();

  /// array operator
  Ref operator[](unsigned int fd);
  const ConstRef operator[](unsigned int fd) const;
};

}  // namespace nix

#endif // FDSET_H_
