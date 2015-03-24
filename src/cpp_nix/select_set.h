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

#ifndef CPP_NIX_SELECT_SET_H_
#define CPP_NIX_SELECT_SET_H_

#include <sys/select.h>
#include <map>


namespace nix {

enum {
  SELECT_READ = 0x01,
  SELECT_WRITE = 0x02,
  SELECT_EXCEPT = 0x04
};


typedef std::map<int, int> SelectMap;

class SelectSurrogate {
 private:
  SelectMap* m_select_map;
  int m_fd;

 public:
  SelectSurrogate(SelectMap* select_map, int fd)
      : m_select_map(select_map),
        m_fd(fd) {
  }

  operator int() {
    return (*m_select_map)[m_fd];
  }

  const SelectSurrogate& operator=(int spec) const {
    if ((spec & (SELECT_READ | SELECT_WRITE | SELECT_EXCEPT)) == 0) {
      SelectMap::iterator fd_iter = m_select_map->find(m_fd);
      if (fd_iter != m_select_map->end()) {
        m_select_map->erase(fd_iter);
      }
    } else {
      (*m_select_map)[m_fd] = spec;
    }
    return *this;
  }
};

class SelectSet {
 public:
  SelectSurrogate operator[](int fd) {
    return SelectSurrogate(&m_map, fd);
  }

  int BuildSets(fd_set* read_set, fd_set* write_set, fd_set* except_set) const;

  int Select(fd_set* read_set, fd_set* write_set, fd_set* except_set,
             timeval* timeout) const;

  int Pselect(fd_set* read_set, fd_set* write_set, fd_set* except_set,
              timespec* timeout, sigset_t* sigmask) const;

 private:
  SelectMap m_map;
};

}  // namespace nix

#endif
