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

#include <cpp_nix/select_set.h>

#include <sys/select.h>
#include <vector>

namespace nix {

int SelectSet::BuildSets(fd_set* read_set, fd_set* write_set,
                         fd_set* except_set) const {
  int nfds = 0;

  std::vector<fd_set*> all_sets( { read_set, write_set, except_set });
  for (fd_set* set : all_sets) {
    if (set != NULL) {
      FD_ZERO(set);
    }
  }

  std::map<int, fd_set*> mask_map({
    { SELECT_READ, read_set },
    { SELECT_WRITE, write_set },
    { SELECT_EXCEPT, except_set }
  });

  for (const auto& mask_and_set : mask_map) {
    int mask = mask_and_set.first;
    fd_set* set = mask_and_set.second;
    if (set == NULL) {
      continue;
    }

    for (const auto& fd_and_spec : m_map) {
      int fd = fd_and_spec.first;
      int spec = fd_and_spec.second;
      if (spec & mask) {
        if(fd > nfds){
          nfds = fd;
        }
        FD_SET(fd, set);
      }
    }
  }
  return nfds+1;
}

int SelectSet::Select(fd_set* read_set, fd_set* write_set, fd_set* except_set,
                      timeval* timeout) const {
  int nfds = BuildSets(read_set,write_set,except_set);
  return select(nfds,read_set,write_set,except_set,timeout);
}

int SelectSet::Pselect(fd_set* read_set, fd_set* write_set, fd_set* except_set,
                       timespec* timeout, sigset_t* sigmask) const {
  int nfds = BuildSets(read_set, write_set, except_set);
  return pselect(nfds, read_set, write_set, except_set, timeout, sigmask);
}



}  // namespace nix
