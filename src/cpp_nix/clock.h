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
 *  @date   Jun 22, 2014
 *  @author Josh Bialkowski (josh.bialkowski@gmail.com)
 *  @brief
 */

#ifndef CPP_NIX_CLOCK_H_
#define CPP_NIX_CLOCK_H_

#include <ctime>
#include "timespec.h"

namespace nix {

class Clock {
 public:
  Clock(clockid_t clock_id)
      : m_clock_id(clock_id) {
  }

  timespec GetRes() {
    timespec output;
    int result = clock_getres(m_clock_id, &output);
    if (!result) {
      return output;
    } else {
      return timespec { result, 0 };
    }
  }

  timespec GetTime() {
    timespec output;
    int result = clock_gettime(m_clock_id, &output);
    if (!result) {
      return output;
    } else {
      return timespec { result, 0 };
    }
  }

  int SetTime(const timespec& time){
    return clock_settime(m_clock_id,&time);
  }

 private:
  clockid_t  m_clock_id;
};

}  // namespace nix

#endif  // CPP_NIX_CLOCK_H_
