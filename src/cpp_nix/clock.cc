/*
 *  Copyright (C) 2015 Josh Bialkowski (josh.bialkowski@gmail.com)
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
 *  @date Apr 2, 2015
 *  @author Josh Bialkowski <josh.bialkowski@gmail.com>
 */

#include <cpp_nix/clock.h>

namespace nix {

Timespec Clock::GetRes() const {
  Timespec output;
  int result = clock_getres(m_clock_id, &output);
  if (!result) {
    return output;
  } else {
    return Timespec { result, 0 };
  }
}

Timespec Clock::GetTime() const {
  Timespec output;
  int result = clock_gettime(m_clock_id, &output);
  if (!result) {
    return output;
  } else {
    return Timespec { result, 0 };
  }
}

int Clock::SetTime(const timespec& time) {
  return clock_settime(m_clock_id, &time);
}

}  // namespace nix
