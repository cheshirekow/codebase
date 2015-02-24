/*
 *  Copyright (C) 2012 Josh Bialkowski (jbialk@mit.edu)
 *
 *  This file is part of mpblocks.
 *
 *  mpblocks is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  mpblocks is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with mpblocks.  If not, see <http://www.gnu.org/licenses/>.
 */
/**
 *  @file
 *  @date   Aug 4, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 */

#include <mpblocks/util/timespec.h>

timespec operator+(const timespec& a, const timespec& b) {
  timespec result{a.tv_sec + b.tv_sec, a.tv_nsec + b.tv_nsec};
  if (result.tv_nsec >= 1000000000) {
    result.tv_sec++;
    result.tv_nsec -= 1000000000;
  }

  return result;
}

timespec operator-(const timespec& a, const timespec& b) {
  timespec result{a.tv_sec - b.tv_sec, a.tv_nsec - b.tv_nsec};
  if (result.tv_nsec < 0) {
    result.tv_sec--;
    result.tv_nsec += 1000000000;
  }
  return result;
}


namespace mpblocks {
namespace  utility {


Timespec operator+(const Timespec& a, const Timespec& b) {
  Timespec result{a.tv_sec + b.tv_sec, a.tv_nsec + b.tv_nsec};
  if (result.tv_nsec >= 1000000000) {
    result.tv_sec++;
    result.tv_nsec -= 1000000000;
  }

  return result;
}

Timespec operator-(const Timespec& a, const Timespec& b) {
  Timespec result{a.tv_sec - b.tv_sec, a.tv_nsec - b.tv_nsec};
  if (result.tv_nsec < 0) {
    result.tv_sec--;
    result.tv_nsec += 1000000000;
  }
  return result;
}

} // namespace utility
} // namespace mpblocks
