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

#include <cpp_nix/timespec.h>

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
