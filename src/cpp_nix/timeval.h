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

#ifndef CPP_NIX_TIMEVAL_H_
#define CPP_NIX_TIMEVAL_H_

#include <sys/time.h>

timeval operator+(const timeval& a, const timeval& b);
timeval operator-(const timeval& a, const timeval& b);

namespace nix {

class Timeval;

class Timeval : public timeval {
 public:
  Timeval(const timeval& other)
      : timeval(other) {
  }

  Timeval(int sec = 0, int usec = 0)
      : timeval { sec, usec } {
  }

  Timeval& operator+=(const Timeval& other) {
    return (*this) = (*this) + other;
  }

  Timeval& operator-=(const Timeval& other) {
    return (*this) = (*this) - other;
  }
};

}  // namespace nix

inline bool operator==(const timeval& a, const timeval& b) {
  return a.tv_sec == b.tv_sec && a.tv_usec == b.tv_usec;
}

inline bool operator!=(const timeval& a, const timeval& b) {
  return !(a == b);
}

inline bool operator<(const timeval& a, const timeval& b) {
  if (a.tv_sec < b.tv_sec) {
    return true;
  } else if (a.tv_sec > b.tv_sec) {
    return false;
  } else {
    return a.tv_usec < b.tv_usec;
  }
}

inline bool operator>(const timeval& a, const timeval& b) {
  if (a.tv_sec > b.tv_sec) {
    return true;
  } else if (a.tv_sec < b.tv_sec) {
    return false;
  } else {
    return a.tv_usec > b.tv_usec;
  }
}

inline bool operator<=(const timeval& a, const timeval& b) {
  if (a.tv_sec < b.tv_sec) {
    return true;
  } else if (a.tv_sec > b.tv_sec) {
    return false;
  } else {
    return a.tv_usec <= b.tv_usec;
  }
}

inline bool operator>=(const timeval& a, const timeval& b) {
  if (a.tv_sec > b.tv_sec) {
    return true;
  } else if (a.tv_sec < b.tv_sec) {
    return false;
  } else {
    return a.tv_usec >= b.tv_usec;
  }
}

#endif // TIMEVAL_H_
