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

#ifndef CPP_NIX_TIMESPEC_H_
#define CPP_NIX_TIMESPEC_H_

#include <sys/time.h>

timespec operator+(const timespec& a, const timespec& b);
timespec operator-(const timespec& a, const timespec& b);

namespace nix {

class Timespec;

/// just a c timeval
class Timespec : public timespec {
 public:
  Timespec(const timespec& other)
      : timespec(other) {
  }

  Timespec(int sec = 0, long int nsec = 0)
      : timespec { sec, nsec } {
  }

  Timespec& operator+=(const Timespec& other) {
    return (*this) = (*this) + other;
  }

  Timespec& operator-=(const Timespec& other) {
    return (*this) = (*this) - other;
  }
};

}  // namespace nix

inline bool operator==(const timespec& a, const timespec& b) {
  return a.tv_sec == b.tv_sec && a.tv_nsec == b.tv_nsec;
}

inline bool operator!=(const timespec& a, const timespec& b) {
  return !(a == b);
}

inline bool operator<(const timespec& a, const timespec& b) {
  if (a.tv_sec < b.tv_sec) {
    return true;
  } else if (a.tv_sec > b.tv_sec) {
    return false;
  } else {
    return a.tv_nsec < b.tv_nsec;
  }
}

inline bool operator>(const timespec& a, const timespec& b) {
  if (a.tv_sec > b.tv_sec) {
    return true;
  } else if (a.tv_sec < b.tv_sec) {
    return false;
  } else {
    return a.tv_nsec > b.tv_nsec;
  }
}

inline bool operator<=(const timespec& a, const timespec& b) {
  if (a.tv_sec < b.tv_sec) {
    return true;
  } else if (a.tv_sec > b.tv_sec) {
    return false;
  } else {
    return a.tv_nsec <= b.tv_nsec;
  }
}

inline bool operator>=(const timespec& a, const timespec& b) {
  if (a.tv_sec > b.tv_sec) {
    return true;
  } else if (a.tv_sec < b.tv_sec) {
    return false;
  } else {
    return a.tv_nsec >= b.tv_nsec;
  }
}


#endif // TIMESPEC_H_
