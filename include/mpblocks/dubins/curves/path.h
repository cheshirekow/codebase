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
 *  @date   Jul 5, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_DUBINS_CURVES_PATH_H_
#define MPBLOCKS_DUBINS_CURVES_PATH_H_

#include <Eigen/Dense>
#include <mpblocks/dubins/curves/types.h>

namespace mpblocks {
namespace   dubins {

/// Encodes a dubins path primitive, which is three connected arc segments
/**
 *  Note: one of the segments may be a line segment
 *  (an arc with infinite radius). In this case the value of 's' for the line
 *  segment is the length of the segment, not an arc-length.
 */
template <typename Format_t>
struct Path {
  typedef Eigen::Matrix<Format_t, 3, 1> Vector3d;

  int id;      ///< identifies the type of path
  bool f;      ///< is feasible
  Vector3d s;  ///< lengths of each segment, how it's interpreted
               ///  depends on id

  Path(const Path<Format_t>& other) {
    id = other.id;
    f = other.f;
    s = other.s;
  }

  Path() : id(INVALID), f(false) { s.fill(0); }

  /// the default constructor marks it as infeasible, so we can return it
  /// immediately

  Path(int id) : id(id), f(false) { s.fill(0); }

  /// we only use this constructor when it's feasible

  Path(int id, const Vector3d& s) : id(id), f(true), s(s) { s.fill(0); }

  /// fill and set to feasible, returns itself so that we can do the
  /// return in one line

  Path<Format_t>& operator=(const Vector3d& s_in) {
    f = true;
    s = s_in;
    return *this;
  }

  Path<Format_t>& operator=(const Path<Format_t>& other) {
    f = other.f;
    s = other.s;
    id = other.id;
    return *this;
  }

  /// compute distance based on id
  Format_t dist(const Format_t r) const {
    if (id < 4)
      return r * s.sum();
    else
      return r * (s[0] + s[2]) + s[1];
  }

  /// because I'm lazy
  operator bool() const { return f; }
};

template <typename Format_t>
Path<Format_t> bestOf(const Path<Format_t>& r0, const Path<Format_t>& r1,
                      const Format_t r) {
  if (r0.f && r1.f) {
    if (r0.dist(r) < r1.dist(r))
      return r0;
    else
      return r1;
  } else if (r0.f)
    return r0;
  else if (r1.f)
    return r1;
  else
    return Path<Format_t>();
}

} // dubins
} // mpblocks

#endif // MPBLOCKS_DUBINS_CURVES_PATH_H_
