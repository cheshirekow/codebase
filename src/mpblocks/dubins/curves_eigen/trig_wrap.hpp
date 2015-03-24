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
 *  @date   Nov 5, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief
 */

#ifndef MPBLOCKS_DUBINS_CURVES_EIGEN_TRIGWRAP_HPP_
#define MPBLOCKS_DUBINS_CURVES_EIGEN_TRIGWRAP_HPP_

namespace     mpblocks {
namespace       dubins {
namespace curves_eigen {

template <typename Format_t>
struct TrigWrap<Format_t, L> {
  static Format_t arc(Format_t a, Format_t b) {
    return leftArc(a, b);
  }

  static Eigen::Matrix<Format_t, 2, 1> center(
      const Eigen::Matrix<Format_t, 3, 1>& q, Format_t r) {
    return leftCenter(q, r);
  }

  static Format_t angleOf(const Eigen::Matrix<Format_t, 3, 1>& q) {
    return leftAngleOf(q);
  }

  static Format_t addDirection(Format_t a) {
    return -a;
  }
};

template <typename Format_t>
struct TrigWrap<Format_t, R> {
  static Format_t arc(Format_t a, Format_t b) { return rightArc(a, b); }

  static Eigen::Matrix<Format_t, 2, 1> center(
      const Eigen::Matrix<Format_t, 3, 1>& q, Format_t r) {
    return rightCenter(q, r);
  }

  static Format_t angleOf(const Eigen::Matrix<Format_t, 3, 1>& q) {
    return rightAngleOf(q);
  }

  static Format_t addDirection(Format_t a) {
    return a;
  }
};

}  // curves_eigen
}  // dubins
}  // mpblocks

#endif // TRIGWRAP_H_
