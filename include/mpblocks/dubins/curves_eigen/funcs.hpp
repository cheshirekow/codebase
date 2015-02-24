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

#ifndef MPBLOCKS_DUBINS_CURVES_EIGEN_FUNCS_HPP_
#define MPBLOCKS_DUBINS_CURVES_EIGEN_FUNCS_HPP_

#include <cmath>
#include <mpblocks/dubins/curves_eigen/funcs.h>
#include <mpblocks/dubins/curves/funcs.hpp>

namespace     mpblocks {
namespace       dubins {
namespace curves_eigen {

/// return the center of a counter clockwise (left) circle coincident to
/// q with radius r
template <typename Scalar>
Eigen::Matrix<Scalar, 2, 1> ccwCenter(const Eigen::Matrix<Scalar, 3, 1>& q,
                                      Scalar r) {
  typedef Eigen::Matrix<Scalar, 2, 1> Vector2d;
  Vector2d x, v;

  // calculate the center of the circle to which q1 is tangent
  x << q[0], q[1];
  v << -sin(q[2]), cos(q[2]);
  return x + r * v;
}

/// return the center of a counter clockwise (left) circle coincident to
/// q with radius r
template <typename Scalar>
Eigen::Matrix<Scalar, 2, 1> leftCenter(const Eigen::Matrix<Scalar, 3, 1>& q,
                                       Scalar r) {
  return ccwCenter(q, r);
}

/// return the center of a clockwise (right) circle coincident to
/// q with radius r
template <typename Scalar>
Eigen::Matrix<Scalar, 2, 1> cwCenter(const Eigen::Matrix<Scalar, 3, 1>& q,
                                     Scalar r) {
  typedef Eigen::Matrix<Scalar, 2, 1> Vector2d;
  Vector2d x, v, dc;

  x << q[0], q[1];
  v << sin(q[2]), -cos(q[2]);

  return x + r * v;
}

/// return the center of a clockwise (right) circle coincident to
/// q with radius r
template <typename Scalar>
Eigen::Matrix<Scalar, 2, 1> rightCenter(const Eigen::Matrix<Scalar, 3, 1>& q,
                                        Scalar r) {
  return cwCenter(q, r);
}

template <typename Scalar>
Scalar ccwAngleOf(const Scalar q_theta) {
  return clampRadian(q_theta - M_PI / 2.0);
}

template <typename Scalar>
Scalar leftAngleOf(const Scalar q_theta) {
  return ccwAngleOf(q_theta);
}

template <typename Scalar>
Scalar ccwAngle_inv(const Scalar alpha) {
  return clampRadian(alpha + M_PI / 2.0);
}

template <typename Scalar>
Scalar leftAngle_inv(const Scalar alpha) {
  return ccwAngle_inv(alpha);
}

template <typename Scalar>
Scalar cwAngleOf(const Scalar q_theta) {
  return clampRadian(q_theta + M_PI / 2.0);
}

template <typename Scalar>
Scalar rightAngleOf(const Scalar q_theta) {
  return cwAngleOf(q_theta);
}

template <typename Scalar>
Scalar cwAngle_inv(const Scalar alpha) {
  return clampRadian(alpha - M_PI / 2.0);
}

template <typename Scalar>
Scalar rightAngle_inv(const Scalar alpha) {
  return cwAngle_inv(alpha);
}

template <typename Scalar>
Scalar ccwAngleOf(const Eigen::Matrix<Scalar, 3, 1>& q) {
  return clampRadian(q[2] - M_PI / 2.0);
}

template <typename Scalar>
Scalar leftAngleOf(const Eigen::Matrix<Scalar, 3, 1>& q) {
  return ccwAngleOf(q);
}

template <typename Scalar>
Scalar cwAngleOf(const Eigen::Matrix<Scalar, 3, 1>& q) {
  return clampRadian(q[2] + M_PI / 2.0);
}

template <typename Scalar>
Scalar rightAngleOf(const Eigen::Matrix<Scalar, 3, 1>& q) {
  return cwAngleOf(q);
}

/// returns the center of a circle which is coincident to the two circles
/// whose centers are given, where all three circles have the same radius
template <typename Scalar>
bool coincidentCenterA(const Eigen::Matrix<Scalar, 2, 1>& c0,
                       const Eigen::Matrix<Scalar, 2, 1>& c1,
                       Eigen::Matrix<Scalar, 2, 1>& c2, Scalar r, Scalar& a) {
  // distance between two circles
  Scalar d = (c0 - c1).norm();

  // if the distance is too large, then this primitive is not the solution,
  // and we can bail here
  if (d > 4 * r) return false;

  // the base angle of the isosceles triangle whose vertices are the centers
  // of the the three circles, note acos returns [0,pi]
  a = acos(d / (4 * r));

  // create a clockwise rotation of magnitude alpha
  Eigen::Rotation2D<Scalar> R(-a);

  // we find the third vertex of this triangle by taking the vector between
  // the two circle centers, normalizing it, and rotating it by alpha, and
  // scaling it to magnitude 2r, then it points from the center of
  // one the circle tangent to q1 to the third vertex
  c2 = c0 + R * (c1 - c0).normalized() * 2 * r;

  return true;
}

/// returns the center of a circle which is coincident to the two circles
/// whose centers are given, where all three circles have the same radius
template <typename Scalar>
bool coincidentCenterB(const Eigen::Matrix<Scalar, 2, 1>& c0,
                       const Eigen::Matrix<Scalar, 2, 1>& c1,
                       Eigen::Matrix<Scalar, 2, 1>& c2, Scalar r, Scalar& a) {
  // distance between two circles
  Scalar d = (c0 - c1).norm();

  // if the distance is too large, then this primitive is not the solution,
  // and we can bail here
  if (d > 4 * r) return false;

  // the base angle of the isosceles triangle whose vertices are the centers
  // of the the three circles, note acos returns [0,pi]
  a = acos(d / (4 * r));

  // create a clockwise rotation of magnitude alpha
  Eigen::Rotation2D<Scalar> R(-a);

  // we find the third vertex of this triangle by taking the vector between
  // the two circle centers, normalizing it, and rotating it by alpha, and
  // scaling it to magnitude 2r, then it points from the center of
  // one the circle tangent to q1 to the third vertex
  c2 = c1 + R * (c0 - c1).normalized() * 2 * r;

  return true;
}

}  // curves_eigen
}  // dubins
}  // mpblocks

#endif  // FUNCS_H_
