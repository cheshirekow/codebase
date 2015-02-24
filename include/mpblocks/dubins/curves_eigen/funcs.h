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

#ifndef MPBLOCKS_DUBINS_CURVES_EIGEN_FUNCS_H_
#define MPBLOCKS_DUBINS_CURVES_EIGEN_FUNCS_H_

#include <Eigen/Dense>

namespace mpblocks {
namespace dubins {
namespace curves_eigen {

/// return the center of a counter clockwise (left) circle coincident to
/// q with radius r
template <typename Scalar>
Eigen::Matrix<Scalar, 2, 1> ccwCenter(const Eigen::Matrix<Scalar, 3, 1>& q,
                                      Scalar r);

/// return the center of a counter clockwise (left) circle coincident to
/// q with radius r
template <typename Scalar>
Eigen::Matrix<Scalar, 2, 1> leftCenter(const Eigen::Matrix<Scalar, 3, 1>& q,
                                       Scalar r);

/// return the center of a clockwise (right) circle coincident to
/// q with radius r
template <typename Scalar>
Eigen::Matrix<Scalar, 2, 1> cwCenter(const Eigen::Matrix<Scalar, 3, 1>& q,
                                     Scalar r);

/// return the center of a clockwise (right) circle coincident to
/// q with radius r
template <typename Scalar>
Eigen::Matrix<Scalar, 2, 1> rightCenter(const Eigen::Matrix<Scalar, 3, 1>& q,
                                        Scalar r);

/// return the angle of the vector from the center of the counter clockwise
/// (left) circle coincident to q... to q
template <typename Scalar>
Scalar ccwAngleOf(const Scalar q_theta);

/// return the angle of the vector from the center of the counter clockwise
/// (left) circle coincident to q... to q
template <typename Scalar>
Scalar leftAngleOf(const Scalar q_theta);

/// return the angle of the vector from the center of the clockwise
/// (right) circle coincident to q... to q
template <typename Scalar>
Scalar cwAngleOf(const Scalar q_theta);

/// return the angle of the vector from the center of the clockwise
/// (right) circle coincident to q... to q
template <typename Scalar>
Scalar rightAngleOf(const Scalar q_theta);

/// return the angle of the vector from the center of the counter clockwise
/// (left) circle coincident to q... to q
template <typename Scalar>
Scalar ccwAngleOf(const Eigen::Matrix<Scalar, 3, 1>& q);

/// return the angle of the vector from the center of the counter clockwise
/// (left) circle coincident to q... to q
template <typename Scalar>
Scalar leftAngleOf(const Eigen::Matrix<Scalar, 3, 1>& q);

/// return the angle of the vector from the center of the clockwise
/// (right) circle coincident to q... to q
template <typename Scalar>
Scalar cwAngleOf(const Eigen::Matrix<Scalar, 3, 1>& q);

/// return the angle of the vector from the center of the clockwise
/// (right) circle coincident to q... to q
template <typename Scalar>
Scalar rightAngleOf(const Eigen::Matrix<Scalar, 3, 1>& q);

/// returns the center of a circle which is coincident to the two circles
/// whose centers are given, where all three circles have the same radius
template <typename Scalar>
bool coincidentCenterA(const Eigen::Matrix<Scalar, 2, 1>& c0,
                       const Eigen::Matrix<Scalar, 2, 1>& c1,
                       Eigen::Matrix<Scalar, 2, 1>& c2, Scalar r, Scalar& a);

/// returns the center of a circle which is coincident to the two circles
/// whose centers are given, where all three circles have the same radius
template <typename Scalar>
bool coincidentCenterB(const Eigen::Matrix<Scalar, 2, 1>& c0,
                       const Eigen::Matrix<Scalar, 2, 1>& c1,
                       Eigen::Matrix<Scalar, 2, 1>& c2, Scalar r, Scalar& a);

}  // curves_eigen
}  // dubins
}  // mpblocks

#endif  // FUNCS_H_
