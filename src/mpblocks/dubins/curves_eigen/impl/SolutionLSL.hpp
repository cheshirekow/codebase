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
 *  @date   Oct 30, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief
 */

#ifndef MPBLOCKS_DUBINS_CURVES_EIGEN_SOLUTION_LSL_HPP_
#define MPBLOCKS_DUBINS_CURVES_EIGEN_SOLUTION_LSL_HPP_

#include <mpblocks/dubins/curves_eigen/solver.h>
#include <mpblocks/dubins/curves/funcs.hpp>
#include <mpblocks/dubins/curves_eigen/funcs.hpp>

namespace     mpblocks {
namespace       dubins {
namespace curves_eigen {

/// solves a left turn, plus a straight segment, plus a left turn
template <typename Scalar>
struct Solver<LSL, Scalar> {
  typedef Eigen::Matrix<Scalar, 3, 1> Vector3d;
  typedef Eigen::Matrix<Scalar, 2, 1> Vector2d;

  static Path<Scalar> solve(const Vector3d& q0, const Vector3d& q1,
                        const Scalar r) {
    Path<Scalar> out(LSL);  ///< output
    Vector2d c[2];    ///< centers of the circles
    Vector3d s;       ///< segment lengths

    // calculate the center of the circle to which q1 is tangent
    c[0] = leftCenter(q0, r);

    // calculate the center of the circle to which q2 is tangent
    c[1] = leftCenter(q1, r);

    // find the vector between the two
    Vector2d v = c[1] - c[0];

    // the length of the straight segment is the length of this vector
    s[1] = v.norm();

    // the angle of that vector is the target angle of the vehicle
    Scalar t0 = q0[2];
    Scalar t1 = std::atan2(v[1], v[0]);
    Scalar t2 = q1[2];
    s[0] = ccwArc(t0, t1);
    s[2] = ccwArc(t1, t2);

    out = s;
    return out;
  }
};

}  // curves_eigen
}  // dubins
}  // mpblocks

#endif  // MPBLOCKS_DUBINS_CURVES_EIGEN_SOLUTION_LSL_HPP_
