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

#ifndef MPBLOCKS_DUBINS_CURVES_EIGEN_SOLUTION_RSL_HPP_
#define MPBLOCKS_DUBINS_CURVES_EIGEN_SOLUTION_RSL_HPP_

#include <mpblocks/dubins/curves_eigen/solver.h>
#include <mpblocks/dubins/curves/funcs.hpp>
#include <mpblocks/dubins/curves_eigen/funcs.hpp>

namespace     mpblocks {
namespace       dubins {
namespace curves_eigen {

/// interface for different solutions
template <typename Scalar>
struct Solver<RSL, Scalar> {
  typedef Eigen::Matrix<Scalar, 3, 1> Vector3d;
  typedef Eigen::Matrix<Scalar, 2, 1> Vector2d;

  static Path<Scalar> solve(const Vector3d& q0, const Vector3d& q1,
                        const Scalar r) {
    Path<Scalar> out(RSL);
    Vector2d c[2];
    Vector3d s;

    // calculate the center of the circle to which q1 is tangent
    c[0] = rightCenter(q0, r);

    // calculate the center of the circle to which q2 is tangent
    c[1] = leftCenter(q1, r);

    // find the vector between the two
    Vector2d v = c[1] - c[0];

    // calculate the distance between the two
    Scalar d = v.norm();

    // if they overlap then this primitive has no solution and is not the
    // optimal primitive
    if (d < 2 * r) {
      return out;
    }

    // find the angle between the line through centers and the radius that
    // points to the tangent on the circle
    Scalar a = std::acos(2 * r / d);

    // the distance of the straight segment
    s[1] = d * std::sin(a);

    // this is the angle of dc
    Scalar b = std::atan2(v[1], v[0]);

    // the arc location of the tanget point gives us the first arc length
    Scalar t0 = rightAngleOf(q0);
    Scalar t1 = clampRadian(b + a);
    s[0] = cwArc(t0, t1);

    // the second arc length
    t0 = clampRadian(t1 + M_PI);
    t1 = leftAngleOf(q1);
    s[2] = ccwArc(t0, t1);

    out = s;
    return out;
  }
};

}  // curves_eigen
}  // dubins
}  // mpblocks

#endif  // MPBLOCKS_DUBINS_CURVES_EIGEN_SOLUTION_RSL_HPP_
