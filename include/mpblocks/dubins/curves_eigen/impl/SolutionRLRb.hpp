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

#ifndef MPBLOCKS_DUBINS_CURVES_EIGEN_SOLUTION_RLR_B_HPP_
#define MPBLOCKS_DUBINS_CURVES_EIGEN_SOLUTION_RLR_B_HPP_

#include <mpblocks/dubins/curves_eigen/solver.h>
#include <mpblocks/dubins/curves/funcs.hpp>
#include <mpblocks/dubins/curves_eigen/funcs.hpp>

namespace     mpblocks {
namespace       dubins {
namespace curves_eigen {

/// solves a right turn, plus a left turn, plus a right turn
/**
 *
                  *  *
               *        *
              *          *
        *o-*  *          *  *o-*
     *        xx        xx        *
    *          *  x  x  *          *
    *          *        *          *
     *        *          *        *
        *  *                *  *
 */
template <typename Scalar>
struct Solver<RLRb, Scalar> {
  typedef Eigen::Matrix<Scalar, 3, 1> Vector3d;
  typedef Eigen::Matrix<Scalar, 2, 1> Vector2d;

  static Path<Scalar> solve(const Vector3d& q0, const Vector3d& q1,
                        const Scalar r) {
    Path<Scalar> out(RLRb);
    Vector2d c[3];
    Vector3d s;

    // calculate the center of the circle to which q1 is tangent
    c[0] = rightCenter(q0, r);

    // calculate the center of the circle to which q2 is tangent
    c[2] = rightCenter(q1, r);

    // the distance between the centers of these two circles
    Scalar d = (c[0] - c[2]).norm();

    // if the distance is too large, then this primitive is not the solution,
    // and we can bail here
    if (d > 4 * r) {
      return out;
    }

    // if the distance is zero then the geometry degenerates
    if (d == 0) {
      out = Vector3d(0, 0, 0);
      return out;
    }

    // the base angle of the isosceles triangle whose vertices are the centers
    // of the the three circles, note acos returns [0,pi]
    Scalar a = acos(d / (4 * r));

    // create a clockwise rotation of magnitude alpha
    Eigen::Rotation2D<Scalar> R(-a);

    // we find the third vertex of this triangle by taking the vector between
    // the two circle centers, normalizing it, and rotating it by alpha, and
    // scaling it to magnitude 2r, then it points from the center of
    // one the circle tangent to q1 to the third vertex
    c[1] = c[2] + R * (c[0] - c[2]).normalized() * 2 * r;

    // calculate the arc distance we travel on the first circle
    Vector2d dc = c[1] - c[0];     //< vector between centers of circles
    Scalar a0 = rightAngleOf(q0);  //< angle of vector from center to q1
    Scalar a1 = std::atan2(dc[1], dc[0]);  //< angle of that vector
    s[0] = cwArc(a0, a1);                    //< ccwise distance

    // calculate the arc distance we travel on the third circle
    s[1] = M_PI - 2 * a;

    // calculate the arc distance we travel on the second circle
    dc = c[1] - c[2];               //< vector between centers of circles
    a0 = std::atan2(dc[1], dc[0]);  //< angle of that vector
    a1 = rightAngleOf(q1);          //< angle of vector from center to q1
    s[2] = cwArc(a0, a1);           //< ccwise distance

    out = s;
    return out;
  }
};

}  // curves_eigen
}  // dubins
}  // mpblocks

#endif  // MPBLOCKS_DUBINS_CURVES_EIGEN_SOLUTION_RLR_B_HPP_
