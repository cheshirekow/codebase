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
 *  @date   Jun 27, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief
 */

#ifndef MPBLOCKS_DUBINS_CURVES_EIGEN_HYPER_ONE_Y_H_
#define MPBLOCKS_DUBINS_CURVES_EIGEN_HYPER_ONE_Y_H_

namespace     mpblocks {
namespace       dubins {
namespace curves_eigen {
namespace        hyper {

/// specialization for a single y constraint (min or max)
template <int ySpec, typename Format_t>
struct Solver<OFF, ySpec, OFF, Format_t> {
  typedef Eigen::Matrix<Format_t, 3, 1> Vector3d_t;
  typedef Eigen::Matrix<Format_t, 2, 1> Vector2d_t;
  typedef Path<Format_t> Result_t;
  typedef HyperRect<Format_t> Hyper_t;

  static const unsigned int SPEC = SpecPack<OFF, ySpec, OFF>::Result;

  /// get the distance for a left turn
  static Result_t solveLS(const Vector3d_t& q0, const Hyper_t& h,
                          const Format_t r) {
    Result_t out(LSL);
    const Format_t _PI = static_cast<Format_t>(M_PI);

    // calculate the center of the circle to which q0 is tangent
    Vector2d_t x, v, c;
    x << q0[0], q0[1];
    v << -std::sin(q0[2]), std::cos(q0[2]);
    c = x + r * v;

    // grab the X value
    Format_t Y = get_constraint<ySpec, 1>(h);

    // if the solution is less than r from this center, then there is no
    // straight section and we simply turn left until we hit x = x
    if (std::abs(c[1] - Y) < r) {
      Format_t theta1 = std::asin((Y - c[1]) / r);
      Format_t theta0 = leftAngleOf(q0);
      Format_t dist0 = ccwArc(theta0, theta1);
      Format_t dist1;
      if (theta1 > 0)
        dist1 = ccwArc(theta0, _PI - theta1);
      else
        dist1 = ccwArc(theta0, -_PI - theta1);
      Format_t arc0 = std::min(dist0, dist1);

      out = Vector3d_t(arc0, 0, 0);
      return out;
    }

    // otherwise we turn left until we are perpendicular to the constraint
    // and pointed in the right direction, then we go straight the rest of
    // the way

    // if the constraint is right of the center then our target is -_PI/2.
    // otherwise it's _PI/2;
    Format_t target = _PI;
    if (Y > c[1]) target = 0;

    // we turn until we hit this point
    Format_t theta0 = leftAngleOf(q0);
    Format_t arc0 = ccwArc(theta0, target);

    // then we drive straight until we hit the constraint
    Format_t d1 = std::abs(Y - c[1]);

    out = Vector3d_t(arc0, d1, 0);
    return out;
  };

  static Result_t solveRS(const Vector3d_t& q0, const Hyper_t& h,
                          const Format_t r) {
    Result_t out(RSR);
    const Format_t _PI = static_cast<Format_t>(M_PI);

    // calculate the center of the circle to which q0 is tangent
    Vector2d_t x, v, c;
    x << q0[0], q0[1];
    v << std::sin(q0[2]), -std::cos(q0[2]);
    c = x + r * v;

    // grab the X value
    Format_t Y = get_constraint<ySpec, 1>(h);

    // if the solution is less than r from this center, then there is no
    // straight section and we simply turn left until we hit x = x
    if (std::abs(c[1] - Y) < r) {
      Format_t theta1 = std::asin((Y - c[1]) / r);
      Format_t theta0 = rightAngleOf(q0);
      Format_t dist0 = cwArc(theta0, theta1);
      Format_t dist1;
      if (theta1 > 0)
        dist1 = cwArc(theta0, _PI - theta1);
      else
        dist1 = cwArc(theta0, -_PI - theta1);
      Format_t arc0 = dist0;
      std::min(dist0, dist1);

      out = Vector3d_t(arc0, 0, 0);
      return out;
    }

    // otherwise we turn left until we are perpendicular to the constraint
    // and pointed in the right direction, then we go straight the rest of
    // the way

    // if the constraint is right of the center then our target is -_PI/2.
    // otherwise it's _PI/2;
    Format_t target = 0;
    if (Y > c[1]) target = _PI;

    // we turn until we hit this point
    Format_t theta0 = rightAngleOf(q0);
    Format_t arc0 = cwArc(theta0, target);

    // then we drive straight until we hit the constraint
    Format_t d1 = std::abs(Y - c[1]);

    out = Vector3d_t(arc0, d1, 0);
    return out;
  };

  static Result_t solve(const Vector3d_t& q0, const Hyper_t& h,
                        const Format_t r) {
    return bestOf(solveLS(q0, h, r), solveRS(q0, h, r), r);
  };
};

}  // namespace hyper
}  // namespace curves_eigen
}  // namespace dubins
}  // namespace mpblocks

#endif  // MPBLOCKS_DUBINS_CURVES_EIGEN_HYPER_ONE_Y_H_
