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

#ifndef MPBLOCKS_DUBINS_CURVES_EIGEN_HYPER_TWO_X_Y_H_
#define MPBLOCKS_DUBINS_CURVES_EIGEN_HYPER_TWO_X_Y_H_

namespace     mpblocks {
namespace       dubins {
namespace curves_eigen {
namespace        hyper {

/// specialization for a single theta constraint (min or max)
template <int xSpec, int ySpec, typename Format_t>
struct Solver<xSpec, ySpec, OFF, Format_t> {
  typedef Eigen::Matrix<Format_t, 3, 1> Vector3d_t;
  typedef Eigen::Matrix<Format_t, 2, 1> Vector2d_t;
  typedef Path<Format_t> Result_t;
  typedef HyperRect<Format_t> Hyper_t;

  static const unsigned int SPEC = SpecPack<xSpec, ySpec, OFF>::Result;

  /// get the distance for a left turn
  static Result_t solveLS(const Vector3d_t& q0, const Hyper_t& h,
                          const Format_t r) {
    Result_t out(LSL);

    // calculate the center of the circle to which q0 is tangent
    Vector2d_t c = leftCenter(q0, r);

    // grab the x & y values (the target point)
    Vector2d_t p;
    p << get_constraint<xSpec, 0>(h), get_constraint<ySpec, 1>(h);

    Vector2d_t cp = (p - c);

    // if the solution is less than r from this center, then the Left
    // solution is infeasible
    if (cp.squaredNorm() < r * r) return out;

    // otherwise we turn left until we are pointed in the right direction,
    // then we go straight the rest of the way
    // see 2013-06-07-Note-17-51_dubins_kdtree.xoj for drawings and math
    Format_t d_cp = cp.norm();

    // this is the interior angle of the right triangle (c,t,p) where
    // c is the circle center, p is the target point, and t is a point
    // on the circle such that c->t is perpendicularo to t->p and t->p
    // is tangent to the circle
    Format_t alpha = std::acos(r / d_cp);

    // this is the angle that the vector c->p makes with the circle
    Format_t beta = std::atan2(cp[1], cp[0]);

    // this is the target angle for the tangent point
    Format_t theta1 = clampRadian(beta - alpha);

    // the is our start angle
    Format_t theta0 = leftAngleOf(q0);

    // this is the arc distance we must travel on the circle
    Format_t arc0 = ccwArc(theta0, theta1);

    // this is the length of the segment from t->p
    Format_t d1 = d_cp * std::sin(alpha);

    out = Vector3d_t(arc0, d1, 0);
    return out;
  };

  static Result_t solveRS(const Vector3d_t& q0, const Hyper_t& h,
                          const Format_t r) {
    Result_t out(RSR);

    // calculate the center of the circle to which q0 is tangent
    Vector2d_t c = rightCenter(q0, r);

    // grab the x & y values (the target point)
    Vector2d_t p;
    p << get_constraint<xSpec, 0>(h), get_constraint<ySpec, 1>(h);

    Vector2d_t cp = (p - c);

    // if the solution is less than r from this center, then the Left
    // solution is infeasible
    if (cp.squaredNorm() < r * r) return out;

    // otherwise we turn left until we are pointed in the right direction,
    // then we go straight the rest of the way
    // see 2013-06-07-Note-17-51_dubins_kdtree.xoj for drawings and math
    Format_t d_cp = cp.norm();

    // this is the interior angle of the right triangle (c,t,p) where
    // c is the circle center, p is the target point, and t is a point
    // on the circle such that c->t is perpendicularo to t->p and t->p
    // is tangent to the circle
    Format_t alpha = std::acos(r / d_cp);

    // this is the angle that the vector c->p makes with the circle
    Format_t beta = std::atan2(cp[1], cp[0]);

    // this is the target angle for the tangent point
    Format_t theta1 = clampRadian(beta + alpha);

    // the is our start angle
    Format_t theta0 = rightAngleOf(q0);

    // this is the arc distance we must travel on the circle
    Format_t arc0 = cwArc(theta0, theta1);

    // this is the length of the segment from t->p
    Format_t d1 = d_cp * std::sin(alpha);

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

#endif // MPBLOCKS_DUBINS_CURVES_EIGEN_HYPER_TWO_X_Y_H_
