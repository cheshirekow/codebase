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

#ifndef MPBLOCKS_DUBINS_CURVES_EIGEN_HYPER_TWO_THETA_X_RSR_H_
#define MPBLOCKS_DUBINS_CURVES_EIGEN_HYPER_TWO_THETA_X_RSR_H_

namespace     mpblocks {
namespace       dubins {
namespace curves_eigen {
namespace        hyper {
namespace  two_theta_x {

/// specialization for a single theta constraint (min or max)
template <int xSpec, int tSpec, typename Format_t>
struct Solver<xSpec, tSpec, RSR, Format_t> {
  typedef Eigen::Matrix<Format_t, 3, 1> Vector3d_t;
  typedef Eigen::Matrix<Format_t, 2, 1> Vector2d_t;
  typedef Path<Format_t> Result_t;
  typedef HyperRect<Format_t> Hyper_t;

  static const unsigned int SPEC = SpecPack<xSpec, OFF, tSpec>::Result;

  static Result_t solve(const Vector3d_t& q0, const Hyper_t& h,
                        const Format_t r) {
    Result_t out[3];
    for (int i = 0; i < 3; i++) {
      out[i].id = RSR;
    }

    // calculate the center of the turn
    const Vector2d_t c0 = rightCenter(q0, r);

    // the goal point (we'll vary the x coordinate)
    Vector3d_t q1;
    q1 << get_constraint<xSpec, 0>(h), q0[1], get_constraint<tSpec, 2>(h);

    // the center of the goal point (we'll vary the x coordinate)
    Vector2d_t c1 = rightCenter(q1, r);

    // vertical distance between centers
    const Format_t dx = c1[0] - c0[0];

    /// intermediates
    Format_t dist, dist_i, d1, arc0, arc1;

    /// the solution minimizing the straight section
    Format_t target;
    if (dx > 0)
      target = 0;
    else
      target = M_PI;
    c1[1] = c0[1];

    d1 = std::abs(dx);
    arc0 = cwArc(q0[2], target);
    arc1 = cwArc(target, q1[2]);
    dist = r * (arc0 + arc1) + d1;

    out[0] = Vector3d_t(arc0, d1, arc1);
    int iBest = 0;

    /// the solution minimizing L0
    /// we have a restriction here b/c minimizing L0 means zero initial
    /// arc length, if we're pointed in the wrong directino, then there is
    /// no zero-arc length solution
    Format_t cosTheta = std::cos(q0[2]);
    if ((dx > 0 && cosTheta > 0) || (dx < 0 && cosTheta < 0)) {
      d1 = dx / cosTheta;
      arc0 = 0;
      arc1 = cwArc(q0[2], q1[2]);
      dist_i = r * (arc0 + arc1) + d1;

      out[1] = Vector3d_t(arc0, d1, arc1);
      if (dist_i < dist) {
        iBest = 1;
        dist = dist_i;
      }
    }

    cosTheta = std::cos(q1[2]);
    if ((dx > 0 && cosTheta > 0) || (dx < 0 && cosTheta < 0)) {
      d1 = dx / cosTheta;
      arc0 = cwArc(q0[2], q1[2]);
      arc1 = 0;
      dist_i = r * (arc0 + arc1) + d1;

      out[2] = Vector3d_t(arc0, d1, arc1);
      if (dist_i < dist) {
        iBest = 2;
        dist = dist_i;
      }
    }

    return out[iBest];
  };
};

}  // namespace two_theta_x
}  // namespace hyper
}  // namespace curves_eigen
}  // namespace dubins
}  // namespace mpblocks

#endif  // MPBLOCKS_DUBINS_CURVES_EIGEN_HYPER_TWO_THETA_X_RSR_H_
