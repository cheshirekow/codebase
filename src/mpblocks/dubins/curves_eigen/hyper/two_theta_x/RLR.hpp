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

#ifndef MPBLOCKS_DUBINS_CURVES_EIGEN_HYPER_TWO_THETA_X_RLR_H_
#define MPBLOCKS_DUBINS_CURVES_EIGEN_HYPER_TWO_THETA_X_RLR_H_

namespace     mpblocks {
namespace       dubins {
namespace curves_eigen {
namespace        hyper {
namespace  two_theta_x {

/// specialization for a single theta constraint (min or max)
template <int xSpec, int tSpec, typename Format_t>
struct Solver<xSpec, tSpec, RLRa, Format_t> {
  typedef Eigen::Matrix<Format_t, 3, 1> Vector3d_t;
  typedef Eigen::Matrix<Format_t, 2, 1> Vector2d_t;
  typedef Path<Format_t> Result_t;
  typedef HyperRect<Format_t> Hyper_t;

  static const unsigned int SPEC = SpecPack<xSpec, OFF, tSpec>::Result;

  static Result_t solve(const Vector3d_t& q0, const Hyper_t& h,
                        const Format_t r) {
    Result_t out[5];
    for (int i = 0; i < 5; i++) {
      out[i].id = RLRa;
    }

    // calculate the center of the right turn
    const Vector2d_t c0 = rightCenter(q0, r);

    // the goal point (we'll vary the y coordinate)
    Vector3d_t q1;
    q1 << get_constraint<xSpec, 0>(h), q0[1], get_constraint<tSpec, 2>(h);

    // the center of the goal point (again, we'll vary the y coordinate)
    Vector2d_t c2 = rightCenter(q1, r);

    // the arc location of the query and goal points with respect to
    // their circle
    const Format_t alpha0 = rightAngleOf(q0);
    const Format_t alpha1 = rightAngleOf(q1);

    // vertical distance between centers
    Format_t dx = c2[0] - c0[0];
    Format_t eps = static_cast<Format_t>(1e-6);

    // if the two centers are more than 4r away from each other than
    // there is no feasible LRL trajectory
    if (std::abs(dx) + eps > 4 * r) return out[0];

    // the solution that minimizes the center arc requires that the
    // two circles be y-aligned
    c2[1] = c0[1];

    // this is the location of the second circle with respect to the
    // first two
    Format_t dy = std::sqrt(4 * r * r - (dx / 2) * (dx / 2));
    Vector2d_t c1;
    c1[0] = (c2[0] + c0[0]) / 2;
    if (c2[0] > c0[0])
      c1[1] = c0[1] + dy;
    else
      c1[1] = c0[1] - dy;

    Vector2d_t dc01 = c1 - c0;
    Vector2d_t dc12 = c2 - c1;
    Format_t beta0 = alpha0;
    Format_t beta1 = clampRadian(std::atan2(dc01[1], dc01[0]));
    Format_t beta2 = clampRadian(beta1 + M_PI);
    Format_t beta3 = clampRadian(std::atan2(dc12[1], dc12[0]));
    Format_t beta4 = clampRadian(beta3 + M_PI);
    Format_t beta5 = alpha1;
    Format_t arc0 = cwArc(beta0, beta1);
    Format_t arc1 = ccwArc(beta2, beta3);
    Format_t arc2 = cwArc(beta4, beta5);
    Format_t dist = r * (arc0 + arc1 + arc2);

    out[0] = Vector3d_t(arc0, arc1, arc2);
    int iBest = 0;
    Format_t dist_i = 0;

    // for the solution that minimizes L0, this requires an immediate
    // R (i.e. arc0 is 0), so find c1
    c1 = leftCenter(q0, r);

    // in order for a solution of this type to exist c1 must be within
    // 2r vertical of c2
    dx = c2[0] - c1[0];
    if (std::abs(dx) + eps < 2 * r) {
      // there are two solutions +dx and -dx
      dy = std::sqrt(4 * r * r - dx * dx);

      // for the +dy solution
      c2[1] = c1[1] + dy;

      dc01 = c1 - c0;
      dc12 = c2 - c1;
      beta0 = alpha0;
      beta1 = clampRadian(std::atan2(dc01[1], dc01[0]));
      beta2 = clampRadian(beta1 + M_PI);
      beta3 = clampRadian(std::atan2(dc12[1], dc12[0]));
      beta4 = clampRadian(beta3 + M_PI);
      beta5 = alpha1;
      arc0 = 0;
      arc1 = ccwArc(beta2, beta3);
      arc2 = cwArc(beta4, beta5);
      dist_i = r * (arc0 + arc1 + arc2);

      out[1] = Vector3d_t(arc0, arc1, arc2);
      if (dist_i < dist) {
        iBest = 1;
        dist = dist_i;
      }

      // for the -dx solution
      c2[1] = c1[1] - dy;

      dc01 = c1 - c0;
      dc12 = c2 - c1;
      beta0 = alpha0;
      beta1 = clampRadian(std::atan2(dc01[1], dc01[0]));
      beta2 = clampRadian(beta1 + M_PI);
      beta3 = clampRadian(std::atan2(dc12[1], dc12[0]));
      beta4 = clampRadian(beta3 + M_PI);
      beta5 = alpha1;
      arc0 = 0;
      arc1 = ccwArc(beta2, beta3);
      arc2 = cwArc(beta4, beta5);
      dist_i = r * (arc0 + arc1 + arc2);

      out[2] = Vector3d_t(arc0, arc1, arc2);
      if (dist_i < dist) {
        iBest = 2;
        dist = dist_i;
      }
    }

    // for the solution that minimizes L1, this requires no arc2
    c1 = rightCenter(q1, r);
    c2 = leftCenter(q1, r);
    dx = c1[0] - c0[0];
    if (std::abs(dx) + eps < 2 * r) {
      // there are two solutions +dx and -dx
      dy = std::sqrt(4 * r * r - dx * dx);

      // for the +dx solution, first we compute how much c0[0] + dx would
      // change c1[0], so we can apply the same translation to
      // c2
      Format_t dyy = (c0[1] + dy) - c1[1];
      c1[1] += dyy;
      c2[1] += dyy;

      dc01 = c1 - c0;
      dc12 = c2 - c1;
      beta0 = alpha0;
      beta1 = clampRadian(std::atan2(dc01[1], dc01[0]));
      beta2 = clampRadian(beta1 + M_PI);
      beta3 = clampRadian(std::atan2(dc12[1], dc12[0]));
      beta4 = clampRadian(beta3 + M_PI);
      arc0 = cwArc(beta0, beta1);
      arc1 = ccwArc(beta2, beta3);
      arc2 = 0;
      dist_i = r * (arc0 + arc1 + arc2);

      out[3] = Vector3d_t(arc0, arc1, arc2);
      if (dist_i < dist) {
        iBest = 3;
        dist = dist_i;
      }

      dyy = -2 * dy;
      c1[1] += dyy;
      c2[1] += dyy;

      dc01 = c1 - c0;
      dc12 = c2 - c1;
      beta0 = alpha0;
      beta1 = clampRadian(std::atan2(dc01[1], dc01[0]));
      beta2 = clampRadian(beta1 + M_PI);
      beta3 = clampRadian(std::atan2(dc12[1], dc12[0]));
      beta4 = clampRadian(beta3 + M_PI);
      arc0 = cwArc(beta0, beta1);
      arc1 = ccwArc(beta2, beta3);
      arc2 = 0;
      dist_i = r * (arc0 + arc1 + arc2);

      out[4] = Vector3d_t(arc0, arc1, arc2);
      if (dist_i < dist) {
        iBest = 4;
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

#endif  // MPBLOCKS_DUBINS_CURVES_EIGEN_HYPER_TWO_THETA_X_RLR_H_
