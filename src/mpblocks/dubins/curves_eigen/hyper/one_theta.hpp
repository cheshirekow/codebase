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

#ifndef MPBLOCKS_DUBINS_CURVES_EIGEN_HYPER_ONE_THETA_H_
#define MPBLOCKS_DUBINS_CURVES_EIGEN_HYPER_ONE_THETA_H_

namespace     mpblocks {
namespace       dubins {
namespace curves_eigen {
namespace        hyper {

/// specialization for a single theta constraint (min or max)
template <int tSpec, typename Format_t>
struct Solver<OFF, OFF, tSpec, Format_t> {
  typedef Eigen::Matrix<Format_t, 3, 1> Vector3d_t;
  typedef Eigen::Matrix<Format_t, 2, 1> Vector2d_t;
  typedef Path<Format_t> Result_t;
  typedef HyperRect<Format_t> Hyper_t;

  static const unsigned int SPEC = SpecPack<OFF, OFF, tSpec>::Result;

  /// get the distance for a left turn
  static Result_t solveL(const Vector3d_t& q0, const Hyper_t& h,
                         const Format_t r) {
    Format_t arc0 =
        ccwArc(clampRadian(q0[2]), clampRadian(get_constraint<tSpec, 2>(h)));
    Result_t out(LRLa);
    out = Vector3d_t(arc0, 0, 0);
    return out;
  };

  static Result_t solveR(const Vector3d_t& q0, const Hyper_t& h,
                         const Format_t r) {
    Format_t arc0 =
        cwArc(clampRadian(q0[2]), clampRadian(get_constraint<tSpec, 2>(h)));
    Result_t out(RLRa);
    out = Vector3d_t(arc0, 0, 0);

    return out;
  };

  static Result_t solve(const Vector3d_t& q0, const Hyper_t& h,
                        const Format_t r) {
    return bestOf(solveL(q0, h, r), solveR(q0, h, r), r);
  };
};

}  // namespace hyper
}  // namespace curves_eigen
}  // namespace dubins
}  // namespace mpblocks

#endif  // MPBLOCKS_DUBINS_CURVES_EIGEN_HYPER_ONE_THETA_H_
