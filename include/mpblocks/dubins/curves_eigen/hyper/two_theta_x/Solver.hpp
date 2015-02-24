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

#ifndef MPBLOCKS_DUBINS_CURVES_EIGEN_HYPER_TWO_THETA_X_SOLVER_H_
#define MPBLOCKS_DUBINS_CURVES_EIGEN_HYPER_TWO_THETA_X_SOLVER_H_

namespace     mpblocks {
namespace       dubins {
namespace curves_eigen {
namespace        hyper {
namespace  two_theta_x {

/// interface for variants of solvers, default template is never instantiated
template <int xSpec, int tSpec, int Variant, typename Format_t>
struct Solver {
  typedef Eigen::Matrix<Format_t, 3, 1> Vector3d_t;
  typedef Eigen::Matrix<Format_t, 2, 1> Vector2d_t;
  typedef Path<Format_t> Result_t;
  typedef HyperRect<Format_t> Hyper_t;

  static Result_t solve(const Vector3d_t& q0, const Hyper_t& h,
                        const Format_t r);

  static Result_t solve_debug(const Vector3d_t& q0, const Hyper_t& h,
                              const Format_t r);
};

}  // namespace two_theta_x
}  // namespace hyper
}  // namespace curves_eigen
}  // namespace dubins
}  // namespace mpblocks

#endif  // MPBLOCKS_DUBINS_CURVES_EIGEN_HYPER_TWO_THETA_X_SOLVER_H_
