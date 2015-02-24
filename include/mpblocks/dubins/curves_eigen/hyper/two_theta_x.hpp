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
 *  @file   dubins/curves_eigen/hyper/two_x_y.h
 *
 *  @date   Jun 27, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_DUBINS_CURVES_EIGEN_HYPER_TWO_THETA_X_H_
#define MPBLOCKS_DUBINS_CURVES_EIGEN_HYPER_TWO_THETA_X_H_

#include <mpblocks/dubins/curves_eigen/hyper/two_theta_x/Solver.hpp>
#include <mpblocks/dubins/curves_eigen/hyper/two_theta_x/LSL.hpp>
#include <mpblocks/dubins/curves_eigen/hyper/two_theta_x/LSR.hpp>
#include <mpblocks/dubins/curves_eigen/hyper/two_theta_x/RSR.hpp>
#include <mpblocks/dubins/curves_eigen/hyper/two_theta_x/RSL.hpp>
#include <mpblocks/dubins/curves_eigen/hyper/two_theta_x/LRL.hpp>
#include <mpblocks/dubins/curves_eigen/hyper/two_theta_x/RLR.hpp>

namespace     mpblocks {
namespace       dubins {
namespace curves_eigen {
namespace        hyper {
namespace  two_theta_x {

/// wraps structures and accessors for the specific xSpec and tSpec
template <int xSpec, int tSpec, typename Format_t>
struct HelperWrap {
  /// provides iteration over solvers
  template <int... List>
  struct SolverIterator {};

  template <int First, int... List>
  struct SolverIterator<First, List...> {
    typedef Eigen::Matrix<Format_t, 3, 1> Vector3d_t;
    typedef Eigen::Matrix<Format_t, 2, 1> Vector2d_t;
    typedef Path<Format_t> Result_t;
    typedef HyperRect<Format_t> Hyper_t;

    typedef Solver<xSpec, tSpec, First, Format_t> Solver_t;

    static Result_t solve(const Vector3d_t& q0, const Hyper_t& h,
                          const Format_t r) {
      return bestOf(Solver_t::solve(q0, h, r),
                    SolverIterator<List...>::solve(q0, h, r), r);
    };
  };

  template <int Last>
  struct SolverIterator<Last> {
    typedef Eigen::Matrix<Format_t, 3, 1> Vector3d_t;
    typedef Eigen::Matrix<Format_t, 2, 1> Vector2d_t;
    typedef Path<Format_t> Result_t;
    typedef HyperRect<Format_t> Hyper_t;

    typedef Solver<xSpec, tSpec, Last, Format_t> Solver_t;

    static Result_t solve(const Vector3d_t& q0, const Hyper_t& h,
                          const Format_t r) {
      return Solver_t::solve(q0, h, r);
    };
  };

  typedef SolverIterator<LSL, LSR, RSL, RSR, LRLa, RLRa> SolverIterator_t;
};

}  // namespace two_theta_x

/// specialization for a single theta constraint (min or max)
template <int xSpec, int tSpec, typename Format_t>
struct Solver<xSpec, OFF, tSpec, Format_t> {
  typedef Eigen::Matrix<Format_t, 3, 1> Vector3d_t;
  typedef Eigen::Matrix<Format_t, 2, 1> Vector2d_t;
  typedef Path<Format_t> Result_t;
  typedef HyperRect<Format_t> Hyper_t;

  typedef two_theta_x::HelperWrap<xSpec, tSpec, Format_t> Helper;
  typedef typename Helper::SolverIterator_t Iterator;

  static Result_t solve(const Vector3d_t& q0, const Hyper_t& h,
                        const Format_t r) {
    return Iterator::solve(q0, h, r);
  };
};

}  // namespace hyper
}  // namespace curves_eigen
}  // namespace dubins
}  // namespace mpblocks

#endif  // MPBLOCKS_DUBINS_CURVES_EIGEN_HYPER_TWO_THETA_X_H_
