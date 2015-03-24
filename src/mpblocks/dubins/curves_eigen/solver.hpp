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

#ifndef MPBLOCKS_DUBINS_CURVES_EIGEN_SOLVER_HPP_
#define MPBLOCKS_DUBINS_CURVES_EIGEN_SOLVER_HPP_

namespace     mpblocks {
namespace       dubins {
namespace curves_eigen {

template <SolutionId idx, typename Format_t>
struct SolverIterator {
  typedef Path<Format_t> Result_t;
  typedef Solver<idx, Format_t> Solver_t;
  typedef SolverIterator<(SolutionId)(idx + 1), Format_t> Next_t;

  static Result_t solve(const Eigen::Matrix<Format_t, 3, 1>& q0,
                        const Eigen::Matrix<Format_t, 3, 1>& q1,
                        const Format_t r) {
    Result_t thisResult = Solver_t::solve(q0, q1, r);
    return bestOf(thisResult, Next_t::solve(q0, q1, r), r);
  }

  static Result_t solve_specific(int solnId,
                                 const Eigen::Matrix<Format_t, 3, 1>& q0,
                                 const Eigen::Matrix<Format_t, 3, 1>& q1,
                                 const Format_t r) {
    if (solnId == idx)
      return Solver_t::solve(q0, q1, r);
    else
      return Next_t::solve_specific(solnId, q0, q1, r);
  }
};

template <typename Format_t>
struct SolverIterator<INVALID, Format_t> {
  typedef Path<Format_t> Result_t;

  static Result_t solve(const Eigen::Matrix<Format_t, 3, 1>& q0,
                        const Eigen::Matrix<Format_t, 3, 1>& q1,
                        const Format_t r) {
    return Result_t();
  }

  static Result_t solve_specific(int solnId,
                                 const Eigen::Matrix<Format_t, 3, 1>& q0,
                                 const Eigen::Matrix<Format_t, 3, 1>& q1,
                                 const Format_t r) {
    return Result_t();
  }
};

template <typename Format_t>
Path<Format_t> solve(const Eigen::Matrix<Format_t, 3, 1>& q0,
                     const Eigen::Matrix<Format_t, 3, 1>& q1,
                     const Format_t r) {
  return SolverIterator<(SolutionId)0, Format_t>::solve(q0, q1, r);
}

template <typename Format_t>
Path<Format_t> solve_specific(int solnId,
                              const Eigen::Matrix<Format_t, 3, 1>& q0,
                              const Eigen::Matrix<Format_t, 3, 1>& q1,
                              const Format_t r) {
  return SolverIterator<(SolutionId)0, Format_t>::solve_specific(
      solnId, q0, q1, r);
}

}  // curves_eigen
}  // dubins
}  // mpblocks

#endif  // MPBLOCKS_DUBINS_CURVES_EIGEN_SOLVER_HPP_
