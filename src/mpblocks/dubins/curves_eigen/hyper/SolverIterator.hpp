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
 *  @file   /home/josh/Codes/cpp/mpblocks2/dubins/include/mpblocks/dubins/curves_eigen/hyper/Solver.h
 *
 *  @date   Jun 27, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_DUBINS_CURVES_EIGEN_HYPER_SOLVER_ITERATOR_H_
#define MPBLOCKS_DUBINS_CURVES_EIGEN_HYPER_SOLVER_ITERATOR_H_

#include <boost/format.hpp>

namespace     mpblocks {
namespace       dubins {
namespace curves_eigen {
namespace        hyper {

// return true if the result satisfies inactive constraints as well
template <typename Format_t, int spec, int idx>
bool validate_off(const Eigen::Matrix<Format_t, 3, 1>& q,
                  const HyperRect<Format_t>& h) {
  if (spec == OFF) {
    if (q[idx] < h.minExt[idx] || q[idx] > h.maxExt[idx]) return false;
  }
  return true;
}

// return true if the result satisfies all in active constraints
template <typename Format_t, int xSpec, int ySpec, int tSpec>
bool validate(const Eigen::Matrix<Format_t, 3, 1>& q,
              const HyperRect<Format_t>& h) {
  return validate_off<Format_t, xSpec, 0>(q, h) &&
         validate_off<Format_t, ySpec, 1>(q, h) &&
         validate_off<Format_t, tSpec, 2>(q, h);
}

/// iterates over solvers and accumulates the solution
template <int xSpec, int ySpec, int tSpec, typename Format_t>
struct SolverIterator {
  typedef Eigen::Matrix<Format_t, 3, 1> Vector3d_t;
  typedef HyperRect<Format_t> Hyper_t;
  typedef Path<Format_t> Result_t;
  typedef Integrate<Format_t> Integrate_t;

  typedef SolverIterator<xSpec, ySpec, tSpec + 1, Format_t> Next_t;
  typedef Solver<xSpec, ySpec, tSpec, Format_t> Solver_t;
  typedef SpecPack<xSpec, ySpec, tSpec> Spec;

  static Result_t solve(const Vector3d_t& q0, const Hyper_t& h,
                        const Format_t r) {
    Result_t thisSoln = Solver_t::solve(q0, h, r);

    // if the solution is nominally feasible, then check to see if it
    // satisfies all non-explicit constraints
    if (thisSoln.f) {
      Vector3d_t q1 = Integrate_t::solve(q0, thisSoln, r);
      thisSoln.f = validate<Format_t, xSpec, ySpec, tSpec>(q1, h);
    }

    Result_t thisAccum = bestOf(thisSoln, Next_t::solve(q0, h, r), r);

    //        std::cout << "solver iterator, spec " << Spec::Result << ", best:
    //        "
    //                  << thisSoln.dist(r) << " " << (thisSoln.f ? "T":"F")
    //                  << "\n";

    return thisAccum;
  };
};

/// specialization to end template recursino
template <int xSpec, int ySpec, typename Format_t>
struct SolverIterator<xSpec, ySpec, 3, Format_t> {
  typedef Eigen::Matrix<Format_t, 3, 1> Vector3d_t;
  typedef DistanceAndId<Format_t> Result2_t;
  typedef HyperRect<Format_t> Hyper_t;
  typedef Path<Format_t> Result_t;

  typedef SolverIterator<xSpec, ySpec + 1, 0, Format_t> Next_t;

  static Result_t solve(const Vector3d_t& q0, const Hyper_t& h,
                        const Format_t r) {
    return Next_t::solve(q0, h, r);
  };
};

/// specialization to end template recursino
template <int xSpec, int tSpec, typename Format_t>
struct SolverIterator<xSpec, 3, tSpec, Format_t> {
  typedef Eigen::Matrix<Format_t, 3, 1> Vector3d_t;
  typedef HyperRect<Format_t> Hyper_t;
  typedef Path<Format_t> Result_t;

  typedef SolverIterator<xSpec + 1, 0, 0, Format_t> Next_t;

  static Result_t solve(const Vector3d_t& q0, const Hyper_t& h,
                        const Format_t r) {
    return Next_t::solve(q0, h, r);
  };
};

/// specialization to end template recursino
template <int ySpec, int tSpec, typename Format_t>
struct SolverIterator<3, ySpec, tSpec, Format_t> {
  typedef Eigen::Matrix<Format_t, 3, 1> Vector3d_t;
  typedef HyperRect<Format_t> Hyper_t;
  typedef Path<Format_t> Result_t;

  static Result_t solve(const Vector3d_t& q0, const Hyper_t& h,
                        const Format_t r) {
    return Result_t();
  };
};

template <typename Format_t>
Path<Format_t> solve(const Eigen::Matrix<Format_t, 3, 1>& q0,
                     const HyperRect<Format_t>& h, const Format_t r) {
  return SolverIterator<0, 0, 0, Format_t>::solve(q0, h, r);
}

}  // namespace hyper
}  // namespace curves_eigen
}  // namespace dubins
}  // namespace mpblocks

#endif  // MPBLOCKS_DUBINS_CURVES_EIGEN_HYPER_SOLVER_ITERATOR_H_
