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

#ifndef MPBLOCKS_DUBINS_CURVES_EIGEN_HYPER_SOLVER_H_
#define MPBLOCKS_DUBINS_CURVES_EIGEN_HYPER_SOLVER_H_

namespace     mpblocks {
namespace       dubins {
namespace curves_eigen {
namespace        hyper {

/// returns either min or max element depending on spec
template <int spec, int idx, typename Format_t>
Format_t get_constraint(const HyperRect<Format_t>& in) {
  if (spec == OFF) assert(false);
  if (spec == MIN) return in.minExt[idx];
  if (spec == MAX) return in.maxExt[idx];
}

/// returns either min or max element depending on spec
template <int xSpec, int ySpec, int tSpec, typename Format_t>
void build_target(const HyperRect<Format_t>& in,
                  Eigen::Matrix<Format_t, 3, 1>& out) {
  if (xSpec == MIN)
    out[0] = in.minExt[0];
  else
    out[0] = in.maxExt[0];

  if (ySpec == MIN)
    out[1] = in.minExt[1];
  else
    out[1] = in.maxExt[1];

  if (tSpec == MIN)
    out[2] = in.minExt[2];
  else
    out[2] = in.maxExt[2];
}

template <int xSpec, int ySpec, int tSpec>
struct ThreeConstraintHelper {
  static const unsigned int SPEC = SpecPack<xSpec, ySpec, tSpec>::Result;

  template <int idx, typename Format_t>
  struct Iterator {
    typedef Path<Format_t> Result_t;
    typedef curves_eigen::Solver<(SolutionId)idx, Format_t> Solver_t;
    typedef Iterator<idx + 1, Format_t> Next_t;

    static Result_t solve(const Eigen::Matrix<Format_t, 3, 1>& q0,
                          const Eigen::Matrix<Format_t, 3, 1>& q1,
                          const Format_t r) {
      Result_t thisResult = Solver_t::solve(q0, q1, r);
      return bestOf(thisResult, Next_t::solve(q0, q1, r), r);
    }
  };

  template <typename Format_t>
  struct Iterator<INVALID, Format_t> {
    typedef Path<Format_t> Result_t;

    static Result_t solve(const Eigen::Matrix<Format_t, 3, 1>& q0,
                          const Eigen::Matrix<Format_t, 3, 1>& q1,
                          const Format_t r) {
      return Result_t();
    }
  };
};

/// the default solver is instantated when not all three constraints are active
/// and it simply dispatches the point-to-point solver
template <int xSpec, int ySpec, int tSpec, typename Format_t>
struct Solver {
  typedef Eigen::Matrix<Format_t, 3, 1> Vector3d_t;
  typedef Eigen::Matrix<Format_t, 2, 1> Vector2d_t;
  typedef Path<Format_t> Result_t;
  typedef HyperRect<Format_t> Hyper_t;
  typedef ThreeConstraintHelper<xSpec, ySpec, tSpec> Helper_t;
  typedef typename Helper_t::template Iterator<0, Format_t> Iterator_t;

  static Result_t solve(const Vector3d_t& q0, const Hyper_t& h,
                        const Format_t r) {
    Vector3d_t q1;
    build_target<xSpec, ySpec, tSpec, Format_t>(h, q1);
    return Iterator_t::solve(q0, q1, r);
  };
};

}  // namespace hyper
}  // namespace curves_eigen
}  // namespace dubins
}  // namespace mpblocks

#endif  // MPBLOCKS_DUBINS_CURVES_EIGEN_HYPER_SOLVER_H_
