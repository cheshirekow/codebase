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

#ifndef MPBLOCKS_DUBINS_CURVES_EIGEN_SOLVER_H_
#define MPBLOCKS_DUBINS_CURVES_EIGEN_SOLVER_H_

#include <mpblocks/dubins/curves/path.h>
#include <mpblocks/dubins/curves/types.h>

namespace     mpblocks {
namespace       dubins {
namespace curves_eigen {

/// interface for different solutions, this is specialized for each Id in
/// the SolutionId enum
template <SolutionId Id, typename Format_t>
struct Solver
{
    typedef Eigen::Matrix<Format_t,3,1>   Vector3d_t;
    typedef Eigen::Matrix<Format_t,2,1>   Vector2d_t;
    typedef Path<Format_t>                Result_t;

    static Result_t solve(
               const Vector3d_t& q0,
               const Vector3d_t& q1,
               const Format_t r );
};

template <typename Format_t>
Path<Format_t> solve(
        const Eigen::Matrix<Format_t,3,1>& q0,
        const Eigen::Matrix<Format_t,3,1>& q1,
        const Format_t r );


template <typename Format_t >
Path<Format_t> solve_specific(
        int solver,
        const Eigen::Matrix<Format_t,3,1>& q0,
        const Eigen::Matrix<Format_t,3,1>& q1,
        const Format_t r );

} // curves_eigen
} // dubins
} // mpblocks

#endif // MPBLOCKS_DUBINS_CURVES_EIGEN_SOLVER_H_
