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
 *  \file   include/mpblocks/dubins/curves/Solution.h
 *
 *  \date   Nov 2, 2012
 *  \author Josh Bialkowski (jbialk@mit.edu)
 *  \brief  
 */

#ifndef MPBLOCKS_DUBINS_CURVES_CUDA2_SOLUTION_H_
#define MPBLOCKS_DUBINS_CURVES_CUDA2_SOLUTION_H_

#include <mpblocks/cuda/linalg2.h>

namespace     mpblocks {
namespace       dubins {
namespace  curves_cuda {


template <typename Format_t>
struct DebugStraight
{
    typedef linalg::Matrix<Format_t,2,1> Vector2d_t;

    Vector2d_t c[2];
    Vector2d_t t[2];
    Format_t   l[3];
};

template <typename Format_t>
struct DebugCurved
{
    typedef linalg::Matrix<Format_t,2,1> Vector2d_t;

    Vector2d_t c[3];
    Format_t   l[3];
};

/// interface for different solutions
template <SolutionId Id, typename Format_t>
class Solver
{
    typedef linalg::Matrix<Format_t,3,1> Vector3d_t;

    struct DebugResult{};

    /// basic interface returns only the total distance
    static Result<Format_t> solve( const Vector3d_t& q0,
                                   const Vector3d_t& q1,
                                   const Format_t r );

    /// Extended interface will return intermediate results
    static Result<Format_t> solve( const Vector3d_t& q0,
                                   const Vector3d_t& q1,
                                   const Format_t r,
                                   DebugResult& out);
};



} // curves
} // dubins
} // mpblocks










#endif // SOLUTION_H_
