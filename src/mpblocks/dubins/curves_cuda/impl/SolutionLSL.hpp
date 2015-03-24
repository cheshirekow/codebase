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
 *  \file   mpblocks/dubins/curves/SolutionLSL.hpp
 *
 *  \date   Oct 30, 2012
 *  \author Josh Bialkowski (jbialk@mit.edu)
 *  \brief  
 */

#ifndef MPBLOCKS_DUBINS_CURVES_CUDA2_SOLUTION_LSL_HPP_
#define MPBLOCKS_DUBINS_CURVES_CUDA2_SOLUTION_LSL_HPP_

#include <limits>


namespace    mpblocks {
namespace      dubins {
namespace curves_cuda {



/// interface for different solutions
template <typename Format_t>
struct Solver<LSL,Format_t>
{
    typedef linalg::Matrix<Format_t,3,1>   Vector3d_t;
    typedef linalg::Matrix<Format_t,2,1>   Vector2d_t;
    typedef linalg::Rotation2d<Format_t>   Rotation2d_t;
    typedef intr::Dispatch<WHICH_CUDA>  Dispatch;
    typedef Result<Format_t> Result_t;
    typedef DebugStraight<Format_t> DebugResult;

    /// static method implementing the solution proces
    __host__ __device__ static Result<Format_t> solve(
            const Vector3d_t& q0,
            const Vector3d_t& q1,
            const Format_t r );

    __host__ __device__ static Result<Format_t> solve_debug(
            const Vector3d_t& q0,
            const Vector3d_t& q1,
            const Format_t r,
            DebugResult& out );
};


template <typename Format_t>
__host__ __device__ Result<Format_t> Solver<LSL,Format_t>::solve(
        const Vector3d_t& q0,
        const Vector3d_t& q1,
        const Format_t r )
{
    using namespace std;
    using namespace cuda::linalg2;

    const Format_t _PI = static_cast<Format_t>(M_PI);
    const Format_t _2  = static_cast<Format_t>(2);

    Vector2d_t x,v,c[3];

    // calculate the center of the circle to which q1 is tangent
    x = view<0,2>( q0 );
    v << -Dispatch::sin( get<2>(q0) ),
          Dispatch::cos( get<2>(q0) );
    c[0] = x + mktmp(r*v);

    // calculate the center of the circle to which q2 is tangent
    x = view<0,2>( q1 );
    v <<  -Dispatch::sin( get<2>(q1) ),
           Dispatch::cos( get<2>(q1) );
    c[1] = x + mktmp(r*v);

    // find the vector between the two
    v = c[1] - c[0];
    c[2] = v;

    // the length of the straight segment is the length of this vector
    Format_t l[3];
    l[2] = norm(v);

    // calculate the angle this vector makes with the circle
    Format_t a = Dispatch::atan2( get<1>(v), get<0>(v) );
    a = clampRadian(a);

    // calculate tangents
    v << Dispatch::cos( a - (_PI/_2) ),
         Dispatch::sin( a - (_PI/_2) );

//    #pragma unroll
//    for(int i=0; i < 2; i++)
//        t[i] = c[i] + mktmp(r*v);

    // now find the arc length for the two circles until it reaches this
    // point
    l[0] = ccwArc( get<2>(q0), a );
    l[1] = ccwArc( a, get<2>(q1) );

    return Result_t( r*(l[0] + l[1]) + l[2] );
}















template <typename Format_t>
__host__ __device__ Result<Format_t> Solver<LSL,Format_t>::solve_debug(
        const Vector3d_t& q0,
        const Vector3d_t& q1,
        const Format_t r,
            DebugResult& out )
{
    using namespace std;
    using namespace cuda::linalg2;

    const Format_t _PI = static_cast<Format_t>(M_PI);
    const Format_t _2  = static_cast<Format_t>(2);

    Vector2d_t x,v,c[3];

    // calculate the center of the circle to which q1 is tangent
    x = view<0,2>( q0 );
    v << -Dispatch::sin( get<2>(q0) ),
          Dispatch::cos( get<2>(q0) );
    c[0] = x + mktmp(r*v);

    // calculate the center of the circle to which q2 is tangent
    x = view<0,2>( q1 );
    v <<  -Dispatch::sin( get<2>(q1) ),
           Dispatch::cos( get<2>(q1) );
    c[1] = x + mktmp(r*v);

    // find the vector between the two
    v = c[1] - c[0];
    c[2] = v;

    // the length of the straight segment is the length of this vector
    Format_t l[3];
    l[2] = norm(v);

    // calculate the angle this vector makes with the circle
    Format_t a = Dispatch::atan2( get<1>(v), get<0>(v) );
    a = clampRadian(a);

    // calculate tangents
    v << Dispatch::cos( a - (_PI/_2) ),
         Dispatch::sin( a - (_PI/_2) );

#ifdef __CUDACC__
    #pragma unroll
#endif
    for(int i=0; i < 2; i++)
    {
        out.t[i] = c[i] + mktmp(r*v);
        out.c[i] = c[i];
    }

    // now find the arc length for the two circles until it reaches this
    // point
    l[0] = ccwArc( get<2>(q0), a );
    l[1] = ccwArc( a, get<2>(q1) );

#ifdef __CUDACC__
    #pragma unroll
#endif
    for(int i=0; i < 3; i++)
        out.l[i] = l[i];

    return Result_t( r*(l[0] + l[1]) + l[2] );
}















} // curves
} // dubins
} // mpblocks


#endif // SOLUTION_LSL_H_
