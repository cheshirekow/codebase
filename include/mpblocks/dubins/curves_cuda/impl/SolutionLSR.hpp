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
 *  \file   mpblocks/dubins/curves/SolutionLSR.hpp
 *
 *  \date   Oct 30, 2012
 *  \author Josh Bialkowski (jbialk@mit.edu)
 *  \brief  
 */

#ifndef MPBLOCKS_DUBINS_CURVES_CUDA2_SOLUTION_LSR_HPP_
#define MPBLOCKS_DUBINS_CURVES_CUDA2_SOLUTION_LSR_HPP_

#include <limits>

namespace mpblocks
{
namespace dubins
{
namespace curves_cuda
{

/// interface for different solutions
template<typename Format_t>
struct Solver<LSR, Format_t>
{
    typedef linalg::Matrix<Format_t, 3, 1> Vector3d_t;
    typedef linalg::Matrix<Format_t, 2, 1> Vector2d_t;
    typedef linalg::Rotation2d<Format_t> Rotation2d_t;
    typedef intr::Dispatch<WHICH_CUDA> Dispatch;
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
__host__ __device__ Result<Format_t> Solver<LSR,Format_t>::solve(
        const Vector3d_t& q0,
        const Vector3d_t& q1,
        const Format_t r )
{
    using namespace std;
    using namespace cuda::linalg2;

    const Format_t _PI = static_cast<Format_t>(M_PI);
    const Format_t _2 = static_cast<Format_t>(2);

    Vector2d_t x,v,c[3];

    // calculate the center of the circle to which q1 is tangent
    x = view<0,2>( q0 );
    v << -Dispatch::sin( get<2>(q0) ),
    Dispatch::cos( get<2>(q0) );
    c[0] = x + mktmp(r*v);

    // calculate the center of the circle to which q2 is tangent
    x = view<0,2>(q1);
    v << Dispatch::sin( get<2>(q1) ),
    -Dispatch::cos( get<2>(q1) );
    c[1] = x + mktmp(r*v);

    // find the vector between the two
    v = c[1] - c[0];

    // calculate the distance between the two
    Format_t d = norm(v);

    // if they overlap then this primitive has no solution and is not the
    // optimal primitive
    if( d < _2*r )\
        return Result_t();

    // find the angle between the line through centers and the radius that
    // points to the tangent on the circle

    Format_t a = -Dispatch::acos( _2*r / d );

    // get a normalized vector in this direction
    Rotation2d_t R( Dispatch::sin(a), Dispatch::cos(a) );
    Vector2d_t n = R * mktmp(normalize( v ));

    // get the tangent points
    Vector2d_t t[2];
    t[0] = c[0] + mktmp(n*r);
    t[1] = c[1] - mktmp(n*r);

    // store the vector in c[2]
    c[2] = t[1] - t[0];

    // get the angle to the tangent points and the arclenghts on the two
    // circles
    Format_t b,l[3];
    a = Dispatch::atan2( get<1>(n) , get<0>(n) );
    b = clampRadian( get<2>(q0) - (_PI/_2) );
    l[0] = ccwArc(b,a);

    a = clampRadian( a + _PI );
    b = clampRadian( get<2>(q1) + (_PI/_2) );
    l[1] = cwArc(a,b);

    // get the length of the segment
    l[2] = norm(t[0]-t[1]);

    return Result_t( r*(l[0] + l[1]) + l[2] );
}














template <typename Format_t>
__host__ __device__ Result<Format_t> Solver<LSR,Format_t>::solve_debug(
        const Vector3d_t& q0,
        const Vector3d_t& q1,
        const Format_t r,
            DebugResult& out )
{
    using namespace std;
    using namespace cuda::linalg2;

    const Format_t _PI = static_cast<Format_t>(M_PI);
    const Format_t _2 = static_cast<Format_t>(2);

    Vector2d_t x,v,c[3];

    // calculate the center of the circle to which q1 is tangent
    x = view<0,2>( q0 );
    v << -Dispatch::sin( get<2>(q0) ),
    Dispatch::cos( get<2>(q0) );
    c[0] = x + mktmp(r*v);

    // calculate the center of the circle to which q2 is tangent
    x = view<0,2>(q1);
    v << Dispatch::sin( get<2>(q1) ),
    -Dispatch::cos( get<2>(q1) );
    c[1] = x + mktmp(r*v);

    // find the vector between the two
    v = c[1] - c[0];

    // calculate the distance between the two
    Format_t d = norm(v);

    // if they overlap then this primitive has no solution and is not the
    // optimal primitive
    if( d < _2*r )\
        return Result_t();

    // find the angle between the line through centers and the radius that
    // points to the tangent on the circle

    Format_t a = -Dispatch::acos( _2*r / d );

    // get a normalized vector in this direction
    Rotation2d_t R( Dispatch::sin(a), Dispatch::cos(a) );
    Vector2d_t n = R * mktmp(normalize( v ));

    // get the tangent points
    Vector2d_t t[2];
    t[0] = c[0] + mktmp(n*r);
    t[1] = c[1] - mktmp(n*r);

    // store the vector in c[2]
    c[2] = t[1] - t[0];

    // get the angle to the tangent points and the arclenghts on the two
    // circles
    Format_t b,l[3];
    a = Dispatch::atan2( get<1>(n) , get<0>(n) );
    b = clampRadian( get<2>(q0) - (_PI/_2) );
    l[0] = ccwArc(b,a);

    a = clampRadian( a + _PI );
    b = clampRadian( get<2>(q1) + (_PI/_2) );
    l[1] = cwArc(a,b);

    // get the length of the segment
    l[2] = norm(t[0]-t[1]);

#ifdef __CUDACC__
    #pragma unroll
#endif
    for(int i=0; i < 2; i++)
    {
        out.t[i] = t[i];
        out.c[i] = c[i];
    }

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


#endif // SOLUTIONLRLA_H_
