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
 *  \file   mpblocks/dubins/curves/SolutionLRLa.hpp
 *
 *  \date   Oct 30, 2012
 *  \author Josh Bialkowski (jbialk@mit.edu)
 *  \brief  
 */

#ifndef MPBLOCKS_DUBINS_CURVES_CUDA2_SOLUTION_LRL_A_HPP_
#define MPBLOCKS_DUBINS_CURVES_CUDA2_SOLUTION_LRL_A_HPP_

#include <limits>

namespace    mpblocks {
namespace      dubins {
namespace curves_cuda {

/// interface for different solutions
template <typename Format_t>
struct Solver<LRLa,Format_t>
{
    typedef linalg::Matrix<Format_t,3,1>   Vector3d_t;
    typedef linalg::Matrix<Format_t,2,1>   Vector2d_t;
    typedef linalg::Rotation2d<Format_t>   Rotation2d_t;
    typedef intr::Dispatch<WHICH_CUDA>  Dispatch;
    typedef Result<Format_t>      Result_t;
    typedef DebugCurved<Format_t> DebugResult;


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
__host__ __device__ Result<Format_t> Solver<LRLa,Format_t>::solve(
        const Vector3d_t& q0,
        const Vector3d_t& q1,
        const Format_t r )
{
    using namespace std;
    using namespace cuda::linalg2;

    const Format_t _PI = static_cast<Format_t>(M_PI);
    const Format_t _2  = static_cast<Format_t>(2);
    const Format_t _4  = static_cast<Format_t>(4);

    Vector2d_t x,v,c[3];

    // calculate the center of the circle to which q1 is tangent
    x = view<0,2>( q0 );
    v << -Dispatch::sin( get<2>(q0) ),
          Dispatch::cos( get<2>(q0) );
    c[0] = x + mktmp(r*v);

    // calculate the center of the circle to which q2 is tangent
    x = view<0,2>(q1);
    v << -Dispatch::sin( get<2>(q1) ),
          Dispatch::cos( get<2>(q1) );
    c[1] = x + mktmp(r*v);

    // the distance between the centers of these two circles
    Format_t d = norm(c[0]-c[1]);

    // if the distance is too large, then this primitive is not the solution,
    // and we can bail here
    if( d > _4*r )
        return Result_t();

    // the base angle of the isosceles triangle whose vertices are the centers
    // of the the three circles, note acos returns [0,pi]
    Format_t a = -Dispatch::acos( d / (_4*r) );

    // create a clockwise rotation of magnitude alpha
    Rotation2d_t R( Dispatch::sin(a), Dispatch::cos(a) );

    // we find the third vertex of this triangle by taking the vector between
    // the two circle centers, normalizing it, and rotating it by alpha, and
    // scaling it to magnitude 2r, then it points from the center of
    // one the circle tangent to q1 to the third vertex
    Vector2d_t A = normalize( mktmp(c[1]-c[0]) );
    Vector2d_t B = mktmp(R*A) * (_2*r);
    c[2] = c[0] + B;

    // calculate the arc distance we travel on the first circle
    Vector2d_t dc;
    Format_t b,l[3];

    dc = c[2] - c[0];         //< vector between centers of circles
    b  = Dispatch::atan2( get<1>(dc), get<0>(dc) );
                                        //< angle of that vector
    d  = clampRadian( get<2>(q0) - (_PI/_2) );
                                        //< angle of vector from center to q1
    l[0] = ccwArc(d,b);            //< ccwise distance


    // calculate the arc distance we travel on the second circle
    dc = c[2] - c[1];         //< vector between centers of circles
    b  = Dispatch::atan2( get<1>(dc), get<0>(dc) );
                                        //< angle of that vector
    d  = clampRadian( get<2>(q1) - (_PI/_2) );
                                        //< angle of vector from center to q1
    l[1] = ccwArc(b,d);            //< ccwise distance

    // calculate the arc distance we travel on the third circle
    l[2] = _PI + _2*a;

    // sum the resulting segments
    Format_t dist = 0;

#ifdef __CUDACC__
    #pragma unroll
#endif
    for(int i=0; i < 3; i++)
        dist += l[i];

    return Result_t(r*dist);
}









template <typename Format_t>
__host__ __device__ Result<Format_t> Solver<LRLa,Format_t>::solve_debug(
        const Vector3d_t& q0,
        const Vector3d_t& q1,
        const Format_t r,
            DebugResult& out )
{
    using namespace std;
    using namespace cuda::linalg2;

    const Format_t _PI = static_cast<Format_t>(M_PI);
    const Format_t _2  = static_cast<Format_t>(2);
    const Format_t _4  = static_cast<Format_t>(4);

    Vector2d_t x,v,c[3];

    // calculate the center of the circle to which q1 is tangent
    x = view<0,2>( q0 );
    v << -Dispatch::sin( get<2>(q0) ),
          Dispatch::cos( get<2>(q0) );
    c[0] = x + mktmp(r*v);

    // calculate the center of the circle to which q2 is tangent
    x = view<0,2>(q1);
    v << -Dispatch::sin( get<2>(q1) ),
          Dispatch::cos( get<2>(q1) );
    c[1] = x + mktmp(r*v);

    // the distance between the centers of these two circles
    Format_t d = norm(c[0]-c[1]);

    // if the distance is too large, then this primitive is not the solution,
    // and we can bail here
    if( d > _4*r )
        return Result_t();

    // the base angle of the isosceles triangle whose vertices are the centers
    // of the the three circles, note acos returns [0,pi]
    Format_t a = -Dispatch::acos( d / (_4*r) );

    // create a clockwise rotation of magnitude alpha
    Rotation2d_t R( Dispatch::sin(a), Dispatch::cos(a) );

    // we find the third vertex of this triangle by taking the vector between
    // the two circle centers, normalizing it, and rotating it by alpha, and
    // scaling it to magnitude 2r, then it points from the center of
    // one the circle tangent to q1 to the third vertex
    Vector2d_t A = normalize( mktmp(c[1]-c[0]) );
    Vector2d_t B = mktmp(R*A) * (_2*r);
    c[2] = c[0] + B;

    // calculate the arc distance we travel on the first circle
    Vector2d_t dc;
    Format_t b,l[3];

    dc = c[2] - c[0];         //< vector between centers of circles
    b  = Dispatch::atan2( get<1>(dc), get<0>(dc) );
                                        //< angle of that vector
    d  = clampRadian( get<2>(q0) - (_PI/_2) );
                                        //< angle of vector from center to q1
    l[0] = ccwArc(d,b);            //< ccwise distance


    // calculate the arc distance we travel on the second circle
    dc = c[2] - c[1];         //< vector between centers of circles
    b  = Dispatch::atan2( get<1>(dc), get<0>(dc) );
                                        //< angle of that vector
    d  = clampRadian( get<2>(q1) - (_PI/_2) );
                                        //< angle of vector from center to q1
    l[1] = ccwArc(b,d);            //< ccwise distance

    // calculate the arc distance we travel on the third circle
    l[2] = _PI + _2*a;

    // sum the resulting segments
    Format_t dist = 0;
#ifdef __CUDACC__
    #pragma unroll
#endif
    for(int i=0; i < 3; i++)
        dist += l[i];

#ifdef __CUDACC__
    #pragma unroll
#endif
    for(int i=0; i < 3; i++)
    {
        out.c[i] = c[i];
        out.l[i] = l[i];
    }

    return Result_t(r*dist);
}



















} // curves
} // dubins
} // mpblocks



#endif // SOLUTIONLRLA_H_
