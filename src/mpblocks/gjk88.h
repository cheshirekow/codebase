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
 *  @file   /home/josh/Codes/cpp/mpblocks2/gjk/include/mpblocks/gjk.h
 *
 *  @date   Sep 14, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_GJK_H_
#define MPBLOCKS_GJK_H_

#include <vector>
#include <set>
#include <cassert>

namespace mpblocks {
namespace    gjk88 {



template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}


template <class Point>
struct MinkowskiPair
{
    Point   a;  //< point from a
    Point   b;  //< point from b
    Point   d;  //< (a-b)

    MinkowskiPair(){}

    MinkowskiPair( const Point& a, const Point& b ):
        a(a),
        b(b),
        d(a-b)
    {}
};


enum Result
{
    COLLISION,
    COLLISION_FREE,
    INDETERMINANT
};


template <class Point>
using PairVec = std::vector< MinkowskiPair<Point> >;


/// returns true if the origin is contained in the segment
template < class Ops, class Point >
bool advanceSimplex2( Ops& ops, PairVec<Point>& simplex, Point& dir  )
{
    Point& b = simplex[0].d;
    Point& a = simplex[1].d;

    Point  ab = b-a;
    Point  ao = -a;

    if( ops.dot( ab,ao ) > 0 )
    {
        // ops.threshold returns true if the origin is sufficiently
        if( ops.threshold( ab,ao ) )
            return true;

        // simplex is unchanged
        dir = ops.cross( ops.cross( ab, ao ), ab );
    }
    else
    {
        simplex[0] = simplex[1];
        simplex.resize(1);
        dir = ao;
    }
    return false;
}

/// returns true if the origin is contained in the triangle
template < class Ops, class Point >
bool advanceSimplex3( Ops& ops, PairVec<Point>& simplex, Point& dir  )
{
    // the points, v0 is the newest
    Point& v2 = simplex[0].d;
    Point& v1 = simplex[1].d;
    Point& v0 = simplex[2].d;

    // rays
    Point  r_01 = v1-v0;
    Point  r_02 = v2-v0;
    Point  r_o  = -v0;

    // [0]
    if( (ops.dot( r_01, r_o ) < 0) && (ops.dot( r_02, r_o ) < 0) )
    {
        simplex[0] = simplex[2];
        simplex.resize(1);
        dir = r_o;
    }
    else
    {
        // normal to the triangle
        Point  f_012 = ops.cross( r_01, r_02 );

        // normals of the two edges
        Point  e_01 = ops.cross( r_01,  f_012 );
        Point  e_02 = ops.cross( f_012, r_02  );

        auto test_e_01 = ops.dot( e_01, r_o );
        auto test_e_02 = ops.dot( e_02, r_o );

        // [3] or [4]
        if( (test_e_01 < 0) && (test_e_02 < 0) )
        {
            // ops.threshold returns true if the origin is sufficiently
            // close to the surface of this simplex, or always in 2D
            if( ops.threshold( v0,v1,v2 ) )
                return true;

            // [3]
            if( ops.dot( f_012, r_o ) > 0 )
            {
                // the simplex is maintained as is for the next iteration
                dir = f_012;
            }
            // [4]
            else
            {
                // flip v1 and v2 so that the picture in our heads is consistent
                std::swap( simplex[0], simplex[1] );
                dir = -f_012;
            }
        }
        // [1], or [2]
        else
        {
            // [1]
            if( test_e_01 > 0 )
            {
                simplex[0] = simplex[1];
                simplex[1] = simplex[2];
                simplex.resize(2);
                dir = ops.cross( ops.cross( r_01, r_o ), r_01 );
            }
            // [2]
            else
            {
                simplex[1] = simplex[2];
                simplex.resize(2);
                dir = ops.cross( ops.cross( r_02, r_o ), r_02 );
            }
        }
    }

    return false;
}


/// returns true if the origin is contained in the tetrahedron
template < class Ops, class Point >
bool advanceSimplex4( Ops& ops, PairVec<Point>& simplex, Point& dir  )
{
    Point& v3 = simplex[0].d;
    Point& v2 = simplex[1].d;
    Point& v1 = simplex[2].d;
    Point& v0 = simplex[3].d;

    Point r3 = v3-v0;
    Point r2 = v2-v0;
    Point r1 = v1-v0;
    Point ro = -v0;

    // exterior
    if( ops.dot(r3,ro) < 0
            && ops.dot(r2,ro) < 0
            && ops.dot(r1,ro) < 0 )
    {
        simplex[0] = simplex[3];
        simplex.resize(1);
        dir = ro;
    }
    else
    {
       Point f_12 = ops.cross(r1,r2);
       Point f_23 = ops.cross(r2,r3);
       Point f_31 = ops.cross(r3,r1);

       auto test_f_12 = ops.dot(f_12,ro);
       auto test_f_23 = ops.dot(f_23,ro);
       auto test_f_31 = ops.dot(f_31,ro);

       // interior
       if( test_f_12 < 0
               && test_f_23 < 0
               && test_f_31 < 0 )
       {
           return true;
       }
       else
       {
           Point e_21 = ops.cross(f_12, r2);
           Point e_23 = ops.cross(r2,   f_23);
           Point e_13 = ops.cross(f_31, r1);
           Point e_12 = ops.cross(r1,   f_12);
           Point e_32 = ops.cross(f_23, r3);
           Point e_31 = ops.cross(r3,   f_31);

           auto test_21 = ops.dot( e_21, ro );
           auto test_23 = ops.dot( e_23, ro );
           auto test_13 = ops.dot( e_13, ro );
           auto test_12 = ops.dot( e_12, ro );
           auto test_32 = ops.dot( e_32, ro );
           auto test_31 = ops.dot( e_31, ro );

           // v is on edge(v0,v1)
           if( test_12 > 0 && test_13 > 0 )
           {
               simplex[0] = simplex[2];
               simplex[1] = simplex[3];
               simplex.resize(2);
               dir = ops.cross( ops.cross (r1, ro ), r1 );
           }
           // wedge 02
           else if( test_23 > 0  && test_21 > 0 )
           {
               simplex[0] = simplex[1];
               simplex[1] = simplex[3];
               simplex.resize(2);
               dir = ops.cross( ops.cross (r2, ro ), r2 );
           }
           // wedge 03
           else if( test_32 > 0 && test_31 > 0)
           {
               simplex[1] = simplex[3];
               simplex.resize(2);
               dir = ops.cross( ops.cross (r3, ro ), r3 );
           }
           // prism 012
           else if( test_f_12 > 0 && test_12 < 0 && test_21 < 0 )
           {
               simplex[0] = simplex[1];
               simplex[1] = simplex[2];
               simplex[2] = simplex[3];
               simplex.resize(3);
               dir = f_12;
           }
           // prism 023
           else if( test_f_23 > 0 && test_23 < 0 && test_32 < 0 )
           {
               simplex[2] = simplex[3];
               simplex.resize(3);
               dir = f_23;
           }
           // prism 031
           else
           {
               simplex[1] = simplex[2];
               simplex[2] = simplex[3];
               simplex.resize(3);
               dir = f_31;
           }

       }
    }

    return false;
}


/// returns true if the origin is contained in the simplex, and if it is
/// not, returns false and generates a new search direction
template < class Ops, class Point >
bool advanceSimplex( Ops& ops, std::vector< MinkowskiPair<Point> >& simplex, Point& dir  )
{
    switch(simplex.size())
    {
        case 2:
            return advanceSimplex2(ops,simplex,dir);
            break;

        case 3:
            return advanceSimplex3(ops,simplex,dir);
            break;

        case 4:
            return advanceSimplex4(ops,simplex,dir);
            break;

        default:
            assert(!"GJK: attempting to advance an overfull simplex");
            break;
    }

    return INDETERMINANT;
}




/// GJK algorithm, determines if two convex objects are in collision
/**
 *
 *  @tparam Ops         class providing vector math, see note
 *  @tparam SupportFn   functional which computes support points, see note
 *  @tparam Point       a three dimensional vector type, must support math
 *                      operators + and -
 *
 *  @param  ops         provides vector math, see note
 *  @param  supportFn   computes support points given a search direction
 *  @param  a[in|out]   initial search point from object A
 *  @param  b[in|out]   initial search point frmo object B
 *  @param  d[out]      output, the last search direction from the algorithm
 *  @param  maxIter     the maximum number of iterations to run
 *  @return either COLLISION, COLLISION_FREE, or INDETERMINANT if maxIter is
 *          reached
 *
 *  @note *Vector Math*:
 *
 *  The point class must support arithmetic operations. For two points a,b the
 *  following must be valid and mean what they usually do:
 *      * a+b
 *      * a-b
 *      * -a
 *
 *  The Ops struct provides dot product and cross product for the point type.
 *  These functions may be static, and ops need not have size (in fact, it
 *  probably shouldn't).
 *
 *  @note *Support Function*:
 *
 *  SupportFn (and the object of it's type @p supportFn) is a functional
 *  (i.e. has an operator() ) which implements the support function for the
 *  object pair. The signature should be something like:
 *
 *      void operator()( const Point& d, Point& a, Point& b )
 *
 *  @p d is the search direction, and @p a and @p b are the function outputs.
 *  They should be the max/min points in the search direction from the two
 *  objects A/B (respectively).
 *
 */
template <class Ops, class SupportFn, class Point>
Result isCollision( Ops ops, SupportFn supportFn,
                    Point& a, Point& b, Point& d,  int maxIter=100 )
{
    typedef MinkowskiPair<Point>   MPair;
    typedef std::vector<MPair>     PointVec;

    PairVec<Point> simplex;

    d = -(a-b);
    simplex.emplace_back( a, b  );

    while( maxIter-- > 0 )
    {
        supportFn(d,a,b);
        if( ops.dot( d,(a-b) ) < 0 )
            return COLLISION_FREE;
        simplex.emplace_back( a, b );

        if( advanceSimplex(ops,simplex,d) )
            return COLLISION;
    }

    return INDETERMINANT;
}























/// returns true if the origin is contained in the segment
template < class Ops, class Point >
bool growSimplex2( Ops& ops, PairVec<Point>& simplex, Point& v  )
{
    Point& b = simplex[0].d;
    Point& a = simplex[1].d;

    Point  ab = b-a;
    Point  ao = -a;
    Point  ab_n = ops.normalize(ab);

    auto d_ab = ops.dot( -a, ab_n );
    if( d_ab > 0 )
    {
        Point c = a + d_ab * ab_n;
        v = c;
    }
    else
    {
        simplex[0] = simplex[1];
        simplex.resize(1);
        v = a;
    }
    return false;
}

/// returns true if the origin is contained in the triangle
template < class Ops, class Point >
bool growSimplex3_2d( Ops& ops, PairVec<Point>& simplex, Point& v  )
{
    // the points, v0 is the newest
    Point& v2 = simplex[0].d;
    Point& v1 = simplex[1].d;
    Point& v0 = simplex[2].d;

    // compute barycentric coordinates
    if( ops.containsOrigin(v0,v1,v2) )
        return true;

    // rays
    Point  r_01 = ops.normalize(v1-v0);
    Point  r_02 = ops.normalize(v2-v0);
    Point  r_o  = -v0;

    // project origin onto each of the sides of the triangle
    auto   p_01 = ops.dot(r_01,r_o);
    auto   p_02 = ops.dot(r_02,r_o);

    // if both are negative then origin is closes to the vertex
    if( p_01 < 0 && p_02 < 0 )
    {
        simplex[0] = simplex[2];
        simplex.resize(1);
        v = v0;
    }

    // otherwise the origin is closer to one of the faces, so see which
    else if( p_01 > 0 )
    {
        v = (v0 + p_01*ops.normalize(r_01));
        simplex.resize(2);
    }
    else
    {
        v = (v0 + p_02*ops.normalize(r_02));
        simplex[1] = simplex[2];
        simplex.resize(2);
    }
    return false;
}

/// returns true if the origin is contained in the triangle
template < class Ops, class Point >
bool growSimplex3_3d( Ops& ops, PairVec<Point>& simplex, Point& v  )
{
    // the points, v0 is the newest
    Point& v2 = simplex[0].d;
    Point& v1 = simplex[1].d;
    Point& v0 = simplex[2].d;

    // rays
    Point  r_01 = ops.normalize(v1-v0);
    Point  r_02 = ops.normalize(v2-v0);
    Point  r_o  = -v0;

    Point  n = ops.normalize( ops.cross(r_01,r_02) );
    auto   f = ops.dot(n,v0);

    // origin, projected onto plane of three points
    Point  o    = f*n;
    Point  r_op = o - v0;

    // find cross product of the two edges of the triangle and of the vector
    // from v0 to the projection of the origin onto the plane, and then
    // use these to determine if projection of the origin lies inside the
    // triangle
//    Point x1 = ops.normalize( ops.cross(v0-o,v1-o) );
//    Point x2 = ops.normalize( ops.cross(v1-o,v2-o) );
//    Point x3 = ops.normalize( ops.cross(v2-o,v0-o) );

    // if the projection of the origin onto the plane of the three points lies
    // inside the triangle, then that point is the nearest point to the origin,
    // and the simplex remains as is
//    if( sgn( ops.dot(x1,x2) ) == sgn( ops.dot(x2,x3) )
//        && sgn( ops.dot(x2,x3) ) == sgn( ops.dot(x3,x1) ) )
//        v = o;

    // find cross product of the two edges of the triangle and of the vector
    // from v0 to the projection of the origin onto the plane, and then
    // use these to determine if projection of the origin lies inside the
    // triangle
    Point bxo = ops.normalize( ops.cross(r_01,r_op) );
    Point oxc = ops.normalize( ops.cross(r_op,r_02) );
    Point bxc = ops.normalize( ops.cross(r_01,r_02) );

    // if the projection of the origin onto the plane of the three points lies
    // inside the triangle, then that point is the nearest point to the origin,
    // and the simplex remains as is
    if( sgn( ops.dot(bxo,oxc) ) > 0 && sgn( ops.dot(bxo,bxc) ) > 0 )
        v = o;
    else
    {
        // project origin onto each of the sides of the triangle
        auto   p_01 = ops.dot(r_01,r_o);
        auto   p_02 = ops.dot(r_02,r_o);

        // if both are negative then origin is closes to the vertex
        if( p_01 < 0 && p_02 < 0 )
        {
            simplex[0] = simplex[2];
            simplex.resize(1);
            v = v0;
        }

        // otherwise the origin is closer to one of the faces, so see which
        else if( p_01 > 0 )
        {
            v = (v0 + p_01*ops.normalize(r_01));
            simplex[0] = simplex[1];
            simplex[1] = simplex[2];
            simplex.resize(2);
        }
        else
        {
            v = (v0 + p_02*ops.normalize(r_02));
            simplex[1] = simplex[2];
            simplex.resize(2);
        }
    }
    return false;
}


/// returns true if the origin is contained in the tetrahedron
template < class Ops, class Point >
bool growSimplex4_3d(
        Ops& ops, PairVec<Point>& simplex, Point& v, int depth=0  )
{
    Point& v3 = simplex[0].d;
    Point& v2 = simplex[1].d;
    Point& v1 = simplex[2].d;
    Point& v0 = simplex[3].d;

    // if triangle(v1,v2,v3) points in the same direction as
    // v0 then we need to flip the order
    Point r12 = ops.normalize(v2-v1);
    Point r13 = ops.normalize(v3-v1);
    if( ops.dot( ops.cross(r12,r13), v0-v1 ) < 0 )
    {
        std::swap(simplex[0],simplex[2]);
        if( depth > 3 )
        {
            v = v0;
            simplex[0] = simplex[3];
            simplex.resize(1);
            return false;
        }
        return growSimplex4_3d(ops,simplex,v,depth+1);
    }

    Point r3 = ops.normalize(v3-v0);
    Point r2 = ops.normalize(v2-v0);
    Point r1 = ops.normalize(v1-v0);
    Point ro = -v0;

    // projection of origin onto each of the edges
    auto  p_1 = ops.dot(ro,r1);
    auto  p_2 = ops.dot(ro,r2);
    auto  p_3 = ops.dot(ro,r3);

    // if the projection of the origin onto all edges is in the negative
    // direction then v is v0 and the new simplex is just v0
    if( p_1 < 0 && p_2 < 0 && p_3 < 0 )
    {
        simplex[0] = simplex[3];
        simplex.resize(1);
        v = v0;
    }
    else
    {
        // compute the normal for each of the three new tetrahedral faces,
        // ignore the fourth face which was pruned by the previous simplex
        // expansion
        Point n_12 = ops.normalize( ops.cross(r1,r2) );
        Point n_23 = ops.normalize( ops.cross(r2,r3) );
        Point n_31 = ops.normalize( ops.cross(r3,r1) );

        auto f_12 = ops.dot(n_12,ro);
        auto f_23 = ops.dot(n_23,ro);
        auto f_31 = ops.dot(n_31,ro);

        // if all faces are oriented outward, then the origin is interior
        // to the simplex
        if( f_12 < 0 && f_23 < 0 && f_31 < 0 )
            return true;
        else
        {
            // For each edge of the tetrahedron, compute two normals, each one
            // pointing "outside" of a face, for each of the two faces that
            // the edge is a member of
            Point n_12_2 = ops.normalize( ops.cross(n_12, r2)   );
            Point n_23_2 = ops.normalize( ops.cross(r2,   n_23) );
            Point n_31_1 = ops.normalize( ops.cross(n_31, r1)   );
            Point n_12_1 = ops.normalize( ops.cross(r1,   n_12) );
            Point n_23_3 = ops.normalize( ops.cross(n_23, r3)   );
            Point n_31_3 = ops.normalize( ops.cross(r3,   n_31) );

            // also compute the projection of ray(v0,o) onto that normal
            auto f_12_2 = ops.dot( n_12_2, ro );
            auto f_23_2 = ops.dot( n_23_2, ro );
            auto f_31_1 = ops.dot( n_31_1, ro );
            auto f_12_1 = ops.dot( n_12_1, ro );
            auto f_23_3 = ops.dot( n_23_3, ro );
            auto f_31_3 = ops.dot( n_31_3, ro );

           // v is on edge(v0,v1)
           if( f_12_1 > 0 && f_31_1 > 0 )
           {
               simplex[0] = simplex[2];
               simplex[1] = simplex[3];
               simplex.resize(2);
               v = v0 + p_1*r1;
           }

           // v is on edge(v0,v2)
           else if( f_12_2 > 0  && f_23_2 > 0 )
           {
               simplex[0] = simplex[1];
               simplex[1] = simplex[3];
               simplex.resize(2);
               v = v0 + p_2*r2;
           }

           // v is on edge(v0,v3)
           else if( f_31_3 > 0 && f_23_3 > 0)
           {
               simplex[1] = simplex[3];
               simplex.resize(2);
               v = v0 + p_3*r3;
           }

           // v is on face(v0,v1,v2)
           else if( f_12 > 0 && f_12_1 < 0 && f_12_2 < 0 )
           {
               simplex[0] = simplex[1];
               simplex[1] = simplex[2];
               simplex[2] = simplex[3];
               simplex.resize(3);
               v = ops.dot(v0,n_12)*n_12;
           }

           // v is on face(v0,v2,v3)
           else if( f_23 > 0 && f_23_2 < 0 && f_23_3 < 0 )
           {
               simplex[2] = simplex[3];
               simplex.resize(3);
               v = ops.dot(v0,n_23)*n_23;
           }
           // v is on face(v0,v3,v1)
           else
           {
               simplex[1] = simplex[2];
               simplex[2] = simplex[3];
               simplex.resize(3);
               v = ops.dot(v0,n_31)*n_31;
           }
       }
    }

    return false;
}



/// returns true if the origin is contained in the simplex, and if it is
/// not, returns false and generates a new search direction
template < class Ops, class Point >
bool growSimplex_2d( Ops& ops, std::vector< MinkowskiPair<Point> >& simplex, Point& v  )
{
    switch(simplex.size())
    {
        case 2:
            return growSimplex2(ops,simplex,v);
            break;

        case 3:
            return growSimplex3_2d(ops,simplex,v);
            break;

        default:
            assert(!"GJK: attempting to advance an overfull simplex");
            break;
    }
    return false;
}

/// returns true if the origin is contained in the simplex, and if it is
/// not, returns false and generates a new search direction
template < class Ops, class Point >
bool growSimplex_3d(
        Ops& ops, std::vector< MinkowskiPair<Point> >& simplex, Point& v  )
{
    switch(simplex.size())
    {
        case 2:
            return growSimplex2(ops,simplex,v);
            break;

        case 3:
            return growSimplex3_3d(ops,simplex,v);
            break;

        case 4:
            return growSimplex4_3d(ops,simplex,v);
            break;

        default:
            assert(!"GJK: attempting to advance an overfull simplex");
            break;
    }
    return false;
}








template <class Ops, class SupportFn, class Point,
            class SimplexHistory,
            class PointHistory>
Result collisionDistance_2d_debug( Ops ops, SupportFn supportFn,
                    Point& a, Point& b, Point& v,
                    SimplexHistory sHist,
                    PointHistory   vHist, int maxIter=100 )
{
    typedef MinkowskiPair<Point>   MPair;
    PairVec<Point> simplex;

    simplex.reserve(4);
    simplex.push_back( MPair(a,b) );
    v = (a-b);

    *(sHist++) = simplex;
    *(vHist++) = v;

    while( maxIter-- > 0 )
    {
        supportFn(-v,a,b);
        if( std::abs( ops.dot(v,v) + ops.dot( -v,(a-b) ) ) < 1e-6 )
            return COLLISION_FREE;

        simplex.push_back( MPair(a,b) );
        *(sHist++) = simplex;
        bool isCollision = growSimplex_2d(ops,simplex,v);
        *(vHist++) = v;

        if( isCollision )
            return COLLISION;
    }

    return INDETERMINANT;
}


template< int NDim, class Point >
bool isEqual( const Point& a, const Point& b )
{
    for(int i=0; i < NDim; i++)
        if( a[i] != b[i] )
            return false;
    return true;
}


template <class Ops, class SupportFn, class Point>
Result collisionDistance_3d( Ops ops, SupportFn supportFn,
                    Point& a, Point& b, Point& v, int maxIter=100 )
{
    typedef MinkowskiPair<Point>   MPair;
    PairVec<Point> simplex;

    simplex.reserve(4);
    simplex.push_back( MPair(a,b) );
    v = (a-b);

    while( maxIter-- > 0 )
    {
        supportFn(-v,a,b);
        if( std::abs( ops.dot(v,v) + ops.dot( -v,(a-b) ) ) < 1e-6 )
            return COLLISION_FREE;

        MPair newPair(a,b);
        for( auto& m : simplex )
            if( isEqual<3>(m.d,newPair.d) )
                return COLLISION_FREE;

        simplex.push_back( newPair );
        if( growSimplex_3d(ops,simplex,v) )
            return COLLISION;
    }

    return INDETERMINANT;
}




template <class Ops, class SupportFn, class Point,
            class SimplexHistory,
            class PointHistory,
            class ErrorHistory>
Result collisionDistance_3d_debug( Ops ops, SupportFn supportFn,
                    Point& a, Point& b, Point& v,
                    SimplexHistory sHist,
                    PointHistory   vHist,
                    ErrorHistory   eHist, int maxIter=100 )
{
    typedef MinkowskiPair<Point>   MPair;
    PairVec<Point> simplex;

    simplex.reserve(4);
    simplex.push_back( MPair(a,b) );
    v = (a-b);

    *(sHist++) = simplex;
    *(vHist++) = v;
    *(eHist++) = 0;

    while( maxIter-- > 0 )
    {
        supportFn(-v,a,b);
        auto e = ops.dot(v,v) + ops.dot( -v,(a-b) );
        *(eHist++) = e;
        if( std::abs( e ) < 1e-6 )
        {
            *(sHist++) = simplex;
            *(vHist++) = v;
            return COLLISION_FREE;
        }

        MPair newPair(a,b);
        for( auto& m : simplex )
        {
            if( isEqual<3>(m.d,newPair.d) )
            {
                *(sHist++) = simplex;
                *(vHist++) = v;
                return COLLISION_FREE;
            }
        }
        simplex.push_back( newPair );
        *(sHist++) = simplex;

        bool isCollision = growSimplex_3d(ops,simplex,v);
        *(vHist++) = v;

        if( isCollision )
            return COLLISION;
    }

    return INDETERMINANT;
}









} //< namespace gjk88
} //< namespace mpblocks















#endif // GJK_H_
