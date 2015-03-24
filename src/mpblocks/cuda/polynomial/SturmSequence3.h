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
 *  @file   include/mpblocks/polynomial/SturmSequence.h
 *
 *  @date   Jan 15, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_CUDA_POLYNOMIAL_STURMSEQUENCE3_H_
#define MPBLOCKS_CUDA_POLYNOMIAL_STURMSEQUENCE3_H_

#include <iostream>
#include <vector>
#include <list>

#include <mpblocks/cuda/polynomial/SturmSequence.h>
#include <mpblocks/cuda/polynomial/SturmSequence2.h>
#include <mpblocks/cuda/polynomial/divide.h>
#include <mpblocks/cuda/polynomial/Derivative.h>


namespace   mpblocks {
namespace       cuda {
namespace polynomial {

namespace sturm3_detail
{


}


template< typename Scalar, class Spec >
struct SturmTypes
{
    typedef Spec Spec0;
    typedef typename sturm2_detail::SturmSpec<Spec,1>::result Spec1;
    typedef typename sturm2_detail::SturmSpec<Spec,2>::result Spec2;
    typedef typename sturm2_detail::SturmSpec<Spec,3>::result Spec3;
    typedef typename sturm2_detail::SturmSpec<Spec,4>::result Spec4;
    typedef typename sturm2_detail::SturmSpec<Spec,5>::result Spec5;
    typedef typename sturm2_detail::SturmSpec<Spec,6>::result Spec6;
    typedef typename sturm2_detail::SturmSpec<Spec,7>::result Spec7;
    typedef typename sturm2_detail::SturmSpec<Spec,8>::result Spec8;
    typedef typename sturm2_detail::SturmSpec<Spec,9>::result Spec9;

    typedef Polynomial<Scalar,Spec0> Poly0;
    typedef Polynomial<Scalar,Spec1> Poly1;
    typedef Polynomial<Scalar,Spec2> Poly2;
    typedef Polynomial<Scalar,Spec3> Poly3;
    typedef Polynomial<Scalar,Spec4> Poly4;
    typedef Polynomial<Scalar,Spec5> Poly5;
    typedef Polynomial<Scalar,Spec6> Poly6;
    typedef Polynomial<Scalar,Spec7> Poly7;
    typedef Polynomial<Scalar,Spec8> Poly8;
    typedef Polynomial<Scalar,Spec9> Poly9;
};


template< int size>
struct CountZeros
{
    template< typename Scalar, class Spec >
    __host__ __device__
    static int count_zeros( const Polynomial<Scalar,Spec>& poly )
    {
        return -1;
    }
};

template <>
struct CountZeros<1>
{
    template< typename Scalar, class Spec >
    __host__ __device__
    static int count_zeros( const Polynomial<Scalar,Spec>& poly,
                            Scalar s0, Scalar s1 )
    {
        typedef SturmTypes<Scalar,Spec> types;

        int count0 = 0;
        int count1 = 1;

        Scalar val0a, val0b, val1a, val1b;
        typename types::Poly0 p0 = normalized(poly);
        val0a = polyval(p0,s0);
        val1a = polyval(p0,s1);

        typename types::Poly1 p1 = normalized( d_dx(poly) );
        val0b = polyval(p1,s0);
        val1b = polyval(p1,s1);
        if( sturm_detail::sgn(val0a) != sturm_detail::sgn(val0b) )
            count0++;
        if( sturm_detail::sgn(val1a) != sturm_detail::sgn(val1b) )
            count1++;
        val0a = val0b;
        val1a = val1b;

        return abs(count1-count0);
    }
};

template <>
struct CountZeros<2>
{
    template< typename Scalar, class Spec >
    __host__ __device__
    static int count_zeros( const Polynomial<Scalar,Spec>& poly,
                            Scalar s0, Scalar s1 )
    {
        typedef SturmTypes<Scalar,Spec> types;

        int count0 = 0;
        int count1 = 1;

        Scalar val0a, val0b, val1a, val1b;
        typename types::Poly0 p0 = normalized(poly);
        val0a = polyval(p0,s0);
        val1a = polyval(p0,s1);

        typename types::Poly1 p1 = normalized( d_dx(poly) );
        val0b = polyval(p1,s0);
        val1b = polyval(p1,s1);
        if( sturm_detail::sgn(val0a) != sturm_detail::sgn(val0b) )
            count0++;
        if( sturm_detail::sgn(val1a) != sturm_detail::sgn(val1b) )
            count1++;
        val0a = val0b;
        val1a = val1b;

        typename types::Poly2 p2,p2temp;
        mod(p0,p1,p2temp);
        p2 = normalized( p2temp );
        val0b = polyval(p2,s0);
        val1b = polyval(p2,s1);
        if( sturm_detail::sgn(val0a) != sturm_detail::sgn(val0b) )
            count0++;
        if( sturm_detail::sgn(val1a) != sturm_detail::sgn(val1b) )
            count1++;
        val0a = val0b;
        val1a = val1b;

        return abs(count1-count0);
    }
};


template <>
struct CountZeros<3>
{
    template< typename Scalar, class Spec >
    __host__ __device__
    static int count_zeros( const Polynomial<Scalar,Spec>& poly,
                            Scalar s0, Scalar s1 )
    {
        typedef SturmTypes<Scalar,Spec> types;

        int count0 = 0;
        int count1 = 0;

        Scalar val0a, val0b, val1a, val1b;
        typename types::Poly0 p0 = normalized(poly);
        val0a = polyval(p0,s0);
        val1a = polyval(p0,s1);

        typename types::Poly1 p1 = normalized( d_dx(poly) );
        val0b = polyval(p1,s0);
        val1b = polyval(p1,s1);
        if( sturm_detail::sgn(val0b) )
        {
            if( sturm_detail::sgn(val0a) != sturm_detail::sgn(val0b) )
                count0++;
            val0a = val0b;
        }

        if( sturm_detail::sgn(val1b) )
        {
            if( sturm_detail::sgn(val1a) != sturm_detail::sgn(val1b) )
                count1++;
            val1a = val1b;
        }

        typename types::Poly2 p2,p2temp;
        mod(p0,p1,p2temp);
        p2 = normalized( p2temp );
        val0b = polyval(p2,s0);
        val1b = polyval(p2,s1);
        if( sturm_detail::sgn(val0b) )
        {
            if( sturm_detail::sgn(val0a) != sturm_detail::sgn(val0b) )
                count0++;
            val0a = val0b;
        }

        if( sturm_detail::sgn(val1b) )
        {
            if( sturm_detail::sgn(val1a) != sturm_detail::sgn(val1b) )
                count1++;
            val1a = val1b;
        }

        typename types::Poly3 p3,p3temp;
        mod(p1,p2,p3temp);
        p3 = normalized( p3temp );
        val0b = polyval(p3,s0);
        val1b = polyval(p3,s1);
        if( sturm_detail::sgn(val0b) )
        {
            if( sturm_detail::sgn(val0a) != sturm_detail::sgn(val0b) )
                count0++;
            val0a = val0b;
        }

        if( sturm_detail::sgn(val1b) )
        {
            if( sturm_detail::sgn(val1a) != sturm_detail::sgn(val1b) )
                count1++;
            val1a = val1b;
        }

        return abs(count1-count0);
    }
};

template <>
struct CountZeros<4>
{
    template< typename Scalar, class Spec >
    __host__ __device__
    static int count_zeros( const Polynomial<Scalar,Spec>& poly,
                            Scalar s0, Scalar s1 )
    {
        typedef SturmTypes<Scalar,Spec> types;

        int count0 = 0;
        int count1 = 1;

        Scalar val0a, val0b, val1a, val1b;
        typename types::Poly0 p0 = normalized(poly);
        val0a = polyval(p0,s0);
        val1a = polyval(p0,s1);

        typename types::Poly1 p1 = normalized( d_dx(poly) );
        val0b = polyval(p1,s0);
        val1b = polyval(p1,s1);
        if( sturm_detail::sgn(val0a) != sturm_detail::sgn(val0b) )
            count0++;
        if( sturm_detail::sgn(val1a) != sturm_detail::sgn(val1b) )
            count1++;
        val0a = val0b;
        val1a = val1b;

        typename types::Poly2 p2,p2temp;
        mod(p0,p1,p2temp);
        p2 = normalized( p2temp );
        val0b = polyval(p2,s0);
        val1b = polyval(p2,s1);
        if( sturm_detail::sgn(val0a) != sturm_detail::sgn(val0b) )
            count0++;
        if( sturm_detail::sgn(val1a) != sturm_detail::sgn(val1b) )
            count1++;
        val0a = val0b;
        val1a = val1b;

        typename types::Poly3 p3,p3temp;
        mod(p1,p2,p3temp);
        p3 = normalized( p3temp );
        val0b = polyval(p3,s0);
        val1b = polyval(p3,s1);
        if( sturm_detail::sgn(val0a) != sturm_detail::sgn(val0b) )
            count0++;
        if( sturm_detail::sgn(val1a) != sturm_detail::sgn(val1b) )
            count1++;
        val0a = val0b;
        val1a = val1b;

        typename types::Poly4 p4,p4temp;
        mod(p2,p3,p4temp);
        p4 = normalized( p4temp );
        val0b = polyval(p4,s0);
        val1b = polyval(p4,s1);
        if( sturm_detail::sgn(val0a) != sturm_detail::sgn(val0b) )
            count0++;
        if( sturm_detail::sgn(val1a) != sturm_detail::sgn(val1b) )
            count1++;
        val0a = val0b;
        val1a = val1b;

        return abs(count1-count0);
    }
};



template <>
struct CountZeros<5>
{
    template< typename Scalar, class Spec >
    __host__ __device__
    static int count_zeros( const Polynomial<Scalar,Spec>& poly,
                            Scalar s0, Scalar s1 )
    {
        typedef SturmTypes<Scalar,Spec> types;

        int count0 = 0;
        int count1 = 0;

        Scalar val0a, val0b, val1a, val1b;
        typename types::Poly0 p0 = normalized(poly);
        val0a = polyval(p0,s0);
        val1a = polyval(p0,s1);

        typename types::Poly1 p1 = normalized( d_dx(poly) );
        val0b = polyval(p1,s0);
        val1b = polyval(p1,s1);
        if( sturm_detail::sgn(val0b) )
        {
            if( sturm_detail::sgn(val0a) != sturm_detail::sgn(val0b) )
                count0++;
            val0a = val0b;
        }

        if( sturm_detail::sgn(val1b) )
        {
            if( sturm_detail::sgn(val1a) != sturm_detail::sgn(val1b) )
                count1++;
            val1a = val1b;
        }

        typename types::Poly2 p2,p2temp;
        mod(p0,p1,p2temp);
        p2 = normalized( p2temp );
        val0b = polyval(p2,s0);
        val1b = polyval(p2,s1);
        if( sturm_detail::sgn(val0b) )
        {
            if( sturm_detail::sgn(val0a) != sturm_detail::sgn(val0b) )
                count0++;
            val0a = val0b;
        }

        if( sturm_detail::sgn(val1b) )
        {
            if( sturm_detail::sgn(val1a) != sturm_detail::sgn(val1b) )
                count1++;
            val1a = val1b;
        }

        typename types::Poly3 p3,p3temp;
        mod(p1,p2,p3temp);
        p3 = normalized( p3temp );
        val0b = polyval(p3,s0);
        val1b = polyval(p3,s1);
        if( sturm_detail::sgn(val0b) )
        {
            if( sturm_detail::sgn(val0a) != sturm_detail::sgn(val0b) )
                count0++;
            val0a = val0b;
        }

        if( sturm_detail::sgn(val1b) )
        {
            if( sturm_detail::sgn(val1a) != sturm_detail::sgn(val1b) )
                count1++;
            val1a = val1b;
        }

        typename types::Poly4 p4,p4temp;
        mod(p2,p3,p4temp);
        p4 = normalized( p4temp );
        val0b = polyval(p4,s0);
        val1b = polyval(p4,s1);
        if( sturm_detail::sgn(val0b) )
        {
            if( sturm_detail::sgn(val0a) != sturm_detail::sgn(val0b) )
                count0++;
            val0a = val0b;
        }

        if( sturm_detail::sgn(val1b) )
        {
            if( sturm_detail::sgn(val1a) != sturm_detail::sgn(val1b) )
                count1++;
            val1a = val1b;
        }

        typename types::Poly5 p5,p5temp;
        mod(p3,p4,p5temp);
        p5 = normalized( p5temp );
        val0b = polyval(p5,s0);
        val1b = polyval(p5,s1);
        if( sturm_detail::sgn(val0b) )
        {
            if( sturm_detail::sgn(val0a) != sturm_detail::sgn(val0b) )
                count0++;
            val0a = val0b;
        }

        if( sturm_detail::sgn(val1b) )
        {
            if( sturm_detail::sgn(val1a) != sturm_detail::sgn(val1b) )
                count1++;
            val1a = val1b;
        }

        return abs(count1-count0);
    }
};




} // namespace polynomial
} // namespace cuda
} // namespace mpblocks
















#endif // STURMSEQUENCE_H_
