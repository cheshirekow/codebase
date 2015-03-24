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

#ifndef MPBLOCKS_CUDA_POLYNOMIAL_STURMSEQUENCE_H_
#define MPBLOCKS_CUDA_POLYNOMIAL_STURMSEQUENCE_H_

#include <iostream>
#include <vector>
#include <list>

#include <mpblocks/cuda/polynomial/divide.h>

namespace   mpblocks {
namespace       cuda {
namespace polynomial {

template < typename Scalar, int max>
struct SturmSequence;

namespace sturm_detail
{

/// signum
template <typename T>
__host__ __device__ __forceinline__
int sgn(T val)
{
    return (T(0) < val) - (val < T(0));
}


/// Storage class, stores one polynomial in teh sequence
template < typename Scalar, int size>
struct Storage:
    public Storage<Scalar,size-1>
{
    Polynomial< Scalar, typename intlist::range<0,size>::result > poly;
};

template < typename Scalar>
struct Storage<Scalar,-1>{};


template <class Scalar, int i, int max>
struct RebuildHelper;

template <class Scalar, int i, int max>
struct SignChangeHelper;

template <class Scalar, int max>
__host__ __device__ __forceinline__
void rebuild( SturmSequence<Scalar,max>& sturm  );

template <class Scalar, int max>
__host__ __device__ __forceinline__
int signChanges( SturmSequence<Scalar,max>& sturm, Scalar s, Scalar prev );


} // namespace sturm_detail


/// convienience accessor, returns the i'th polynomial in the sequence, where
/// the 0'th polynomial is the original
template < int i, typename Scalar, int max >
__host__ __device__ __forceinline__
Polynomial<Scalar, typename intlist::range<0,max-i>::result>&
    get( SturmSequence<Scalar,max>& sturm);



/// stores a squence of polynomials which satisfy the properties of
/// sturms theorem and provides methods for computing the number of sign
/// changes
template < typename Scalar, int max>
struct SturmSequence:
    public sturm_detail::Storage<Scalar, max >
{
    __host__ __device__ __forceinline__
    SturmSequence()
    {}

    /// builds a sturm sequence from a polynomial
    template < class Exp, class Spec >
    __host__ __device__ __forceinline__
    SturmSequence( const RValue<Scalar,Exp,Spec>& rhs )
    {
        rebuild(rhs);
    }

    template <class Exp, class Spec>
    __host__ __device__ __forceinline__
    void rebuild( const RValue<Scalar,Exp,Spec>& rhs  )
    {
        typedef typename intlist::range<0,max-1>::result Spec1;
        typedef Polynomial< Scalar, Spec1 > Poly1;
        Poly1  p1  = d_ds<1>( rhs );

//        std::cout << "Building sturm sequence:\n";
        get<0>(*this) = normalized(rhs);
//        std::cout << "\n  spec: " << Printer<Spec>()
//                  << "\n  max : " << intlist::max<Spec>::value
//                  << "\n input: " << rhs
//                  << "\n  d/ds: " << p1
//                  << "\n    p0: " << get<0>(*this);
        get<1>(*this) = normalized(p1);
//        std::cout << "\n    p0: " << get<0>(*this)
//                  << "\n    p1: " << get<1>(*this)
//                  << "\n";

        sturm_detail::rebuild(*this);
    }

    /// return the number of sign changes at the specified point
    __host__ __device__ __forceinline__
    int signChanges( Scalar s )
    {
        Scalar y0 = polyval( get<0>(*this),s );
        return sturm_detail::signChanges(*this,s,y0);
    }
};


template < int i, typename Scalar, int max >
__host__ __device__ __forceinline__
Polynomial<Scalar, typename intlist::range<0,max-i>::result>&
    get( SturmSequence<Scalar,max>& sturm)
{
    return static_cast< sturm_detail::Storage<Scalar,max-i>& >(sturm).poly;
}


namespace sturm_detail
{

template <class Scalar, int i, int size>
struct RebuildHelper
{
    enum{ max = size-1 };
    __host__ __device__ __forceinline__
    static void rebuild( SturmSequence<Scalar,max>& sturm )
    {
        typedef typename intlist::range<0,max-(i-2)>::result Spec1;
        typedef typename intlist::range<0,max-(i-1)>::result Spec2;

        typedef Quotient<Scalar,
                Polynomial<Scalar,Spec1>,Spec1,
                Polynomial<Scalar,Spec2>,Spec2 > Quotient_t;
        Quotient_t q( get<i-2>(sturm) / get<i-1>(sturm) );
        get<i>(sturm) = -q.r;

//        std::cout << "Building sturm seq " << i ;
//        std::cout << "\np[" << (i-2) << "] : " << get<i-2>(sturm)
//                  << "\np[" << (i-1) << "] : " << get<i-1>(sturm)
//                  << "\n    q : " << q.q
//                  << "\n    r : " << q.r
//                  << "\np[" << i << "] : " << get<i>(sturm)
//                  << "\n\n";

        RebuildHelper<Scalar,i+1,size>::rebuild(sturm);
    }
};


template <class Scalar, int size>
struct RebuildHelper<Scalar,size,size>
{
    enum{ max = size-1 };
    __host__ __device__ __forceinline__
    static void rebuild( SturmSequence<Scalar,size-1>& sturm )
    {
        typedef typename intlist::range<0,2>::result Spec1;
        typedef typename intlist::range<0,1>::result Spec2;

        typedef Quotient<Scalar,
                Polynomial<Scalar,Spec1>,Spec1,
                Polynomial<Scalar,Spec2>,Spec2 > Quotient_t;
        Quotient_t q( get<max-2>(sturm) / get<max-1>(sturm) );
        get<max>(sturm) = -q.r;

//        std::cout << "Building sturm seq " << i ;
//        std::cout << "\np[" << (i-2) << "] : " << get<i-2>(sturm)
//                  << "\np[" << (i-1) << "] : " << get<i-1>(sturm)
//                  << "\n    q : " << q.q
//                  << "\n    r : " << q.r
//                  << "\np[" << i << "] : " << get<i>(sturm)
//                  << "\n\n";
    }
};

template <class Scalar, int i, int size>
struct SignChangeHelper
{
    enum{ max = size-1 };
    __host__ __device__ __forceinline__
    static int count( SturmSequence<Scalar,size-1>& sturm, Scalar s, Scalar prev )
    {
        Scalar next = polyval( get<i>(sturm), s );
        if( sgn(next) )
        {
            return ( sgn(next) == sgn(prev) ? 0 : 1 )
                + SignChangeHelper<Scalar,i+1,size>::count(sturm,s,next);
        }
        else
            return SignChangeHelper<Scalar,i+1,size>::count(sturm,s,prev);
    }
};

template <class Scalar, int size>
struct SignChangeHelper<Scalar,size,size>
{
    enum{ max = size-1 };
    __host__ __device__ __forceinline__
    static int count(SturmSequence<Scalar,max>& sturm, Scalar s, Scalar prev)
    {
        return 0;
    }
};

template <class Scalar, int max>
__host__ __device__ __forceinline__
void rebuild( SturmSequence<Scalar,max>& sturm  )
{
    RebuildHelper<Scalar,2,max+1>::rebuild(sturm);
}

template <class Scalar, int max>
__host__ __device__ __forceinline__
int signChanges( SturmSequence<Scalar,max>& sturm, Scalar s, Scalar y0)
{
    return SignChangeHelper<Scalar,1,max+1>::count(sturm,s, y0);
}


} // namespace sturm_detail









} // namespace polynomial
} // namespace cuda
} // namespace mpblocks
















#endif // STURMSEQUENCE_H_
