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

#ifndef MPBLOCKS_CUDA_POLYNOMIAL_STURMSEQUENCE2_H_
#define MPBLOCKS_CUDA_POLYNOMIAL_STURMSEQUENCE2_H_

#include <iostream>
#include <vector>
#include <list>

#include <mpblocks/cuda/polynomial/SturmSequence.h>
#include <mpblocks/cuda/polynomial/divide.h>
#include <mpblocks/cuda/polynomial/Derivative.h>


namespace   mpblocks {
namespace       cuda {
namespace polynomial {

namespace sturm2_detail
{

/// returns the the specification of the i'th polynomial in the sequence
template< class Spec, int idx >
struct SturmSpec
{
    typedef typename SturmSpec<Spec,idx-2>::result Spec2;
    typedef typename SturmSpec<Spec,idx-1>::result Spec1;
    typedef typename RemainderSpec< Spec2, Spec1 >::result result;
};

template< class Spec >
struct SturmSpec<Spec,1>
{
    typedef typename DerivativeSpec<Spec,1>::result result;
};

template< class Spec >
struct SturmSpec<Spec,0>
{
    typedef Spec result;
};

template< class Spec, int idx >
struct Index
{
    enum
    {
        max_coeff = intlist::max<Spec>::value,
        enabled   = (idx <= max_coeff)
    };
};



/// Storage class, stores one polynomial in teh sequence
template < bool Enabled, typename Scalar, class Spec, int idx >
struct Storage:
    public Storage< Index<Spec,idx+1>::enabled, Scalar, Spec, idx+1 >
{
    typedef typename SturmSpec<Spec,idx>::result ThisSpec;
    Polynomial< Scalar, ThisSpec > poly;
};

/// end of recursion
template < typename Scalar, class Spec, int idx >
struct Storage<false,Scalar,Spec,idx>
{};


} // namespace sturm2_detail


/// stores a squence of polynomials which satisfy the properties of
/// sturms theorem and provides methods for computing the number of sign
/// changes
template < typename Scalar, class Spec>
struct SturmSequence2:
    public sturm2_detail::Storage<true,Scalar,Spec,0>
{};


/// return the i'th polynomial in the sturm sequence
template < int i, typename Scalar, class Spec >
__host__ __device__ __forceinline__
Polynomial<Scalar, typename sturm2_detail::SturmSpec<Spec,i>::result>&
    get( SturmSequence2<Scalar,Spec>& sturm)
{
    return sturm.sturm2_detail::Storage<true,Scalar,Spec,i>::poly;
}

template < int i, typename Scalar, class Spec >
__host__ __device__ __forceinline__
const Polynomial<Scalar, typename sturm2_detail::SturmSpec<Spec,i>::result>&
    get( const SturmSequence2<Scalar,Spec>& sturm)
{
    return sturm.sturm2_detail::Storage<true,Scalar,Spec,i>::poly;
}


namespace sturm2_detail
{
    template < bool Enable, typename Scalar, class Spec, int idx>
    struct Construct
    {
        __host__ __device__  __forceinline__
        static void step( SturmSequence2<Scalar,Spec>& sturm ){}
    };

    template < typename Scalar, class Spec, int idx>
    struct Construct<true,Scalar,Spec,idx>
    {
        __host__ __device__  __forceinline__
        static void step( SturmSequence2<Scalar,Spec>& sturm )
        {
            typedef typename SturmSpec<Spec,idx-2>::result NumSpec;
            typedef typename SturmSpec<Spec,idx-1>::result DenSpec;
            typedef typename SturmSpec<Spec,idx>::result   ModSpec;

            typedef Polynomial<Scalar,NumSpec> NumExp;
            typedef Polynomial<Scalar,DenSpec> DenExp;

            mod(
                sturm.Storage<true,Scalar,Spec,idx-2>::poly,
                sturm.Storage<true,Scalar,Spec,idx-1>::poly,
                sturm.Storage<true,Scalar,Spec,idx>::poly );
        }
    };

    template < bool Enable, typename Scalar, class Spec, int idx>
    struct CountSignChanges
    {
        __host__ __device__  __forceinline__
        static void step( SturmSequence2<Scalar,Spec>& sturm,
                            Scalar arg, Scalar& prevVal, int& count){}
    };

    template < typename Scalar, class Spec, int idx>
    struct CountSignChanges<true,Scalar,Spec,idx>
    {
        __host__ __device__  __forceinline__
        static void step( SturmSequence2<Scalar,Spec>& sturm,
                            Scalar arg, Scalar& prevVal, int& count )
        {
            using sturm_detail::sgn;
            Scalar nextVal = polyval( get<idx>(sturm), arg );
            if( sgn(nextVal) )
            {
                if( sgn(nextVal) != sgn(prevVal) )
                    count++;
                prevVal = nextVal;
            }
        }
    };
}




/// construct a sturm sequence
template < typename Scalar, class Exp, class Spec>
__host__ __device__ __forceinline__
void construct( SturmSequence2<Scalar,Spec>& sturm,
                const Exp& exp )
{
    assign( sturm.sturm2_detail::Storage<true,Scalar,Spec,0>::poly, exp );
    assign( sturm.sturm2_detail::Storage<true,Scalar,Spec,1>::poly, d_dx( exp ) );

    using namespace sturm2_detail;
    Construct< Index<Spec,2>::enabled, Scalar,Spec,2 >::step( sturm );
    Construct< Index<Spec,3>::enabled, Scalar,Spec,3 >::step( sturm );
    Construct< Index<Spec,4>::enabled, Scalar,Spec,4 >::step( sturm );
    Construct< Index<Spec,5>::enabled, Scalar,Spec,5 >::step( sturm );
    Construct< Index<Spec,6>::enabled, Scalar,Spec,6 >::step( sturm );
    Construct< Index<Spec,7>::enabled, Scalar,Spec,7 >::step( sturm );
    Construct< Index<Spec,8>::enabled, Scalar,Spec,8 >::step( sturm );
    Construct< Index<Spec,9>::enabled, Scalar,Spec,9 >::step( sturm );
}





/// construct a sturm sequence
template < typename Scalar, class Spec>
__host__ __device__
int signChanges( SturmSequence2<Scalar,Spec>& sturm, Scalar arg)
{
    int count=0;
    Scalar prevVal = polyval( get<0>(sturm), arg );

    using namespace sturm2_detail;
    CountSignChanges< Index<Spec,2>::enabled, Scalar,Spec,2 >::step( sturm, arg, prevVal, count );
    CountSignChanges< Index<Spec,3>::enabled, Scalar,Spec,3 >::step( sturm, arg, prevVal, count );
    CountSignChanges< Index<Spec,4>::enabled, Scalar,Spec,4 >::step( sturm, arg, prevVal, count );
    CountSignChanges< Index<Spec,5>::enabled, Scalar,Spec,5 >::step( sturm, arg, prevVal, count );
    CountSignChanges< Index<Spec,6>::enabled, Scalar,Spec,6 >::step( sturm, arg, prevVal, count );
    CountSignChanges< Index<Spec,7>::enabled, Scalar,Spec,7 >::step( sturm, arg, prevVal, count );
    CountSignChanges< Index<Spec,8>::enabled, Scalar,Spec,8 >::step( sturm, arg, prevVal, count );
    CountSignChanges< Index<Spec,9>::enabled, Scalar,Spec,9 >::step( sturm, arg, prevVal, count );

    return count;
}







} // namespace polynomial
} // namespace cuda
} // namespace mpblocks
















#endif // STURMSEQUENCE_H_
