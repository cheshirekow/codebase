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
 *  @file   src/polynomial/Polynomial.h
 *
 *  @date   Dec 3, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_CUDA_POLYNOMIAL_POLYNOMIAL_H_
#define MPBLOCKS_CUDA_POLYNOMIAL_POLYNOMIAL_H_

#include <cassert>
#include <iostream>

#include <mpblocks/cuda/polynomial/IntList.h>
#include <mpblocks/cuda/polynomial/RValue.h>
#include <mpblocks/cuda/polynomial/LValue.h>
#include <mpblocks/cuda/polynomial/get_spec.h>


namespace   mpblocks {
namespace       cuda {
namespace polynomial {




/// class actually providing storage for a coefficient
template <typename Scalar, int Idx>
struct Coefficient
{
    Scalar value;
};

/// recursive inheritance tree which provides storage for each of the
/// required coefficients
template <typename Scalar, class IntList>
struct Storage;

template <typename Scalar, int SpecHead, class SpecTail>
struct Storage<Scalar, IntList<SpecHead,SpecTail> >:
    Coefficient<Scalar,SpecHead>,
    Storage<Scalar,SpecTail>
{
    /// variadic initializer, recursively sets all coefficients with a
    /// function call
//    template <typename ParamHead, typename... ParamTail>
//    void init( ParamHead pHead, ParamTail... pTail )
//    {
//        Coefficient<Scalar,SpecHead>::value = pHead;
//        Storage<Scalar,SpecTail>::init( pTail );
//    }

    /// variadic initializer
    __host__ __device__ __forceinline__
    void assign( const Scalar& v0 )
    { Coefficient<Scalar,SpecHead>::value = v0; }

    __host__ __device__ __forceinline__
    void assign( const Scalar& v0, const Scalar& v1 )
    {
        assign(v0);
        Storage<Scalar,SpecTail>::assign(v1);
    }

    __host__ __device__ __forceinline__
    void assign( const Scalar& v0, const Scalar& v1, const Scalar& v2 )
    {
        assign(v0);
        Storage<Scalar,SpecTail>::assign(v1,v2);
    }

    __host__ __device__ __forceinline__
    void assign( const Scalar& v0, const Scalar& v1, const Scalar& v2, const Scalar& v3 )
    {
        assign(v0);
        Storage<Scalar,SpecTail>::assign(v1,v2,v3);
    }

    __host__ __device__ __forceinline__
    void assign( const Scalar& v0, const Scalar& v1, const Scalar& v2, const Scalar& v3, const Scalar& v4 )
    {
        assign(v0);
        Storage<Scalar,SpecTail>::assign(v1,v2,v3,v4);
    }

    __host__ __device__ __forceinline__
    void assign( const Scalar& v0, const Scalar& v1, const Scalar& v2, const Scalar& v3, const Scalar& v4, const Scalar& v5 )
    {
        assign(v0);
        Storage<Scalar,SpecTail>::assign(v1,v2,v3,v4,v5);
    }

    /// assignment by rvalue
    template < class Exp, class Spec2 >
    __host__ __device__ __forceinline__
    void assign( const RValue<Scalar,Exp,Spec2>& exp )
    {
//        if( intlist::contains<Spec2,SpecHead>::value )
//            Coefficient<Scalar,SpecHead>::value =
//                    get<SpecHead>(static_cast<const Exp&>(exp) );
//        else
//            Coefficient<Scalar,SpecHead>::value = Scalar(0);
        Coefficient<Scalar,SpecHead>::value =
                    get<SpecHead>(static_cast<const Exp&>(exp) );
        Storage<Scalar,SpecTail>::assign(exp);
    }

    /// assignment of all coefficients
    template <typename T>
    __host__ __device__ __forceinline__
    void fill( T value )
    {
        Coefficient<Scalar,SpecHead>::value = value;
        Storage<Scalar,SpecTail>::fill( value );
    }

    /// evaluate the polynomial
    template< int idx >
    __host__ __device__ __forceinline__
    Scalar eval( const Scalar& s, Scalar sn = Scalar(1.0) )
    {
        if( idx < SpecHead )
            return eval<idx+1>( s, sn*s );
        else
            return Coefficient<Scalar,SpecHead>::value*sn
                    + Storage<Scalar,SpecTail>::eval<idx+1>(s,sn*s);
    }

    /// runtime indexing of coefficients, i is the array offset (i.e. the i'th
    /// nonzero coefficient, not the coefficient of the i'th power of the
    /// polynomial parameter )
    __host__ __device__ __forceinline__
    Scalar dyn_get( int i )
    {
        return i == 0 ?
                Coefficient<Scalar,SpecHead>::value
                : Storage<Scalar,SpecTail>::dyn_get(i-1);
    }

    /// runtime indexing of coefficients, i is the array offset (i.e. the i'th
    /// nonzero coefficient, not the coefficient of the i'th power of the
    /// polynomial parameter )
    __host__ __device__ __forceinline__
    void dyn_set( int i, const Scalar& value )
    {
        if( i == 0 )
            Coefficient<Scalar,SpecHead>::value = value;
        else
            Storage<Scalar,SpecTail>::dyn_set(i,value);
    }
};

template <typename Scalar, int SpecTail>
struct Storage<Scalar, IntList<SpecTail,intlist::Terminal> >:
    Coefficient<Scalar,SpecTail>
{
//    template <typename ParamTail>
//    void init( ParamTail pTail )
//    {
//        Coefficient<Scalar,SpecTail>::value = pTail;
//    }
    __host__ __device__ __forceinline__
    void assign( const Scalar& v0 )
    { Coefficient<Scalar,SpecTail>::value = v0; }

    /// assignment by rvalue
    template < class Exp, class Spec2 >
    __host__ __device__ __forceinline__
    void assign( const RValue<Scalar,Exp,Spec2>& exp )
    {
//        if( intlist::contains<Spec2,SpecTail>::value )
        Coefficient<Scalar,SpecTail>::value = get<SpecTail>(
                    static_cast<const Exp&>(exp) );
    }


    template <typename T>
    __host__ __device__ __forceinline__
    void fill( T value )
    {
        Coefficient<Scalar,SpecTail>::value = value;
    }

    template< int idx >
    __host__ __device__ __forceinline__
    Scalar eval( const Scalar& s, Scalar sn = Scalar(1.0) )
    {
        if( idx < SpecTail )
            return eval<idx+1>( s, sn*s );
        else
            return Coefficient<Scalar,SpecTail>::value*sn;
    }

    __host__
    Scalar dyn_get( int i )
    {
        assert( i == 0 );
        return Coefficient<Scalar,SpecTail>::value;
    }

    __host__
    void dyn_set( int i, const Scalar& value )
    {
        assert( i == 0 );
        Coefficient<Scalar,SpecTail>::value = value;
    }
};


template <int idx>
struct CoefficientKey{};

namespace coefficient_key
{
    const CoefficientKey<0> _0;
    const CoefficientKey<1> _1;
    const CoefficientKey<2> _2;
    const CoefficientKey<3> _3;
    const CoefficientKey<4> _4;
    const CoefficientKey<5> _5;
    const CoefficientKey<6> _6;
    const CoefficientKey<7> _7;
    const CoefficientKey<8> _8;
    const CoefficientKey<9> _9;
}

namespace device_coefficient_key
{
    __device__ const CoefficientKey<0> _0;
    __device__ const CoefficientKey<1> _1;
    __device__ const CoefficientKey<2> _2;
    __device__ const CoefficientKey<3> _3;
    __device__ const CoefficientKey<4> _4;
    __device__ const CoefficientKey<5> _5;
    __device__ const CoefficientKey<6> _6;
    __device__ const CoefficientKey<7> _7;
    __device__ const CoefficientKey<8> _8;
    __device__ const CoefficientKey<9> _9;
}





/// A sparse, statically sized polynomial
template <typename Scalar, class Spec>
struct Polynomial:
    Storage<Scalar,Spec>,
    RValue< Scalar, Polynomial<Scalar,Spec>, Spec >,
    LValue< Scalar, Polynomial<Scalar,Spec> >
{
    /// Default constructor
    __host__ __device__ __forceinline__
    Polynomial(){}

    /// Construct from any PolynomialExpression:
    template <typename Exp, class Spec2>
    __host__ __device__ __forceinline__
    Polynomial( const RValue<Scalar,Exp,Spec2>& exp )
    {
        Storage<Scalar,Spec>::assign(exp);
//        AssignmentHelper<Scalar, Polynomial<Scalar,Spec>, Exp, Spec2 >
//            ::assign(*this, static_cast<const Exp&>(exp));
    }

    template <typename Exp, class Spec2>
    __host__ __device__ __forceinline__
    void operator=( const RValue<Scalar,Exp,Spec2>& exp )
    {
        Storage<Scalar,Spec>::assign(exp);
//        AssignmentHelper<Scalar, Polynomial<Scalar,Spec>, Exp, Spec >
//            ::assign(*this, static_cast<const Exp&>(exp));
    }

    template <int n, class Exp2, class Spec2>
    __host__ __device__ __forceinline__
    Polynomial( const DerivativeSurrogate<n,Scalar,Exp2,Spec2>& surrogate )
    {
        differentiate<n>( surrogate.exp, *this );
    }

    template <int n, class Exp2, class Spec2>
    __host__ __device__ __forceinline__
    void operator=( const DerivativeSurrogate<n,Scalar,Exp2,Spec2>& surrogate )
    {
        differentiate( surrogate.exp, *this );
    }

    __host__ __device__ __forceinline__
    Polynomial( const Scalar& v0 )
        { Storage<Scalar,Spec>::assign(v0); }

    __host__ __device__ __forceinline__
    Polynomial( const Scalar& v0, const Scalar& v1 )
        { Storage<Scalar,Spec>::assign(v0,v1); }

    __host__ __device__ __forceinline__
    Polynomial( const Scalar& v0, const Scalar& v1, const Scalar& v2 )
        { Storage<Scalar,Spec>::assign(v0,v1,v2); }

    __host__ __device__ __forceinline__
    Polynomial( const Scalar& v0, const Scalar& v1, const Scalar& v2, const Scalar& v3 )
        { Storage<Scalar,Spec>::assign(v0,v1,v2,v3); }

    __host__ __device__ __forceinline__
    Polynomial( const Scalar& v0, const Scalar& v1, const Scalar& v2, const Scalar& v3, const Scalar& v4 )
        { Storage<Scalar,Spec>::assign(v0,v1,v2,v3,v4); }

    __host__ __device__ __forceinline__
    Polynomial( const Scalar& v0, const Scalar& v1, const Scalar& v2, const Scalar& v3, const Scalar& v4, const Scalar& v5 )
        { Storage<Scalar,Spec>::assign(v0,v1,v2,v3,v4,v5); }



    /// evaluate the polynomial at a particular value
    __host__ __device__ __forceinline__
    Scalar eval( Scalar x )
    {
        return Storage<Scalar,Spec>::eval(x,1);
    }
};



template< bool HasCoefficient, int idx, typename Scalar, class Spec >
struct GetHelper
{
    __host__ __device__ __forceinline__
    static Scalar get( const Polynomial<Scalar,Spec>& poly )
    {
        return 0;
    }
};

template< int idx, typename Scalar, class Spec >
struct GetHelper<true,idx,Scalar,Spec>
{
    __host__ __device__ __forceinline__
    static Scalar get( const Polynomial<Scalar,Spec>& poly )
    {
        return poly.Coefficient<Scalar,idx>::value;
    }
};





template <int idx, typename Scalar, class Spec>
__host__ __device__ __forceinline__
Scalar get( const Polynomial<Scalar,Spec>& poly )
{
    return GetHelper< intlist::contains<Spec,idx>::value, idx,Scalar,Spec >
        ::get(poly);
}

template <int idx, typename Scalar, class Spec>
__host__ __device__ __forceinline__
Scalar& set( Polynomial<Scalar,Spec>& poly )
{
    return static_cast< Coefficient<Scalar,idx>& >(poly).value;
}

template <int idx, typename Scalar, class Spec>
__host__ __device__ __forceinline__
Scalar& set_storage( Polynomial<Scalar,Spec>& poly )
{
    return static_cast<
        Coefficient<Scalar, intlist::get<Spec,idx>::value >& >(poly).value;
}




template< class Scalar, class Spec >
struct get_spec< Polynomial<Scalar,Spec> >
{
    typedef Spec result;
};

} // polynomial
} // cuda
} // mpblocks





#endif // POLYNOMIAL_H_
