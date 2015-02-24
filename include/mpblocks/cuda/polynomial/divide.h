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
 *  @file   /home/josh/Codes/cpp/mpblocks2/cuda/include/mpblocks/cuda/polynomial/divide.h
 *
 *  @date   Oct 26, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_CUDA_POLYNOMIAL_DIVIDE_H_
#define MPBLOCKS_CUDA_POLYNOMIAL_DIVIDE_H_

namespace   mpblocks {
namespace       cuda {
namespace polynomial {

template< class NumSpec, class DenSpec >
struct QuotientSpec
{
    enum
    {
        max_coeff_N = intlist::max<NumSpec>::value,
        max_coeff_D = intlist::max<DenSpec>::value,
        max_coeff_Q = max_coeff_N - max_coeff_D
    };

    typedef typename intlist::range<0,max_coeff_Q>::result result;
};

template < class NumSpec, class DenSpec >
struct RemainderSpec
{
    enum
    {
        max_coeff_D = intlist::max<DenSpec>::value,
        max_coeff_R = max_coeff_D - 1
    };

    typedef typename intlist::range<0,max_coeff_R>::result result;
};

template < class NumSpec, class DenSpec >
struct ScratchSpec
{
    enum
    {
        max_coeff_N = intlist::max<NumSpec>::value,
        max_coeff_S = max_coeff_N - 1
    };

    typedef typename intlist::range<0,max_coeff_S>::result result;
};



/// specialization for the first step uses actual numerator, not remainder
template< bool Enabled, int i, typename Scalar,
            class NumExp, class NumSpec,
            class DenExp, class DenSpec >
struct Divide
{
    __host__ __device__  __forceinline__
    static void step( const NumExp& n, const DenExp& d,
     Polynomial<Scalar, typename QuotientSpec<NumSpec,DenSpec>::result >& q,
     Polynomial<Scalar, typename ScratchSpec<NumSpec,DenSpec>::result >& r )
    {}
};

template< int i, typename Scalar,
            class NumExp, class NumSpec,
            class DenExp, class DenSpec >
struct Divide< true, i, Scalar, NumExp, NumSpec, DenExp, DenSpec>
{
    enum
    {
        max_coeff_N = intlist::max<NumSpec>::value,
        max_coeff_D = intlist::max<DenSpec>::value,
        max_coeff_Q = max_coeff_N - max_coeff_D,
        max_coeff_R = max_coeff_D - 1,
        coeff_N = max_coeff_N - i,
        coeff_Q = max_coeff_Q - i,
    };

    __host__ __device__  __forceinline__
    static void step( const NumExp& n, const DenExp& d,
     Polynomial<Scalar, typename QuotientSpec<NumSpec,DenSpec>::result >& q,
     Polynomial<Scalar, typename ScratchSpec<NumSpec,DenSpec>::result >& r )
    {
        set<coeff_Q>(q) = get<coeff_N>(r) / get<max_coeff_D>(d);
        r =  n - q*d;
    }
};

/// specialization for the first step uses actual numerator, not remainder
template< typename Scalar,
            class NumExp, class NumSpec,
            class DenExp, class DenSpec >
struct Divide< true, 0, Scalar, NumExp,NumSpec,DenExp,DenSpec >
{
    enum
    {
        i = 0,
        max_coeff_N = intlist::max<NumSpec>::value,
        max_coeff_D = intlist::max<DenSpec>::value,
        max_coeff_Q = max_coeff_N - max_coeff_D,
        max_coeff_R = max_coeff_D - 1,
        coeff_N = max_coeff_N - i,
        coeff_Q = max_coeff_Q - i,
    };

    __host__ __device__  __forceinline__
    static void step( const NumExp& n, const DenExp& d,
     Polynomial<Scalar, typename QuotientSpec<NumSpec,DenSpec>::result >& q,
     Polynomial<Scalar, typename ScratchSpec<NumSpec,DenSpec>::result >& r )
    {
        set<coeff_Q>(q) = get<coeff_N>(n) / get<max_coeff_D>(d);
        r =  n - q*d;
    }
};




template< typename Scalar,
                class NumExp, class NumSpec,
                class DenExp, class DenSpec >
__host__ __device__  __forceinline__
void divide( const NumExp& n, const DenExp& d,
     Polynomial<Scalar, typename QuotientSpec<NumSpec,DenSpec>::result >& q,
     Polynomial<Scalar, typename RemainderSpec<NumSpec,DenSpec>::result >& r)
{
    enum
    {
        max_coeff_N = intlist::max<NumSpec>::value,
        max_coeff_D = intlist::max<DenSpec>::value,
        max_coeff_Q = max_coeff_N - max_coeff_D,
        max_coeff_R = max_coeff_D - 1,
        end_step    = max_coeff_Q + 1
    };

    Polynomial<Scalar,typename ScratchSpec<NumSpec,DenSpec>::result > s;
    q.fill(0);
    Divide<(0 < end_step),0,Scalar,NumExp,NumSpec,DenExp,DenSpec>::step(n,d,q,s);
    Divide<(1 < end_step),1,Scalar,NumExp,NumSpec,DenExp,DenSpec>::step(n,d,q,s);
    Divide<(2 < end_step),2,Scalar,NumExp,NumSpec,DenExp,DenSpec>::step(n,d,q,s);
    Divide<(3 < end_step),3,Scalar,NumExp,NumSpec,DenExp,DenSpec>::step(n,d,q,s);
    Divide<(4 < end_step),4,Scalar,NumExp,NumSpec,DenExp,DenSpec>::step(n,d,q,s);
    Divide<(5 < end_step),5,Scalar,NumExp,NumSpec,DenExp,DenSpec>::step(n,d,q,s);
    Divide<(6 < end_step),6,Scalar,NumExp,NumSpec,DenExp,DenSpec>::step(n,d,q,s);
    Divide<(7 < end_step),7,Scalar,NumExp,NumSpec,DenExp,DenSpec>::step(n,d,q,s);
    Divide<(8 < end_step),8,Scalar,NumExp,NumSpec,DenExp,DenSpec>::step(n,d,q,s);
    Divide<(9 < end_step),9,Scalar,NumExp,NumSpec,DenExp,DenSpec>::step(n,d,q,s);
    r = s;
}

template< typename Scalar, class NumSpec, class DenSpec >
__host__ __device__  __forceinline__
void mod(
    const Polynomial<Scalar,NumSpec>& n,
    const Polynomial<Scalar,DenSpec>& d,
     Polynomial<Scalar, typename RemainderSpec<NumSpec,DenSpec>::result >& r)
{
    enum
    {
        max_coeff_N = intlist::max<NumSpec>::value,
        max_coeff_D = intlist::max<DenSpec>::value,
        max_coeff_Q = max_coeff_N - max_coeff_D,
        max_coeff_R = max_coeff_D - 1,
        end_step    = max_coeff_Q + 1
    };

    typedef Polynomial<Scalar,NumSpec> NumExp;
    typedef Polynomial<Scalar,DenSpec> DenExp;

    typedef typename QuotientSpec<NumSpec,DenSpec>::result SpecQ;
    typedef typename ScratchSpec<NumSpec,DenSpec>::result  SpecS;

    Polynomial<Scalar,SpecQ> q;
    Polynomial<Scalar,SpecS>  s;

    q.fill(0);
    Divide<(0 < end_step),0,Scalar,NumExp,NumSpec,DenExp,DenSpec>::step(n,d,q,s);
    Divide<(1 < end_step),1,Scalar,NumExp,NumSpec,DenExp,DenSpec>::step(n,d,q,s);
    Divide<(2 < end_step),2,Scalar,NumExp,NumSpec,DenExp,DenSpec>::step(n,d,q,s);
    Divide<(3 < end_step),3,Scalar,NumExp,NumSpec,DenExp,DenSpec>::step(n,d,q,s);
    Divide<(4 < end_step),4,Scalar,NumExp,NumSpec,DenExp,DenSpec>::step(n,d,q,s);
    Divide<(5 < end_step),5,Scalar,NumExp,NumSpec,DenExp,DenSpec>::step(n,d,q,s);
    Divide<(6 < end_step),6,Scalar,NumExp,NumSpec,DenExp,DenSpec>::step(n,d,q,s);
    Divide<(7 < end_step),7,Scalar,NumExp,NumSpec,DenExp,DenSpec>::step(n,d,q,s);
    Divide<(8 < end_step),8,Scalar,NumExp,NumSpec,DenExp,DenSpec>::step(n,d,q,s);
    Divide<(9 < end_step),9,Scalar,NumExp,NumSpec,DenExp,DenSpec>::step(n,d,q,s);
    r = s;
}





} // namespace polynomial
} // namespace cuda
} // nampespace mpblocks














#endif // DIVIDE_H_
