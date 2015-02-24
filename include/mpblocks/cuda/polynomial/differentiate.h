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
 *  @file   include/mpblocks/polynomial/differentiate.h
 *
 *  @date   Jan 16, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_CUDA_POLYNOMIAL_DIFFERENTIATE_H_
#define MPBLOCKS_CUDA_POLYNOMIAL_DIFFERENTIATE_H_


namespace   mpblocks {
namespace       cuda {
namespace polynomial {


template < class Spec, int n >
struct DerivativeSpec{};

template< int Head, class Tail, int n>
struct DerivativeSpec< IntList<Head,Tail>, n >
{
    typedef typename
    intlist::if_else< (Head < n),
        typename DerivativeSpec<Tail,n>::result,
        IntList<Head-n, typename DerivativeSpec<Tail,n>::result>
        >::result
        result;
};

template< int Head, int n>
struct DerivativeSpec< IntList<Head,intlist::Terminal>, n >
{
    typedef typename
    intlist::if_else< (Head < n),
        IntList<0,intlist::Terminal>,
        IntList<Head-n,intlist::Terminal>
        >::result
        result;
};


namespace derivative_detail
{
    template< int n, class Exp1, class Exp2, class Spec>
    struct InitHelper{};

    template< int n, class Exp1, class Exp2, int Head, class Tail >
    struct InitHelper< n, Exp1, Exp2, IntList<Head,Tail> >
    {
        __host__ __device__
        static void copy( const Exp1& in, Exp2& out )
        {
            set<Head>(out) = get<Head+n>(in);
            InitHelper<n,Exp1,Exp2,Tail>::copy(in,out);
        }
    };

    template< int n, class Exp1, class Exp2, int Head >
    struct InitHelper< n, Exp1, Exp2, IntList<Head,intlist::Terminal> >
    {
        __host__ __device__
        static void copy( const Exp1& in, Exp2& out )
        {
            set<Head>(out) = get<Head+n>(in);
        }
    };

    template< bool B, class Exp, int j, int i >
    struct SetHelper
    {
        __host__ __device__
        static void do_it( Exp& out ){}
    };

    template< class Exp, int j, int i >
    struct SetHelper<true,Exp,j,i>
    {
        __host__ __device__
        static void do_it( Exp& out )
        {
            set<j>(out) *= i;
        }
    };


    template< class Exp, class Spec, int j, int i, int size2>
    struct Looper2
    {
        __host__ __device__
        static void go( Exp& out )
        {
            SetHelper< intlist::contains<Spec,j>::value, Exp,j,i >::do_it(out);
            Looper2<Exp,Spec,j+1,i,size2>::go(out);
        }
    };

    /// loop termination, j == i
    template< class Exp, class Spec, int i, int size2>
    struct Looper2<Exp, Spec, i, i, size2 >
    {
        __host__ __device__
        static void go( Exp& out ){}
    };

    /// loop termination: j = out.size()
    template< class Exp, class Spec, int i, int size2>
    struct Looper2<Exp, Spec, size2, i, size2 >
    {
        __host__ __device__
        static void go( Exp& out ){}
    };

    /// loop termination: j = out.size()
    template< class Exp, class Spec, int size2>
    struct Looper2<Exp, Spec, size2, size2, size2 >
    {
        __host__ __device__
        static void go( Exp& out ){}
    };

    template< int n, int size1, class Exp2, class Spec2, int i>
    struct Looper1
    {
        __host__ __device__
        static void go( Exp2& out )
        {
            Looper2< Exp2,Spec2,i-n,i, intlist::max<Spec2>::value+1 >::go(out);
            Looper1<n,size1,Exp2,Spec2,i+1>::go(out);
        }
    };

    template< int n, int size1, class Exp2, class Spec2>
    struct Looper1<n,size1,Exp2,Spec2,size1>
    {
        __host__ __device__
        static void go( Exp2& out ){}
    };

}


/// evaluate a polynomial
template <int n, typename Scalar, class Exp1, class InSpec, class Exp2>
__host__ __device__
void differentiate( const RValue<Scalar,Exp1,InSpec>& in,
                    LValue<Scalar,Exp2>& out )
{
    typedef typename DerivativeSpec<InSpec,n>::result OutSpec;

    enum{ max_in  = intlist::max<InSpec>::value  };
    enum{ max_out = intlist::max<OutSpec>::value };

    // initialize the output
    derivative_detail::InitHelper<n,Exp1,Exp2,OutSpec>::copy(
            static_cast<const Exp1&>(in),
            static_cast<Exp2&>(out) );
//    for(int i=0; i < out.size(); i++)
//        out[i] = in[i+n];


    // fill the output with the initial coefficients
    derivative_detail::Looper1<n,max_in+1,Exp2,OutSpec,2>::go(
            static_cast<Exp2&>(out) );
//    for(int i=2; i < in.size(); i++)
//    {
//        // the factor i contributes to all coefficients of the output
//        // with index >= (i-n) < (i)
//        for(int j = i-n; j < i && j < out.size(); j++)
//            out[j] *= i;
//    }
}

/// return a surrogate which notifies LValue to call differentiate
template <int n, typename Scalar, class Exp, class Spec>
__host__ __device__
DerivativeSurrogate<n,Scalar,Exp,Spec> d_ds( const RValue<Scalar,Exp,Spec>& exp )
{
    return DerivativeSurrogate<n,Scalar,Exp,Spec>(
            static_cast< const Exp& >(exp) );
};



} // namespace polynomial
} // namespace cuda
} // namespace mpblocks















#endif // DIFFERENTIATE_H_
