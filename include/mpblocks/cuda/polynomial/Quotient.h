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
 *  @file   include/mpblocks/polynomial/Quotient.h
 *
 *  @date   Dec 9, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_CUDA_POLYNOMIAL_QUOTIENT_H_
#define MPBLOCKS_CUDA_POLYNOMIAL_QUOTIENT_H_

namespace   mpblocks {
namespace       cuda {
namespace polynomial {



/// surrogate for initialzing with  / operator
template <typename Scalar,
            class ExpNum, class SpecNum,
            class ExpDen, class SpecDen >
struct QuotientSurrogate
{
    const ExpNum& Num;
    const ExpDen& Den;

    __host__ __device__
    QuotientSurrogate( const ExpNum& Num,
                       const ExpDen& Den )
    :
        Num(Num),
        Den(Den)
    {}
};



/// expression template for polynomial long division
template <typename Scalar,
            class ExpNum, class SpecNum,
            class ExpDen, class SpecDen >
struct Quotient
{
    enum
    {
        max_coeff_numerator   = intlist::max< SpecNum >::value,
        max_coeff_denominator = intlist::max< SpecDen >::value,
        max_coeff_quotient    = max_coeff_numerator - max_coeff_denominator,
        max_coeff_remainder   = max_coeff_numerator,
        size_numerator        = max_coeff_numerator   + 1,
        size_denominator      = max_coeff_denominator + 1
    };

    typedef typename intlist::range<0,max_coeff_quotient >::result SpecQuot;
    typedef typename intlist::range<0,max_coeff_remainder>::result SpecRem;

    Polynomial<Scalar,SpecQuot> q;
    Polynomial<Scalar,SpecRem>  r;

    template< int size, int min >
    struct QuotientWork
    {
         __host__ __device__
        static void work( Polynomial<Scalar,SpecQuot>& q,
                          Polynomial<Scalar,SpecRem>&  r,
                          ExpNum const& N, ExpDen const& D )
        {
            // degree of the new term
            enum{ n = size - size_denominator };

            // coefficient of the new term
            Scalar t = get<size-1>(r) /
                       get<max_coeff_denominator>(D);

            // append the new term to the quotient
            set<n>(q) = t;

            // update the remainder
            r = N - q*D;
        }
    };

    template< int size, int min >
    struct QuotientStep
    {
        __host__ __device__
        static void step( Polynomial<Scalar,SpecQuot>& q,
                          Polynomial<Scalar,SpecRem>&  r,
                          ExpNum const& N, ExpDen const& D )
        {
            QuotientWork<size,min>::work( q,r,N,D );

            // next step
            QuotientStep<size-1,min>::step( q,r,N,D );
        }
    };

    template< int min >
    struct QuotientStep<min,min>
    {
        __host__ __device__
        static void step( Polynomial<Scalar,SpecQuot>& q,
                          Polynomial<Scalar,SpecRem>&  r,
                          ExpNum const& N, ExpDen const& D )
        {}
    };

    __host__ __device__
    void construct( ExpNum const& N, ExpDen const& D )
    {
        q.fill(0);
        r = N;
        QuotientStep<size_numerator,size_denominator-1>::step(q,r,N,D);
    }

    __host__ __device__
    Quotient(){}

    __host__ __device__
    Quotient( ExpNum const& N, ExpDen const& D )
    {
        construct(N,D);
    }

    __host__ __device__
    Quotient( const QuotientSurrogate<Scalar,ExpNum,SpecNum
                                            ,ExpDen,SpecDen>& surrogate )
    {
        construct(surrogate.Num,surrogate.Den);
    }

    __host__ __device__
    void operator=( const QuotientSurrogate<Scalar,ExpNum,SpecNum
                                            ,ExpDen,SpecDen>& surrogate )
    {
        construct(surrogate.Num,surrogate.Den);
    }
};




template <typename Scalar,
            class Exp1, class Spec1,
            class Exp2, class Spec2 >
__host__ __device__
QuotientSurrogate<Scalar,Exp1,Spec1,Exp2,Spec2> operator/(
        RValue<Scalar,Exp1,Spec1> const& A,
        RValue<Scalar,Exp2,Spec2> const& B )
{
    return QuotientSurrogate<Scalar,Exp1,Spec1,Exp2,Spec2>(
            static_cast<Exp1 const&>(A),
            static_cast<Exp2 const&>(B));
}




} // namespace polynomial
} // namespace cuda
} // nampespace mpblocks





#endif // QUOTIENT_H_
