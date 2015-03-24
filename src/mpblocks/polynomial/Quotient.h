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

#ifndef MPBLOCKS_POLYNOMIAL_QUOTIENT_H_
#define MPBLOCKS_POLYNOMIAL_QUOTIENT_H_

namespace mpblocks {
namespace polynomial   {

/// expression template for sum of two matrix expressions
template <typename Scalar, class Exp1, class Exp2>
class Quotient
{
    public:
        typedef Polynomial<Scalar,Dynamic>  Quotient_t;
        typedef Polynomial<Scalar,Dynamic>  Remainder_t;

    private:
        Quotient_t  m_q;
        Remainder_t m_r;

    public:
        typedef unsigned int Size_t;

        Quotient( Exp1 const& numerator, Exp2 const& denominator ):
            m_r(numerator)
        {
            m_q.resize(numerator.size()-denominator.size()+1);
            m_q.fill(0);

            for( int size = numerator.size();
                    size > 0 && size >= denominator.size(); )
            {
                // degree of the new term
                int n = size - denominator.size();

                // coefficient of the new term
                Scalar t = m_r[m_r.size()-1] /
                            denominator[denominator.size()-1];

                // append the new term to the quotient
                m_q[n] = t;

                // update the remainder
                lvalue(m_r) = numerator - m_q*denominator;

                // remove the term from the remainder
                m_r.resize( --size );
            }
        }

        RValue<Scalar,Polynomial<Scalar,Dynamic> >& q()
        {
            return m_q;
        }

        RValue<Scalar,Polynomial<Scalar,Dynamic> >& r()
        {
            return m_r;
        }
};




template <typename Scalar, class Exp1, class Exp2>
Quotient<Scalar,Exp1,Exp2> operator/(
        RValue<Scalar,Exp1> const& A,
        RValue<Scalar,Exp2> const& B )
{
    typedef Quotient<Scalar,Exp1,Exp2> Quotient_t;
    return Quotient_t(
            static_cast<Exp1 const&>(A),
            static_cast<Exp2 const&>(B));
}




} // polynomial
} // mpblocks





#endif // QUOTIENT_H_
