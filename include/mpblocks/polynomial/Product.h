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
 *  @file   include/mpblocks/polynomial/Product.h
 *
 *  @date   Dec 9, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_POLYNOMIAL_PRODUCT_H_
#define MPBLOCKS_POLYNOMIAL_PRODUCT_H_

namespace mpblocks {
namespace polynomial   {

/// expression template for sum of two matrix expressions
template <typename Scalar, class Exp1, class Exp2>
class Product :
    public RValue<
            Scalar,
            Product<Scalar,Exp1,Exp2> >
{
    Exp1 const& m_A;
    Exp2 const& m_B;

    public:
        typedef unsigned int Size_t;

        Product( Exp1 const& A, Exp2 const& B ):
            m_A(A),
            m_B(B)
        {
        }

        /// return the size for a vector
        Size_t size() const
        {
            return m_A.size() + m_B.size() - 1;
        }

        /// return the evaluated i'th element of a vector expression
        Scalar operator[]( Size_t n ) const
        {
            Scalar c = 0;
            for(int i=0; i <= n; i++)
            {
                int j = n-i;
                if( i < m_A.size() && j < m_B.size() )
                    c += m_A[i] * m_B[j];
            }

            return c;
        }

        Scalar eval( Scalar x )
        {
            return m_A.eval(x) * m_B.eval(x);
        }
};




template <typename Scalar, class Exp1, class Exp2>
Product<Scalar,Exp1,Exp2> operator*(
        RValue<Scalar,Exp1> const& A,
        RValue<Scalar,Exp2> const& B )
{
    typedef Product<Scalar,Exp1,Exp2> Product_t;
    return Product_t(
            static_cast<Exp1 const&>(A),
            static_cast<Exp2 const&>(B));
}




} // polynomial
} // mpblocks





#endif // PRODUCT_H_
