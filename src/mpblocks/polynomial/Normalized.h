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
 *  @file   include/mpblocks/polynomial/Normalized.h
 *
 *  @date   Dec 9, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_POLYNOMIAL_NORMALIZED_H_
#define MPBLOCKS_POLYNOMIAL_NORMALIZED_H_

namespace mpblocks {
namespace polynomial   {

/// expression template for sum of two matrix expressions
template <typename Scalar, class Exp>
class Normalized :
    public RValue< Scalar,Normalized<Scalar,Exp> >
{
    Exp const& m_exp;

    public:
        typedef std::size_t Size_t;

        Normalized( Exp const& exp ):
            m_exp(exp)
        {
        }

        /// return the size for a vector
        Size_t size() const
        {
            return  m_exp.size();
        }

        /// return the evaluated i'th element of a vector expression
        Scalar operator[]( Size_t i ) const
        {
            return m_exp[i] / m_exp[ m_exp.size()-1 ];
        }

        Scalar eval( Scalar x )
        {
            return m_exp.eval(x);
        }
};




template <typename Scalar, class Exp>
Normalized<Scalar,Exp> normalized(
        RValue<Scalar,Exp> const& exp )
{
    typedef Normalized<Scalar,Exp> Normalized_t;
    return Normalized_t( static_cast<Exp const&>(exp) );
}




} // polynomial
} // mpblocks





#endif // NORMALIZED_H_
