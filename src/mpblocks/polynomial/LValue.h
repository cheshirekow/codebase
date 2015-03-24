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
 *  @file   pblocks/polynomial/LValue.h
 *
 *  @date   Dec 9, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_POLYNOMIAL_LVALUE_H_
#define MPBLOCKS_POLYNOMIAL_LVALUE_H_

namespace mpblocks {
namespace polynomial   {

/// expression template for rvalues
template <typename Scalar, class Exp>
class LValue
{
    public:
        typedef StreamAssignment< LValue<Scalar,Exp> >        Stream_t;

        typedef unsigned int Size_t;

        /// return the size for a vector
        Size_t size() const
        {
            return static_cast<Exp const&>(*this).size();
        }

        Size_t degree() const
        {
            return size()-1;
        }

        /// return the evaluated i'th element of a vector expression
        Scalar& operator[]( Size_t i )
        {
            return static_cast<Exp&>(*this)[i];
        }

        /// returns a stream for assignment
        Stream_t operator<<( Scalar x )
        {
            Stream_t stream(*this);
            return   stream << x;
        }

        template <class Exp2>
        LValue<Scalar,Exp>& operator=( RValue<Scalar,Exp2> const& B )
        {
            resize(B.size());
            for( int i=0; i < B.size(); i++)
                    (*this)[i] = B[i];
            return *this;
        }

        void resize( Size_t size )
        {
            static_cast<Exp*>(this)->resize( size );
        }


        operator RValue<Scalar,Exp>()
        {
            return
                static_cast< RValue<Scalar,Exp>& >(
                    static_cast< Exp& >(*this) );
        }

        void fill( Scalar val )
        {
            for(int i=0; i < size(); i++)
                (*this)[i] = val;
        }
};


template <typename Scalar, class Exp>
LValue<Scalar,Exp>& lvalue( LValue<Scalar,Exp>& exp )
{
    return exp;
}






} // polynomial
} // mpblocks





#endif // MATRIXEXPRESSION_H_
