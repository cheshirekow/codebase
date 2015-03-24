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
 *  @file   include/mpblocks/polynomial/RValue.h
 *
 *  @date   Dec 9, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_POLYNOMIAL_RVALUE_H_
#define MPBLOCKS_POLYNOMIAL_RVALUE_H_

namespace mpblocks {
namespace polynomial   {

/// expression template for rvalues
template <typename Scalar, class Exp>
class RValue
{
    public:
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
        Scalar operator[]( Size_t i ) const
        {
            if( i < size() )
                return static_cast<Exp const&>(*this)[i];
            else
                return 0;
        }

        /// return a reference to the derived type
        operator Exp&()
        {
            return static_cast<Exp&>(*this);
        }

        /// return a const reference to the derived type
        operator Exp const&()
        {
            return static_cast<Exp const&>(*this);
        }


};




} // polynomial
} // mpblocks





#endif // MATRIXEXPRESSION_H_
