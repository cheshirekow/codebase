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
 *  @file   include/mpblocks/polynomial/StreamAssignment.h
 *
 *  @date   Dec 3, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_POLYNOMIAL_STREAMASSIGNMENT_H_
#define MPBLOCKS_POLYMOMIAL_STREAMASSIGNMENT_H_

#include <iostream>
#include <cassert>

namespace   mpblocks {
namespace polynomial {



/// assignment
template <class Exp>
class StreamAssignment
{
    public:
        typedef StreamAssignment<Exp>   Stream_t;

    private:
        Exp&            m_mat;
        unsigned int    m_i;

    public:
         
        StreamAssignment( Exp& M ):
            m_mat(M),
            m_i(0)
        {}

        template <typename Scalar>
        Stream_t& append( Scalar x )
        {
            if(m_mat.size() <= m_i )
                m_mat.resize(m_i+1);
            m_mat[m_i++] = x;
            return *this;
        }

        template <typename Scalar>
        Stream_t& operator<<( Scalar x )
        {
            return append<Scalar>(x);
        }

        template <typename Scalar>
        Stream_t& operator,( Scalar x )
        {
            return append<Scalar>(x);
        }
};








} // polynomial
} // mpblocks





#endif // POLYMOMIAL_H_
