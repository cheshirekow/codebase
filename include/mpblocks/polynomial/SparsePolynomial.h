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
 *  @file   src/polynomial/SparsePolynomial.h
 *
 *  @date   Dec 3, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_POLYNOMIAL_SPARSEPOLYNOMIAL_H_
#define MPBLOCKS_POLYNOMIAL_SPARSEPOLYNOMIAL_H_

#include <iostream>
#include <cassert>
#include <map>

namespace mpblocks {
namespace polynomial   {


template <typename Scalar>
class SparsePolynomial
{
    public:
        typedef SparsePolynomial<Scalar>            Polynomial_t;
        typedef LValue< Scalar, Polynomial_t >      LValue_t;

        typedef std::map<int,Scalar>    Map_t;
        typedef std::pair<int,Scalar>   Pair_t;

    protected:
        Map_t m_data;

    public:
        int size() const
        {
            return m_data.back()->first + 1;
        }

        /// access for assignment
        Scalar&   getMutable(int i)
        {
            typename Map_t::iterator iPair = m_data.find(i);
            if( iPair == m_data.end() )
                return m_data.insert( Pair_t(i,0) )->first->second;
            else
                return iPair->second;
        }

        /// vector accessor
        Scalar getImmutable(int i) const
        {
            typename Map_t::iterator iPair = m_data.find(i);
            if( iPair == m_data.end() )
                return 0;
            else
                return iPair->second;
        }

        /// Default constructor
        SparsePolynomial(){}
};








} // polynomial
} // mpblocks





#endif // POLYNOMIAL_H_
