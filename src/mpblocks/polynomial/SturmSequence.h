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
 *  @file   include/mpblocks/polynomial/SturmSequence.h
 *
 *  @date   Jan 15, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_POLYNOMIAL_STURMSEQUENCE_H_
#define MPBLOCKS_POLYNOMIAL_STURMSEQUENCE_H_

#include <vector>
#include <list>

namespace   mpblocks {
namespace polynomial {



template < typename Scalar>
class SturmSequence
{
    public:
        typedef Polynomial<Scalar,Dynamic>                  Poly_t;
        typedef Quotient<Scalar,Poly_t,Poly_t>              Quotient_t;
        typedef std::vector< Polynomial<Scalar,Dynamic> >   PolyList_t;

    private:
        PolyList_t  m_seq;
//        PolyList_t  m_q;

    public:
        SturmSequence()
        {}

        /// builds a sturm sequence from a polynomial
        template < class Exp >
        SturmSequence( const RValue<Scalar,Exp>& rhs )
        {
            rebuild(rhs);
        }

        template <class Exp>
        void rebuild( const RValue<Scalar,Exp>& rhs )
        {
            m_seq.resize( rhs.size() );
//            m_q.resize( rhs.size() );

            ///< f_0 is original
            lvalue(m_seq[0]) = normalized(rhs);

            ///< f_1 is derivative
            Poly_t f1;
            differentiate( rhs, f1, 1 );
            lvalue(m_seq[1]) = normalized(f1);

            // all others are found by the following recursion
            for(int i=2; i < rhs.size(); i++)
            {
                Quotient_t quot = m_seq[i-2] / m_seq[i-1];
//                lvalue(m_q[i])   = quot.q();
                lvalue(m_seq[i]) = -quot.r();
            }
        }

        /// return the number of sign changes at the specified point
        int signChanges( Scalar x )
        {
            int nChanges = 0;
            Scalar prev = polyval(m_seq[0],x);
            for(int i=1; i < m_seq.size(); i++)
            {
                Scalar next = polyval(m_seq[i],x);
                if( sgn(next) )
                {
                    if( sgn(next) != sgn(prev) )
                        nChanges++;
                    prev = next;
                }
            }

            return nChanges;
        }

        const RValue< Scalar, Poly_t >& operator[]( int i ) const
        {
            return m_seq[i];
        }

//        const RValue< Scalar, Poly_t >& operator()( int i ) const
//        {
//            return m_q[i];
//        }

        int size() const
        {
            return m_seq.size();
        }

};


} // namespace polynomial
} // namespace mpblocks
















#endif // STURMSEQUENCE_H_
