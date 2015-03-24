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
 *  @file   src/polynomial/Polynomial.h
 *
 *  @date   Dec 3, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_POLYNOMIAL_POLYNOMIAL_H_
#define MPBLOCKS_POLYNOMIAL_POLYNOMIAL_H_

#include <iostream>
#include <cassert>
#include <vector>
#include <map>

namespace mpblocks     {
namespace polynomial   {

/// A dense, statically sized polynomial
template <typename Scalar, int Degree>
class Polynomial :
    public LValue< Scalar, Polynomial<Scalar,Degree> >,
    public RValue< Scalar, Polynomial<Scalar,Degree> >
{
    public:
        typedef Polynomial<Scalar,Degree >      Polynomial_t;
        typedef LValue< Scalar, Polynomial_t >  LValue_t;

        enum
        {
            DEGREE=Degree,
            FACTORED=0
        };

        struct iterator
        {
            int idx;

            iterator& operator++()
            {
                ++idx;
                return *this;
            }

            bool operator()( const Polynomial_t& poly )
            {
                return idx < poly.size();
            }
        };

    protected:
        Scalar    m_data[Degree+1];

    public:
        typedef unsigned int Size_t;

        Size_t size() const
            { return Degree+1; }

        void resize( Size_t size ){}

        /// vector accessor
        Scalar&   operator[](int i)
        {
            return m_data[i];
        }

        /// vector accessor
        Scalar const&   operator[](int i) const
        {
            return m_data[i];
        }

        Scalar& operator()( const iterator iter )
        {
            return m_data[iter.idx];
        }

        Scalar const& operator()( const iterator iter ) const
        {
            return m_data[iter.idx];
        }

        /// Default constructor
        Polynomial()
        {
            for( int i=0; i < size(); i++)
                (*this)[i] = 0;
        }

        /// Construct from any PolynomialExpression:
        template <typename Exp>
        Polynomial( const RValue<Scalar,Exp>& exp )
        {
            for( int i=0; i < size(); i++)
                (*this)[i] = exp[i];
        }

        /// fixed size construction
        Polynomial( Scalar a0 )
        {
            for(int i=0; i <= Degree; i++)
                m_data[i] = a0;
        }

        Polynomial( Scalar a0, Scalar a1 )
        {
            assert( Degree == 1 );
            m_data[0] = a0;
            m_data[1] = a1;
        }

        Polynomial( Scalar a0, Scalar a1, Scalar a2 )
        {
            assert( Degree == 2 );
            m_data[0] = a0;
            m_data[1] = a1;
            m_data[2] = a2;
        }

        Scalar eval( Scalar x )
        {
            Scalar x_i = 1.0;
            Scalar r   = (*this)[0];

            for(int i=1; i < size(); ++i)
            {
                x_i *= x;
                r   += (*this)[i]*x_i;
            }

            return r;
        }
};


/// A dense, dynamically sized polynomial
template <typename Scalar>
class Polynomial<Scalar,Dynamic> :
    public std::vector<Scalar>,
    public LValue< Scalar, Polynomial<Scalar,Dynamic> >,
    public RValue< Scalar, Polynomial<Scalar,Dynamic> >
{
    public:
        enum
        {
            FACTORED=0,
        };

        typedef Polynomial<Scalar,Dynamic>      Polynomial_t;
        typedef LValue< Scalar, Polynomial_t >  LValue_t;

        struct iterator
        {
            int idx;

            iterator& operator++()
            {
                ++idx;
                return *this;
            }

            bool operator()( const Polynomial_t& poly )
            {
                return idx < poly.size();
            }
        };

    public:
        typedef unsigned int        Size_t;
        typedef std::vector<Scalar> Vector_t;

        /// Default constructor
        Polynomial( int size=0 ):
            Vector_t(size)
        {}

        template <typename Exp>
        Polynomial( const RValue<Scalar,Exp>& exp )
        {
            Vector_t::reserve(exp.size());
            for( int i=0; i < exp.size(); i++)
                Vector_t::push_back(exp[i]);
        }

        /// fixed size construction
        Polynomial( Scalar a0 ):
            Vector_t(1)
        {
            (*this)[0] = a0;
        }

        Polynomial( Scalar a0, Scalar a1 ):
            Vector_t(2)
        {
            (*this)[0] = a0;
            (*this)[1] = a1;
        }

        Polynomial( Scalar a0, Scalar a1, Scalar a2 ):
            Vector_t(3)
        {
            (*this)[0] = a0;
            (*this)[1] = a1;
            (*this)[2] = a2;
        }

        Scalar& operator()( const iterator iter )
        {
            return (*this)[iter.idx];
        }

        Scalar const& operator()( const iterator iter ) const
        {
            return (*this)[iter.idx];
        }

        Scalar eval( Scalar x )
        {
            Scalar x_i = 1.0;
            Scalar r   = (*this)[0];

            for(int i=1; i < size(); ++i)
            {
                x_i *= x;
                r   += (*this)[i]*x_i;
            }

            return r;
        }


        using Vector_t::size;
        using Vector_t::resize;
        using Vector_t::operator[];
};









} // polynomial
} // mpblocks





#endif // POLYNOMIAL_H_
