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
 *  \file   mpblocks/clarkson93/SimplexBase.h
 *
 *  \date   Aug 17, 2012
 *  \author Josh Bialkowski (jbialk@mit.edu)
 *  \brief  
 */

#ifndef MPBLOCKS_CLARKSON93_DYNAMIC_SIMPLEX_HPP_
#define MPBLOCKS_CLARKSON93_DYNAMIC_SIMPLEX_HPP_

#include <mpblocks/clarkson93.hpp>

namespace   mpblocks {
namespace clarkson93 {
namespace    dynamic {


template <class Traits>
void SimplexBase<Traits>::calculateConstraint( uint ndim, PointDeref& deref )
{
    typedef Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> Matrix;
    typedef Eigen::Matrix<Scalar,Eigen::Dynamic,1>              Vector;

    Matrix A(ndim,ndim);
    Vector b(ndim);

    for(int i=1; i < ndim+1; i++)
        A.row(i-1) = deref(vertices[i]);
    b.setConstant(1);

    // solve for the normal
    n = A.partialPivLu().solve(b);
    n.normalize();

    // and then find the value of 'c' (hyperplane offset)
    c = deref(vertices[1]).dot(n);
}

template <class Traits>
void SimplexBase<Traits>::
    orientConstraint( const Point& x, Orientation orient )
{
    // then orient the hyperplane so that the peak vertex is on the
    // "positive" side (i.e. the 'less than' side)

    switch( orient )
    {
        case INSIDE:
        {
            if( n.dot( x ) > c )
            {
                n = -n;
                c = -c;
            }
            break;
        }

        case OUTSIDE:
        {
            if( n.dot( x ) < c )
            {
                n = -n;
                c = -c;
            }
            break;
        }

        default:
            assert(false);
            break;
    }
}


template <class Traits>
typename Traits::Scalar SimplexBase<Traits>::normalProjection( const Point& x )
{
    return c - n.dot(x);
}


template <class Traits>
bool SimplexBase<Traits>::isInfinite( PointRef antiOrigin )
{
    return vertices[0] == antiOrigin;
}

template <class Traits>
bool SimplexBase<Traits>::isVisible( const Point& x )
{
    return ( n.dot(x) < c );
}


} // namespace dynamic
} // namespace clarkson93
} // namespace mpblocks




#endif // SIMPLEX_HPP_
