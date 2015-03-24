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
 *  @file   include/mpblocks/polynomial/polyval.h
 *
 *  @date   Jan 15, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_POLYNOMIAL_POLYVAL_H_
#define MPBLOCKS_POLYNOMIAL_POLYVAL_H_

namespace   mpblocks {
namespace polynomial {

/// evaluate a polynomial
template <typename Scalar, class Exp>
Scalar polyval( const RValue<Scalar,Exp>& poly, Scalar x )
{
    Scalar x_i = 1.0;
    Scalar r   = poly[0];

    for(int i=1; i < poly.size(); ++i)
    {
        x_i *= x;
        r   += poly[i]*x_i;
    }

    return r;
}

} // namespace polynomial
} // namespace mpblocks














#endif // POLYVAL_H_
