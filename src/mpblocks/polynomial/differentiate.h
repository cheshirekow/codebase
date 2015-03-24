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
 *  @file   include/mpblocks/polynomial/differentiate.h
 *
 *  @date   Jan 16, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_POLYNOMIAL_DIFFERENTIATE_H_
#define MPBLOCKS_POLYNOMIAL_DIFFERENTIATE_H_


namespace   mpblocks {
namespace polynomial {

/// evaluate a polynomial
template <typename Scalar, class Exp1, class Exp2>
void differentiate( const RValue<Scalar,Exp1>& in,
                    LValue<Scalar,Exp2>& out,
                    int n )
{
    // the number of the derivative to compute
    out.resize( in.size() - n );

    // initialize the output
    for(int i=0; i < out.size(); i++)
        out[i] = in[i+n];

    // fill the output with the initial coefficients
    for(int i=2; i < in.size(); i++)
    {
        // the factor i contributes to all coefficients of the output
        // with index >= (i-n) < (i)
        for(int j = i-n; j < i && j < out.size(); j++)
            out[j] *= i;
    }
}

} // namespace polynomial
} // namespace mpblocks















#endif // DIFFERENTIATE_H_
