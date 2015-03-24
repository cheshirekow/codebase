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
 *  @file   /home/josh/Codes/cpp/mpblocks2/cuda/include/mpblocks/cuda/polynomial/Construct.h
 *
 *  @date   Oct 24, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_CONSTRUCT_H_
#define MPBLOCKS_CONSTRUCT_H_

namespace   mpblocks {
namespace       cuda {
namespace polynomial {

struct ParamKey{};

namespace param_key
{
    const ParamKey s;
    const ParamKey x;
}

namespace device_param_key
{
    __device__ const ParamKey s;
    __device__ const ParamKey x;
}

template <int pow>
__host__ __device__
CoefficientKey<pow> operator^( ParamKey, CoefficientKey<pow> c )
{
    return c;
}

template <typename Scalar, int pow>
__host__ __device__
Polynomial<Scalar, IntList<pow,intlist::Terminal> >
    operator*( Scalar v, CoefficientKey<pow> c )
{
    return Polynomial<Scalar, IntList<pow,intlist::Terminal> >( v );
}








} // polynomial
} // cuda
} // mpblocks




#endif // CONSTRUCT_H_
