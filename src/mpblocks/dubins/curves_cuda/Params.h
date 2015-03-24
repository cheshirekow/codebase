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
 *  @file   dubins/curves_cuda/dispatch.h
 *
 *  @date   Jun 13, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_DUBINS_CURVES_CUDA2_DISPATCH_H_
#define MPBLOCKS_DUBINS_CURVES_CUDA2_DISPATCH_H_


#include <cuda.h>
#include <cuda_runtime.h>
#include <map>
#include <string>

namespace      mpblocks {
namespace        dubins {
namespace  curves_cuda {

template <typename Format_t>
struct Params
{
    Format_t  r;    ///< turning radius
    Format_t  q[3]; ///< the query state

    void set_q( Format_t q_in[3] )
    {
        for(int i=0; i < 3; i++)
            q[i] = q_in[i];
    }
};

template <typename Format_t>
struct EuclideanParams
{
    Format_t  q[3]; ///< the query state

    void set_q( Format_t q_in[3] )
    {
        for(int i=0; i < 3; i++)
            q[i] = q_in[i];
    }
};




} // curves
} // dubins
} // mpblocks

#endif // DISPATCH_H_
