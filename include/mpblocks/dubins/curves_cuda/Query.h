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
 *  \file   mpblocks/dubins/curves/Query.h
 *
 *  \date   Oct 30, 2012
 *  \author Josh Bialkowski (jbialk@mit.edu)
 *  \brief  
 */

#ifndef MPBLOCKS_DUBINS_CURVES_CUDA2_QUERY_H_
#define MPBLOCKS_DUBINS_CURVES_CUDA2_QUERY_H_

#include <mpblocks/cuda/linalg2.h>


namespace      mpblocks {
namespace        dubins {
namespace  curves_cuda {

/// encapsulates two states and a raidus
template <typename Format_t>
struct Query
{
    typedef cuda::linalg2::Matrix<Format_t,3,1>   Vector3d_t;
    typedef cuda::linalg2::Matrix<Format_t,2,1>   Vector2d_t;

    Vector3d_t  q0; ///< start state
    Vector3d_t  q1; ///< end state
    Format_t    r;  ///< min radius
};



} // curves
} // dubins
} // mpblocks




#endif // QUERY_H_
