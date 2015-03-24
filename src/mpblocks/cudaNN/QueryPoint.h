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
 *  @file   include/mpblocks/dubins/curves_cuda/kernels.h
 *
 *  @date   Jun 13, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_CUDANN_QUERYPOINT_CU_H_
#define MPBLOCKS_CUDANN_QUERYPOINT_CU_H_

namespace  mpblocks {
namespace    cudaNN {

template< typename Format_t, unsigned int NDim>
struct QueryPoint
{
    Format_t data[NDim];
};

template< typename Format_t, unsigned int NDim>
struct RectangleQuery
{
    Format_t min[NDim];
    Format_t max[NDim];
    Format_t point[NDim];
};


} // cudaNN
} // mpblocks




#endif // KERNELS_H_
