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
 *  @file   cuda/bitonic/Direction.h
 *
 *  @date   Jun 18, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_CUDA_BITONIC_DIRECTION_H_
#define MPBLOCKS_CUDA_BITONIC_DIRECTION_H_



namespace mpblocks {
namespace     cuda {
namespace  bitonic {

/// specifies values for the direction the sorter should sort the keys
enum Direction
{
    Descending  = 0,    ///< sort should be descending, i.e. a[i] > [aj], i < j
    Ascending   = 1     ///< sort should be ascending, i.e. a[i] < a[j], i < j
};






} // namespace bitonic
} // namespace cuda
} // namespace mpblocks












#endif // DIRECTION_H_
