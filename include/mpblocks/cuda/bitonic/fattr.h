/*
 *  Copyright (C) 2013 Josh Bialkowski (jbialk@mit.edu)
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
 *  @file   fattr.h
 *
 *  @date   Sep 3, 2011
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  class definition for bitonic sort class
 */

#ifndef MPBLOCKS_CUDA_BITONIC_FATTR_H_
#define MPBLOCKS_CUDA_BITONIC_FATTR_H_

#include <map>
#include <string>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <mpblocks/cuda.h>


namespace mpblocks {
namespace     cuda {
namespace  bitonic {


typedef std::map<std::string,FuncAttributes> fattrMap_t;

template <typename KeyType, typename ValueType>
static void get_fattr_kv( fattrMap_t& map );

template <typename KeyType>
static void get_fattr_k( fattrMap_t& map );


} // namespace bitonic
} // namespace cuda
} // namespace mpblocks


#endif// BITONIC_SORTER_H
