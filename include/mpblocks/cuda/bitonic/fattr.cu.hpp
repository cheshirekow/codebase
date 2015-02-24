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
 *  @file   fattr.cu.h
 *
 *  @date   Sep 3, 2011
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief
 *
 *  Note: adapted from the NVIDIA SDK bitonicSort.cu which references
 *  http://www.iti.fh-flensburg.de/lang/algorithmen/sortieren/bitonic/bitonicen.htm
 */



#ifndef MPBLOCKS_BITONIC_FATTR_CU_HPP
#define MPBLOCKS_BITONIC_FATTR_CU_HPP

#include <mpblocks/cuda/bitonic/fattr.h>
#include <mpblocks/cuda/bitonic.cu.hpp>
#include <mpblocks/cuda/bitonic/kernels.cu.hpp>

namespace mpblocks {
namespace     cuda {
namespace  bitonic {

template <typename KeyType, typename ValueType>
void get_fattr_kv( fattrMap_t& map )
{
    map["sortShared"]   .getFrom( &sortShared<KeyType,ValueType>    );
    map["sortSharedInc"].getFrom( &sortSharedInc<KeyType,ValueType> );
    map["mergeGlobal"]  .getFrom( &mergeGlobal<KeyType,ValueType>   );
    map["mergeShared"]  .getFrom( &mergeShared<KeyType,ValueType>   );
    map["prepare"]      .getFrom( &bitonic::prepare<KeyType>        );
}



template <typename KeyType>
void get_fattr_k( fattrMap_t& map )
{
    map["sortShared"]   .getFrom( &sortShared<KeyType>       );
    map["sortSharedInc"].getFrom( &sortSharedInc<KeyType>    );
    map["mergeGlobal"]  .getFrom( &mergeGlobal<KeyType>      );
    map["mergeShared"]  .getFrom( &mergeShared<KeyType>      );
    map["prepare"]      .getFrom( &bitonic::prepare<KeyType> );
}


} // namespace bitonic
} // namespace cuda
} // namespace mpblocks



#endif
