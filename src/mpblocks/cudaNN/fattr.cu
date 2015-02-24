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
 *  @file   /home/josh/Codes/cpp/mpblocks2/cuda_nn/src/PointSet.cpp
 *
 *  @date   Oct 6, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#include <mpblocks/cudaNN/fattr.cu.hpp>

namespace mpblocks {
namespace   cudaNN {

template void get_fattr<float,2>( fattrMap_t& map );
template void get_fattr<float,3>( fattrMap_t& map );
template void get_fattr<float,4>( fattrMap_t& map );
template void get_fattr<float,7>( fattrMap_t& map );


} //< namespace cudaNN
} //< namespace mpblocks


