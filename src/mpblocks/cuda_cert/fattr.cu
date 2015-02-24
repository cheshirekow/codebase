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
 *  @file   /home/josh/Codes/cpp/mpblocks2/cuda_nn/include/mpblocks/cudaNN/fattr.cu.hpp
 *
 *  @date   Oct 7, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_CUDA_CERT_FATTR_CU_HPP_
#define MPBLOCKS_CUDA_CERT_FATTR_CU_HPP_

#include <mpblocks/cuda.hpp>
#include <mpblocks/cuda_cert/fattr.h>
#include <mpblocks/cuda_cert/kernels.cu.h>

namespace  mpblocks {
namespace cuda_cert {

void get_fattr( fattrMap_t& map )
{
     map["check_cert"]
        .getFrom( &kernels::check_cert );
     map["check_cert_dbg"]
        .getFrom( &kernels::check_cert );
     map["check_cert2"]
        .getFrom( &kernels::check_cert2 );
     map["check_cert2_dbg"]
        .getFrom( &kernels::check_cert2 );
}


} //< namespace cudaNN
} //< namespace mpblocks


#endif // FATTR_CU_HPP_
