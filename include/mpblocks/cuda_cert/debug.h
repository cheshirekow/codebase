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
 *  @file   /home/josh/Codes/cpp/mpblocks2/cuda_cert/include/mpblocks/cuda_cert/debug.h
 *
 *  @date   Oct 27, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_CUDA_CERT_DEBUG_H_
#define MPBLOCKS_CUDA_CERT_DEBUG_H_

#include <mpblocks/cuda.h>

namespace mpblocks  {
namespace cuda_cert {

typedef unsigned int uint_t;
__device__ const uint_t numDebugThreads = 10;
__device__ const uint_t sizeDebugOutput = 70;
__device__ const uint_t sizeDebugBuffer = sizeDebugOutput*numDebugThreads;

const uint_t host_numDebugThreads = 10;
const uint_t host_sizeDebugOutput = 70;
const uint_t host_sizeDebugBuffer = sizeDebugOutput*numDebugThreads;



} //< namespace cuda_cert
} //< namespace mpblocks














#endif // DEBUG_H_
