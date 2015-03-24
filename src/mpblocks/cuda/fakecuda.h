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
 *  @file   mpblocks/cuda/fakecuda.h
 *
 *  @date   Dec 3, 2012
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief  
 */

#ifndef MPBLOCKS_CUDA_FAKECUDA_H_
#define MPBLOCKS_CUDA_FAKECUDA_H_


#ifndef __CUDACC__

#define __global__
#define __device__
#define __constant__
#define __shared__
#define __host__
#define __forceinline__

void __syncthreads();

struct Dim3
{
    int x,y,z;
};

extern Dim3 threadIdx;
extern Dim3 blockIdx;
extern Dim3 blockDim;
extern Dim3 gridDim;

#endif



#endif // FAKECUDA_H_
