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
 *  @file   cuda/wrap.hpp
 *
 *  @date   Aug 25, 2011
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief
 */
#ifndef MPBLOCKS_CUDA_WRAP_HPP_
#define MPBLOCKS_CUDA_WRAP_HPP_


namespace mpblocks {
namespace     cuda {


template <typename T>
void FuncAttributes::getFrom( T* entry )
{
    cudaError_t result = cudaFuncGetAttributes( this, (void*)entry );
    cudaEx(result) << "In cudaFuncGetAttributes";
}

template <typename T>
T* mallocT( size_t nObjs )
{
    T* devPtr = (T*)cuda::malloc(nObjs*sizeof(T) );
    return devPtr;
}

template <typename T>
T* mallocPitchT( size_t& pitch, size_t obsPerRow, size_t cols )
{
    T* devPtr = (T*)cuda::mallocPitch(pitch, obsPerRow*sizeof(T), cols );
    return devPtr;
}

template <typename T>
void memcpyT( T* dst, const T* src,  size_t nObjs, MemcpyKind kind )
{
    cuda::memcpy( dst, src, nObjs*sizeof(T), kind );
}

template <typename T>
void memcpy2DT(
        T*          dst,
        size_t      dpitchBytes,
        const T*    src,
        size_t      spitchBytes,
        size_t      widthObjs,
        size_t      height,
        MemcpyKind  kind )
{
    cuda::memcpy2D( (void*)dst,       dpitchBytes,
                    (const void*)src, spitchBytes,
                    widthObjs*sizeof(T), height,
                    kind );

}


/// wraps cudaMemset
template <typename T>
void memset( T* devPtr, int value, size_t nObs )
{
    cuda::memset( (void*)devPtr, value, nObs*sizeof(T) );
}

/// wraps cudaMemset2D
template <typename T>
void memset2DT(
        T*      devPtr,
        size_t  pitchBytes,
        int     value,
        size_t  widthObjs,
        size_t  height )
{
    cuda::memset2D( (void*)devPtr, pitchBytes, value,
                    widthObjs*sizeof(T), height );

}



} // namespace cuda
} // namespace mpblocks








#endif /* UTILS_H_ */
