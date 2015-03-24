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
 *  @file   cuda/wrap.h
 *
 *  @date   Aug 25, 2011
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief
 */

#ifndef MPBLOCKS_CUDA_WRAP_H_
#define MPBLOCKS_CUDA_WRAP_H_

#include <map>

namespace mpblocks {
namespace     cuda {


typedef cudaMemcpyKind      MemcpyKind;

struct FuncAttributes:
    public cudaFuncAttributes
{
    FuncAttributes();

    template <typename T>
    void getFrom( T* entry );
};


struct DeviceProp:
    public cudaDeviceProp
{
    DeviceProp();
    DeviceProp(int device);

    void getFrom( int device );
};


/// throws an exception if result is not success
void checkResult( cudaError_t result, const std::string& where);

/// wraps cudaMalloc
void* malloc(size_t size);

/// wraps cudaMallocPitch
void* mallocPitch( size_t& pitch, size_t cols, size_t rows );

/// allocates @p nObjs objects of type T
template <typename T>
T* mallocT( size_t nObjs );

/// allocates @p nObjs objects of type T
/// @note the pitch returned is in *bytes*, not in nObjs
template <typename T>
T* mallocPitchT( size_t& pitch, size_t obsPerRow, size_t cols );

/// wraps cudaFree
void free(void* devPtr);

/// wraps cudaMemcpy
void memcpy( void *dst, const void *src, size_t count, MemcpyKind kind);

/// wraps cudaMemcpy2D
void memcpy2D(
        void*       dst,
        size_t      dpitch,
        const void* src,
        size_t      spitch,
        size_t      width,
        size_t      height,
        MemcpyKind  kind );

/// wraps cudaMemcpy2D
template <typename T>
void memcpy2DT(
        T*          dst,
        size_t      dpitchBytes,
        const T*    src,
        size_t      spitchBytes,
        size_t      widthObs,
        size_t      height,
        MemcpyKind  kind );

/// allocates \p nObjs objects of type T
template <typename T>
void memcpyT( T* dst, const T* src,  size_t nObjs, MemcpyKind kind );

/// wraps cudaMemset
void memset( void* devPtr, int value, size_t count );

/// wraps cudaMemset
template <typename T>
void memset( T* devPtr, int value, size_t nObs );

/// wraps cudaMemset2D
void memset2D(
        void*   devPtr,
        size_t  pitch,
        int     value,
        size_t  width,
        size_t  height );

/// wraps cudaMemset2D
template <typename T>
void memset2DT(
        T*      devPtr,
        size_t  pitchBytes,
        int     value,
        size_t  widthObjs,
        size_t  height );

/// wraps cudaGetDeviceCount
int getDeviceCount();

/// wraps cudaGetDeviceProperties
DeviceProp getDeviceProperties(int dev);

/// wraps cudaGetDevice
int getDevice();

/// wraps cudaSetDevice
void setDevice(int dev);

/// blocks the host thread until kernels are done executing
void deviceSynchronize();

/// wraps getLastError
void checkLastError(const std::string& msg="checkLastError");

/// build a list of device names
template <class oiter>
void getDeviceList( oiter& out )
{
    int nDevices = getDeviceCount();
    for(int dev=0; dev < nDevices; dev++)
    {
        DeviceProp prop = getDeviceProperties(dev);
        *(out++) = prop.name;
    }
}

/// print a formatted report about the specified device
void printDeviceReport( std::ostream& out, int dev );
void printDeviceReport( std::ostream& out );

/// print a formatted report about the specified kernels
typedef std::map<std::string,FuncAttributes> fattrMap_t;
void printKernelReport( std::ostream& out, const fattrMap_t& attr );


} // namespace cuda
} // namespace mpblocks



#endif /* UTILS_H_ */
