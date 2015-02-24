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
 *  @file
 *  @date   Aug 25, 2011
 *  @author Josh Bialkowski (jbialk@mit.edu)
 *  @brief
 */


#include <sstream>
#include <stdexcept>
#include <cstring>
#include <iostream>
#include <iomanip>

#include <mpblocks/cuda.h>

namespace mpblocks {
namespace     cuda {

void checkResult(cudaError_t result, const std::string& where) {
  if (result != cudaSuccess) cudaEx(result)() << "In " << where;
}

void* malloc(size_t size) {
  void* devPtr;

  cudaError_t error = cudaMalloc(&devPtr, size);
  if (error) {
    cudaEx(error)() << "In cudaMalloc, failed to allocate " << size << "bytes";
  }

  return devPtr;
}

void* mallocPitch(size_t& pitch, size_t cols, size_t rows) {
  void* devPtr;
  cudaError_t error = cudaMallocPitch(&devPtr, &pitch, cols, rows);
  if (error) {
    cudaEx(error)() << "In cudaMallocPitch, failed to allocate (" << rows << "x"
                    << cols << ") array";
  }

  return devPtr;
}

void free(void* devPtr) {
  cudaError_t error = cudaFree(devPtr);
  if (error) cudaEx(error)() << "In cudaFree, failed to free " << devPtr;
}

void memcpy(void* dst, const void* src, size_t count, MemcpyKind kind) {
  cudaError_t error = cudaMemcpy(dst, src, count, (cudaMemcpyKind)kind);
  if (error)
    cudaEx(error)() << "cudaMemcpy: Failed to copy " << count << "bytes from "
                    << src << " to " << dst;
}

void memcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch,
              size_t width, size_t height, MemcpyKind kind) {
  cudaError_t error = cudaMemcpy2D(dst, dpitch, src, spitch, width, height,
                                   (cudaMemcpyKind)kind);

  if (error)
    cudaEx(error)() << "cudaMemcpy2D: Failed to copy (" << height << "x"
                    << width << ") bytes  from" << src << "(" << spitch
                    << ") to " << dst << "(" << dpitch << ") ";
}

void memset(void* devPtr, int value, size_t count) {
  cudaError_t error = cudaMemset(devPtr, value, count);

  if (error)
    cudaEx(error)() << "cudaMemset: Failed to set " << count
                    << " bytes to value " << value << " at " << devPtr;
}

/// wraps cudaMemset2D
void memset2D(void* devPtr, size_t pitch, int value, size_t width,
              size_t height) {
  cudaError_t error = cudaMemset2D(devPtr, pitch, value, width, height);

  if (error)
    cudaEx(error)() << "cudaMemset: Failed to set (" << height << " x " << width
                    << ") bytes to value " << value << " at " << devPtr
                    << " (pitch=" << pitch << ")";
}

int getDeviceCount() {
  int count;
  cudaError_t error = cudaGetDeviceCount(&count);
  if (error) cudaEx(error)() << "cudaGetDeviceCount";
  return count;
}

FuncAttributes::FuncAttributes() { ::memset(this, 0, sizeof(FuncAttributes)); }

DeviceProp::DeviceProp() { ::memset(this, 0, sizeof(DeviceProp)); }

DeviceProp::DeviceProp(int device) { getFrom(device); }

void DeviceProp::getFrom(int device) {
  cudaError_t error = cudaGetDeviceProperties(this, device);
  if (error) cudaEx(error)() << "DeviceProp::getFrom(" << device << ")";
}

int getDevice() {
  int dev;
  cudaError_t error = cudaGetDevice(&dev);
  if (error) cudaEx(error)() << "cudaGetDevice";
  return dev;
}

void setDevice(int dev) {
  cudaError_t error = cudaSetDevice(dev);
  if (error) cudaEx(error)() << "cudaSetDevice(" << dev << ")";
}

void deviceSynchronize() {
  cudaError_t error = cudaDeviceSynchronize();
  if (error) cudaEx(error)() << "cudaDeviceSynchronize";
}

void checkLastError(const std::string& msg) {
  cudaError_t error = cudaGetLastError();
  if (error) cudaEx(error)() << msg;
}

void printDeviceReport(std::ostream& out) {
  printDeviceReport(out, getDevice());
}

void printDeviceReport( std::ostream& out, int dev )
{
  typedef std::map<CUdevice_attribute,std::string> AttrMap_t;
  AttrMap_t  attrMap;
  attrMap[CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK             ] = "Maximum number of threads per block ";
  attrMap[CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X                   ] = "Maximum block dimension X ";
  attrMap[CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y                   ] = "Maximum block dimension Y ";
  attrMap[CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z                   ] = "Maximum block dimension Z ";
  attrMap[CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X                    ] = "Maximum grid dimension X ";
  attrMap[CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y                    ] = "Maximum grid dimension Y ";
  attrMap[CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z                    ] = "Maximum grid dimension Z ";
  attrMap[CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK       ] = "Maximum shared memory available per block in bytes ";
  attrMap[CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY             ] = "Memory available on device for __constant__ vars in bytes ";
  attrMap[CU_DEVICE_ATTRIBUTE_WARP_SIZE                         ] = "Warp size in threads ";
  attrMap[CU_DEVICE_ATTRIBUTE_MAX_PITCH                         ] = "Maximum pitch in bytes allowed by memory copies ";
  attrMap[CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK           ] = "Maximum number of 32-bit registers available per block ";
  attrMap[CU_DEVICE_ATTRIBUTE_CLOCK_RATE                        ] = "Peak clock frequency in kilohertz ";
  attrMap[CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT                 ] = "Alignment requirement for textures ";
  attrMap[CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT              ] = "Number of multiprocessors on device ";
  attrMap[CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT               ] = "Specifies whether there is a run time limit on kernels ";
  attrMap[CU_DEVICE_ATTRIBUTE_INTEGRATED                        ] = "Device is integrated with host memory ";
  attrMap[CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY               ] = "Device can map host memory into CUDA address space ";
  attrMap[CU_DEVICE_ATTRIBUTE_COMPUTE_MODE                      ] = "Compute mode (See ::CUcomputemode for details) ";
  attrMap[CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH           ] = "Maximum 1D texture width ";
  attrMap[CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH           ] = "Maximum 2D texture width ";
  attrMap[CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT          ] = "Maximum 2D texture height ";
  attrMap[CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH           ] = "Maximum 3D texture width ";
  attrMap[CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT          ] = "Maximum 3D texture height ";
  attrMap[CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH           ] = "Maximum 3D texture depth ";
  attrMap[CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH   ] = "Maximum 2D layered texture width ";
  attrMap[CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT  ] = "Maximum 2D layered texture height ";
  attrMap[CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS  ] = "Maximum layers in a 2D layered texture ";
  attrMap[CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT                 ] = "Alignment requirement for surfaces ";
  attrMap[CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS                ] = "Device can possibly execute multiple kernels concurrently ";
  attrMap[CU_DEVICE_ATTRIBUTE_ECC_ENABLED                       ] = "Device has ECC support enabled ";
  attrMap[CU_DEVICE_ATTRIBUTE_PCI_BUS_ID                        ] = "PCI bus ID of the device ";
  attrMap[CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID                     ] = "PCI device ID of the device ";
  attrMap[CU_DEVICE_ATTRIBUTE_TCC_DRIVER                        ] = "Device is using TCC driver model ";
  attrMap[CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE                 ] = "Peak memory clock frequency in kilohertz ";
  attrMap[CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH           ] = "Global memory bus width in bits ";
  attrMap[CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE                     ] = "Size of L2 cache in bytes ";
  attrMap[CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR    ] = "Maximum resident threads per multiprocessor ";
  attrMap[CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT                ] = "Number of asynchronous engines ";
  attrMap[CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING                ] = "Device shares a unified address space with the host ";
  attrMap[CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH   ] = "Maximum 1D layered texture width ";
  attrMap[CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS  ] = "Maximum layers in a 1D layered texture ";
  attrMap[CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID                     ] = "PCI domain ID of the device ";

  // get a handle for the i'th device
  CUdevice cuDevice;
  cuDeviceGet(&cuDevice, dev);

  // query all the interesting stuff
  // first get the largest string size
  size_t nMax = 0;
  for (AttrMap_t::iterator iPair = attrMap.begin(); iPair != attrMap.end();
       ++iPair) {
    nMax = std::max(nMax, iPair->second.size());
  }

  out << "Device Attributes:\n--------------------------\n";
  for (AttrMap_t::iterator iPair = attrMap.begin(); iPair != attrMap.end();
       ++iPair) {
    int val;
    CUresult result = cuDeviceGetAttribute(&val, iPair->first, cuDevice);
    if (result != CUDA_SUCCESS) out << "Error getting device attributes\n";

    std::string& msg = iPair->second;
    int n = msg.size();
    for (size_t i = n; i < nMax; i++) out << " ";
    out << msg;
    out << ": " << val << "\n";
  }

  out << "\nDevice Properties:\n--------------------------\n";
  CUdevprop propMap;
  cuDeviceGetProperties(&propMap, cuDevice);
  out
      <<   "    threads per block: " << propMap.maxThreadsPerBlock
      << "\n      max threads dim: " << propMap.maxThreadsDim[0] << 'x'
                                     << propMap.maxThreadsDim[1] << 'x'
                                     << propMap.maxThreadsDim[2]
      << "\n        max grid size: " << propMap.maxGridSize[0] << 'x'
                                     << propMap.maxGridSize[1] << 'x'
                                     << propMap.maxGridSize[2] << 'x'
      << "\n shared mem per block: " << propMap.sharedMemPerBlock
      << "\n   total const memory: " << propMap.totalConstantMemory
      << "\n            SIMDWidth: " << propMap.SIMDWidth
      << "\n            mem pitch: " << propMap.memPitch
      << "\n       regs per block: " << propMap.regsPerBlock
      << "\n           clock rate: " << propMap.clockRate
      << "\n        texture align: " << propMap.textureAlign
      << "\n";
}

static int getAttr(const cudaFuncAttributes& attr, int i) {
  switch (i) {
    case 0:
      return attr.binaryVersion;
    case 1:
      return attr.constSizeBytes;
    case 2:
      return attr.localSizeBytes;
    case 3:
      return attr.maxThreadsPerBlock;
    case 4:
      return attr.numRegs;
    case 5:
      return attr.ptxVersion;
    case 6:
      return attr.sharedSizeBytes;
    default:
      return 0;
  }
}

static const char* attrName[] = {
    "binary version", "const size (bytes)", "local size (bytes)",
    "max threads per block", "num regs", "ptx version", "shared size (bytes)"};

static char getAttrKey(int i) { return 'a' + i; }

void printKernelReport(std::ostream& out, const fattrMap_t& fMap) {
  // first pass, output kernel ids
  out << "Kernels:\n----------------\n";
  int i = 0;
  for (fattrMap_t::const_iterator iPair = fMap.begin(); iPair != fMap.end();
       ++iPair) {
    const std::string& kernel = iPair->first;
    //        const FuncAttributes&  fAttr  = iPair->second;
    out << std::setw(3) << i++ << " : " << kernel << "\n";
  }

  out << "\nAttributes:\n-----------------\n";
  for (int i = 0; i < 7; i++)
    out << std::setw(3) << getAttrKey(i) << " : " << attrName[i] << "\n";
  out << "\n";

  const int fieldW = 5;
  const int nameW = 3;
  const int numAttr = 7;
  //    const char* pad = "  ";

  out << std::setw(nameW) << " "
      << "   ";
  for (int i = 0; i < numAttr; i++) out << std::setw(fieldW) << getAttrKey(i);
  out << "\n";

  out << std::setw(nameW) << " "
      << "   ";
  for (int i = 0; i < numAttr; i++) out << std::setw(fieldW) << "----";
  out << "\n";

  i = 0;
  for (fattrMap_t::const_iterator iPair = fMap.begin(); iPair != fMap.end();
       ++iPair) {
    out << std::setw(nameW) << i++ << " : ";
    for (int j = 0; j < numAttr; j++)
      out << std::setw(fieldW) << getAttr(iPair->second, j);
    out << "\n";
  }
}

} // namespace cuda
} // namespace mpblocks
