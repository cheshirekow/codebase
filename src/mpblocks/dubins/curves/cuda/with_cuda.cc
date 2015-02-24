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
 *  \file   src/CudaHelper.h
 *
 *  \date   Oct 24, 2012
 *  \author Josh Bialkowski (jbialk@mit.edu)
 *  \brief  
 */


#include <cuda.h>
#include <cuda_runtime.h>
#include <mpblocks/gtk.hpp>
#include <mpblocks/dubins/curves_cuda.hpp>
#include "algo.h"
#include "cuda_helper.h"

namespace mpblocks {
namespace examples {
namespace   dubins {

class CudaHelper_impl : public CudaHelper {
 private:
  int m_deviceCount;

 public:
  CudaHelper_impl();
  virtual ~CudaHelper_impl(){};
  virtual void populateDevices(gtk::LayoutMap& layout);
  virtual void populateDetails(gtk::LayoutMap& layout);

  virtual void solve(gtk::LayoutMap& layout, Vector3d& q0, Vector3d& q1,
                     double r);
};

CudaHelper* create_cudaHelper() { return new CudaHelper_impl(); }

CudaHelper_impl::CudaHelper_impl() {
  // initialize cuda
  cuInit(0);
}

static int getAttr(cudaFuncAttributes& attr, int i) {
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

void CudaHelper_impl::populateDevices(gtk::LayoutMap& layout) {
  Gtk::ComboBoxText* combo = layout.widget<Gtk::ComboBoxText>("cudaDevices");

  // get the number of devices supporting cuda
  cuDeviceGetCount(&m_deviceCount);

  if (m_deviceCount < 1)
    combo->append("No devices found");

  else {
    const size_t nName = 100;
    char nameBuf[nName];

    for (int i = 0; i < m_deviceCount; i++) {
      // get a handle for the i'th device
      CUdevice cuDevice;
      cuDeviceGet(&cuDevice, i);

      // get the name of the device
      cuDeviceGetName(nameBuf, nName, cuDevice);

      std::stringstream strm;
      strm << "[" << i << "]: " << nameBuf;
      combo->append(strm.str());
    }
  }

  combo->set_active(0);

  // get kernel info for cuda2
  typedef std::map<std::string, cuda::FuncAttributes> fattrMap_t;
  fattrMap_t fMap;
  mpblocks::dubins::curves_cuda::PointSet<float>::get_fattr(fMap);

  // first pass, output kernel ids
  std::stringstream outbuf;
  outbuf << "Kernels:\n----------------\n";
  int i = 0;
  for (fattrMap_t::iterator iPair = fMap.begin(); iPair != fMap.end();
       ++iPair) {
    const std::string& kernel = iPair->first;
    outbuf << std::setw(3) << i++ << " : " << kernel << "\n";
  }
  outbuf << "\n";
  outbuf << "Attributes:\n-----------------\n";
  for (int i = 0; i < 7; i++) {
    outbuf << std::setw(3) << getAttrKey(i) << " : " << attrName[i] << "\n";
  }
  outbuf << "\n";

  const int fieldW = 5;
  const int nameW = 3;
  const int numAttr = 7;

  outbuf << std::setw(nameW) << " "
         << "   ";
  for (int i = 0; i < numAttr; i++)
    outbuf << std::setw(fieldW) << getAttrKey(i);
  outbuf << "\n";

  outbuf << std::setw(nameW) << " "
         << "   ";
  for (int i = 0; i < numAttr; i++) outbuf << std::setw(fieldW) << "----";
  outbuf << "\n";

  i = 0;
  for (fattrMap_t::iterator iPair = fMap.begin(); iPair != fMap.end();
       ++iPair) {
    outbuf << std::setw(nameW) << i++ << " : ";
    for (int j = 0; j < numAttr; j++)
      outbuf << std::setw(fieldW) << getAttr(iPair->second, j);
    outbuf << "\n";
  }

  layout.object<Gtk::TextBuffer>("cudaKernels")->set_text(outbuf.str());
}

void CudaHelper_impl::populateDetails(gtk::LayoutMap& layout) {
  int iDevice =
      layout.widget<Gtk::ComboBoxText>("cudaDevices")->get_active_row_number();
  if (iDevice >= m_deviceCount) return;

  cudaError_t error = cudaSetDevice(iDevice);
  if (error) {
    std::cerr << "Failed to set cuda device to : " << iDevice << " "
              << cudaGetErrorString(error) << std::endl;
  }

  typedef std::map<CUdevice_attribute,std::string> AttrMap_t;
  AttrMap_t  attrMap;
  attrMap[CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK] =
      "Maximum number of threads per block ";
  attrMap[CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X] = "Maximum block dimension X ";
  attrMap[CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y] = "Maximum block dimension Y ";
  attrMap[CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z] = "Maximum block dimension Z ";
  attrMap[CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X] = "Maximum grid dimension X ";
  attrMap[CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y] = "Maximum grid dimension Y ";
  attrMap[CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z] = "Maximum grid dimension Z ";
  attrMap[CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK] =
      "Maximum shared memory available per block in bytes ";
  attrMap[CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY] =
      "Memory available on device for __constant__ vars in bytes ";
  attrMap[CU_DEVICE_ATTRIBUTE_WARP_SIZE] = "Warp size in threads ";
  attrMap[CU_DEVICE_ATTRIBUTE_MAX_PITCH] =
      "Maximum pitch in bytes allowed by memory copies ";
  attrMap[CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK] =
      "Maximum number of 32-bit registers available per block ";
  attrMap[CU_DEVICE_ATTRIBUTE_CLOCK_RATE] =
      "Peak clock frequency in kilohertz ";
  attrMap[CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT] =
      "Alignment requirement for textures ";
  attrMap[CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT] =
      "Number of multiprocessors on device ";
  attrMap[CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT] =
      "Specifies whether there is a run time limit on kernels ";
  attrMap[CU_DEVICE_ATTRIBUTE_INTEGRATED] =
      "Device is integrated with host memory ";
  attrMap[CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY] =
      "Device can map host memory into CUDA address space ";
  attrMap[CU_DEVICE_ATTRIBUTE_COMPUTE_MODE] =
      "Compute mode (See ::CUcomputemode for details) ";
  attrMap[CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH] =
      "Maximum 1D texture width ";
  attrMap[CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH] =
      "Maximum 2D texture width ";
  attrMap[CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT] =
      "Maximum 2D texture height ";
  attrMap[CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH] =
      "Maximum 3D texture width ";
  attrMap[CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT] =
      "Maximum 3D texture height ";
  attrMap[CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH] =
      "Maximum 3D texture depth ";
  attrMap[CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH] =
      "Maximum 2D layered texture width ";
  attrMap[CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT] =
      "Maximum 2D layered texture height ";
  attrMap[CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS] =
      "Maximum layers in a 2D layered texture ";
  attrMap[CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT] =
      "Alignment requirement for surfaces ";
  attrMap[CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS] =
      "Device can possibly execute multiple kernels concurrently ";
  attrMap[CU_DEVICE_ATTRIBUTE_ECC_ENABLED] = "Device has ECC support enabled ";
  attrMap[CU_DEVICE_ATTRIBUTE_PCI_BUS_ID] = "PCI bus ID of the device ";
  attrMap[CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID] = "PCI device ID of the device ";
  attrMap[CU_DEVICE_ATTRIBUTE_TCC_DRIVER] = "Device is using TCC driver model ";
  attrMap[CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE] =
      "Peak memory clock frequency in kilohertz ";
  attrMap[CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH] =
      "Global memory bus width in bits ";
  attrMap[CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE] = "Size of L2 cache in bytes ";
  attrMap[CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR] =
      "Maximum resident threads per multiprocessor ";
  attrMap[CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT] =
      "Number of asynchronous engines ";
  attrMap[CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING] =
      "Device shares a unified address space with the host ";
  attrMap[CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH] =
      "Maximum 1D layered texture width ";
  attrMap[CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS] =
      "Maximum layers in a 1D layered texture ";
  attrMap[CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID] = "PCI domain ID of the device ";

  // where we put the output
  std::stringstream outbuf;

  Glib::RefPtr<Gtk::TextBuffer> attr =
      layout.object<Gtk::TextBuffer>("cudaAttributes");
  Glib::RefPtr<Gtk::TextBuffer> prop =
      layout.object<Gtk::TextBuffer>("cudaProperties");

  // get a handle for the i'th device
  CUdevice cuDevice;
  cuDeviceGet(&cuDevice, iDevice);

  // query all the interesting stuff
  // first get the largest string size
  size_t nMax = 0;
  for (AttrMap_t::iterator iPair = attrMap.begin(); iPair != attrMap.end();
       ++iPair) {
    nMax = std::max(nMax, iPair->second.size());
  }

  for (AttrMap_t::iterator iPair = attrMap.begin(); iPair != attrMap.end();
       ++iPair) {
    int val;
    CUresult result = cuDeviceGetAttribute(&val, iPair->first, cuDevice);
    if (result == CUDA_SUCCESS) {
      std::string& msg = iPair->second;
      int n = msg.size();
      for (size_t i = n; i < nMax; i++) outbuf << " ";
      outbuf << msg;
      outbuf << ": " << val << "\n";
    }
  }
  attr->set_text(outbuf.str());

  CUdevprop propMap;
  cuDeviceGetProperties(&propMap, cuDevice);

  outbuf.str("");
  outbuf
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
  prop->set_text( outbuf.str() );
}

void CudaHelper_impl::solve(gtk::LayoutMap& layout, Vector3d& q0, Vector3d& q1,
                            double r) {
  using namespace mpblocks::dubins::curves_cuda;
  using namespace cuda::linalg2;

  float q0in[3], q1in[3];
  for (int i = 0; i < 3; i++) {
    q0in[i] = q0[i];
    q1in[i] = q1[i];
  }

  try {
    PointSet<float> pointset;
    pointset.set_r(r);
    pointset.insert(q1in);

    const int rows = 11 * 4 + 13 * 4;
    const int cols = 1;
    ResultBlock<float> results(rows, cols);
    pointset.distance_to_set(q0in, results);

    // solutions for a curve in the middle
    for (int i = 0; i < 4; i++) {
      int r = 11 * i;
      float feasible = results(r + 10, 0);
      if (feasible) {
        std::stringstream buf;
        buf << std::setprecision(6) << results(r + 9, 0);
        layout.object<Gtk::EntryBuffer>(algo::buf[i])->set_text(buf.str());

        for (int j = 0; j < 3; j++) {
          float val = results(r + 6 + j, 0) * 180 / M_PI;
          std::stringstream entry;
          entry << algo::buf[i] << (j + 1);
          std::stringstream buf;
          buf << std::setprecision(6) << val;
          layout.object<Gtk::EntryBuffer>(entry.str())->set_text(buf.str());
        }
      } else {
        layout.object<Gtk::EntryBuffer>(algo::buf[i])->set_text("");

        for (int j = 0; j < 3; j++) {
          std::stringstream entry;
          entry << algo::buf[i] << (j + 1);
          layout.object<Gtk::EntryBuffer>(entry.str())->set_text("");
        }
      }
    }

    // solutions for a straight in the middle
    for (int i = 4; i < mpblocks::dubins::INVALID; i++) {
      int r = 11 * 4 + 13 * (i - 4);
      float feasible = results(r + 12, 0);
      if (feasible) {
        std::stringstream buf;
        buf << std::setprecision(6) << results(r + 11, 0);
        layout.object<Gtk::EntryBuffer>(algo::buf[i])->set_text(buf.str());

        for (int j = 0; j < 3; j++) {
          float val = results(r + 6 + j, 0);
          if (j != 2) val *= 180 / M_PI;

          std::stringstream entry;
          entry << algo::buf[i] << (j + 1);
          std::stringstream buf;
          buf << std::setprecision(6) << val;
          layout.object<Gtk::EntryBuffer>(entry.str())->set_text(buf.str());
        }
      } else {
        layout.object<Gtk::EntryBuffer>(algo::buf[i])->set_text("");

        for (int j = 0; j < 3; j++) {
          std::stringstream entry;
          entry << algo::buf[i] << (j + 1);
          layout.object<Gtk::EntryBuffer>(entry.str())->set_text("");
        }
      }
    }
  } catch (const std::exception& ex) {
    std::cerr << "Failed to launch distance kernel\n   " << ex.what()
              << std::endl;
  }
}

} // dubins
} // examples
} // mpblocks


