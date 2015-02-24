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
 *  @date   Jun 18, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 */

#ifndef MPBLOCKS_DUBINS_CURVES_CUDA2_POINTSET_HPP_
#define MPBLOCKS_DUBINS_CURVES_CUDA2_POINTSET_HPP_

#include <map>
#include <string>
#include <mpblocks/util/exception_stream.hpp>

namespace    mpblocks {
namespace      dubins {
namespace curves_cuda {

template <typename Format_t>
void ResultBlock<Format_t>::allocate(uint_t rows, uint_t cols) {
  if (m_buf) delete[] m_buf;

  m_buf = new float[rows * cols];

  m_rows = rows;
  m_cols = cols;
}

template <typename Format_t>
PointSet<Format_t>::PointSet(uint_t n, Format_t r)
    : m_sorter(-std::numeric_limits<Format_t>::max(),
               std::numeric_limits<Format_t>::max()) {
  m_g_in = 0;
  m_g_out = 0;
  m_g_sorted = 0;
  m_params.r = r;

  m_threadsPerBlock = 0;
  m_nSM = 0;

  deallocate();

  try {
    config();
    allocate(n);
  } catch (const std::exception& ex) {
    std::cerr << "Error in constructing dubins CUDA PointSet: " << ex.what()
              << "\nNote: point set is unallocated\n";
  }
}

template <typename Format_t>
PointSet<Format_t>::~PointSet() {
  deallocate();
}

template <typename Format_t>
void PointSet<Format_t>::deallocate() {
  if (m_g_in) {
    cuda::free(m_g_in);
    m_g_in = 0;
  }

  if (m_g_out) {
    cuda::free(m_g_out);
    m_g_out = 0;
  }

  if (m_g_sorted) {
    cuda::free(m_g_sorted);
    m_g_sorted = 0;
  }

  m_dbAlloc = 0;
  m_dbAlloc2 = 0;
  m_dbSize = 0;
}

template <typename Format_t>
void PointSet<Format_t>::allocate(uint_t n) {
  deallocate();

  m_dbAlloc = n;
  m_dbAlloc2 = cuda::nextPow2(n);

  m_g_in = cuda::mallocPitchT<Format_t>(m_pitchIn, m_dbAlloc, 3);
  std::cout << "allocated m_g_in for " << m_dbAlloc
            << " object with pitch: " << m_pitchIn << "\n";

  m_g_out = cuda::mallocPitchT<Format_t>(m_pitchOut, m_dbAlloc2, 2);
  std::cout << "allocated m_g_out for " << m_dbAlloc
            << " object with pitch: " << m_pitchOut << "\n";

  m_g_sorted = cuda::mallocPitchT<Format_t>(m_pitchOut, m_dbAlloc2, 2);
  std::cout << "allocated m_g_sorted for " << m_dbAlloc
            << " object with pitch: " << m_pitchOut << "\n";
}

template <typename Format_t>
void PointSet<Format_t>::set_r(Format_t r) {
  m_params.r = r;
}

template <typename Format_t>
void PointSet<Format_t>::clear(bool clearmem) {
  m_dbSize = 0;
  if (clearmem) {
    cuda::memset2DT(m_g_in, m_pitchIn, 0, m_dbAlloc, 3);
    cuda::memset2DT(m_g_out, m_pitchOut, 0, m_dbAlloc2, 2);
    cuda::memset2DT(m_g_sorted, m_pitchOut, 0, m_dbAlloc2, 2);
  }
}

template <typename Format_t>
void PointSet<Format_t>::config() {
  int devId = cuda::getDevice();
  config(devId);
}

template <typename Format_t>
void PointSet<Format_t>::config(int devId) {
  cuda::DeviceProp devProps(devId);
  cuda::FuncAttributes attr;
  uint_t maxRegs = 0;

  attr.getFrom(&kernels::distance_to_set<Format_t>);
  maxRegs = std::max(maxRegs, (uint_t)attr.numRegs);

  attr.getFrom(&kernels::distance_to_set_with_id<Format_t>);
  maxRegs = std::max(maxRegs, (uint_t)attr.numRegs);

  attr.getFrom(&kernels::distance_to_set_debug<Format_t>);
  maxRegs = std::max(maxRegs, (uint_t)attr.numRegs);

  attr.getFrom(&kernels::distance_from_set<Format_t>);
  maxRegs = std::max(maxRegs, (uint_t)attr.numRegs);

  attr.getFrom(&kernels::distance_from_set_with_id<Format_t>);
  maxRegs = std::max(maxRegs, (uint_t)attr.numRegs);

  attr.getFrom(&kernels::distance_from_set_debug<Format_t>);
  maxRegs = std::max(maxRegs, (uint_t)attr.numRegs);

  // the maximum number of threads we can put into a block is given by the
  // number of registers on each SM divided by the number of registers that
  // are used by each thread in the kernel
  uint_t threadCount_max = (uint_t)devProps.regsPerBlock / maxRegs;

  // make sure that the number of threads per block computed as above doesn't
  // exceed the max per-block for the architectture
  m_threadsPerBlock =
      std::min(threadCount_max, (uint_t)devProps.maxThreadsPerBlock);

  // get the number of multiprocessors
  m_nSM = devProps.multiProcessorCount;

  // configure the sorter
  m_sorter.config(devId);
}

template <typename Format_t>
void PointSet<Format_t>::computeGrid(uint_t& blocks, uint_t& threads) {
  threads = cuda::intDivideRoundUp(m_dbSize, m_nSM);
  if (threads > m_threadsPerBlock) threads = m_threadsPerBlock;
  blocks = cuda::intDivideRoundUp(m_dbSize, threads);
}

template <typename Format_t>
int PointSet<Format_t>::insert(Format_t q[3]) {
  cuda::memcpy2DT(m_g_in + m_dbSize, m_pitchIn, q, sizeof(Format_t), 1, 3,
                  cudaMemcpyHostToDevice);

  m_dbSize++;
  return m_dbSize - 1;
}

template <typename Format_t>
void PointSet<Format_t>::distance_to_set(Format_t q[3],
                                         ResultBlock<Format_t>& out) {
  m_params.set_q(q);

  uint_t blocks, threads;
  computeGrid(blocks, threads);

  size_t pitchIn = m_pitchIn / sizeof(Format_t);
  size_t pitchOut = m_pitchOut / sizeof(Format_t);

  const int DEBUG_ROWS = 4 * 11 + 4 * 13;

  switch (out.rows()) {
    case 1: {
      // call the kernel
      kernels::distance_to_set<Format_t><<<blocks,threads>>>(
              m_params,
              m_g_in,
              pitchIn,
              m_g_out,
              pitchOut,
              m_dbSize
              );
      cuda::deviceSynchronize();

      // retrieve results
      cuda::memcpy2DT(out.ptr(), out.pitch(), m_g_out, m_pitchOut, out.cols(),
                      2, cudaMemcpyDeviceToHost);

      break;
    }

    case 2: {
      // call the kernel
      kernels::distance_to_set_with_id<Format_t><<<blocks,threads>>>(
              m_params,
              m_g_in,
              pitchIn,
              m_g_out,
              pitchOut,
              m_dbSize
              );

      cuda::deviceSynchronize();

      // retrieve results
      cuda::memcpy2DT(out.ptr(), out.pitch(), m_g_out, m_pitchOut, out.cols(),
                      2, cudaMemcpyDeviceToHost);
      break;
    }

    case DEBUG_ROWS: {
      // allocate output storage
      size_t pitch = 1;
      Format_t* g_out =
          cuda::mallocPitchT<Format_t>(pitch, m_dbSize, DEBUG_ROWS);

      // call the kernel
      kernels::distance_to_set_debug<Format_t><<<blocks,threads>>>(
              m_params,
              m_g_in,
              pitchIn,
              g_out,
              pitch/sizeof(Format_t),
              m_dbSize
              );
      cuda::deviceSynchronize();

      // retrieve results
      cuda::memcpy2DT(out.ptr(), out.pitch(), g_out, pitch, out.cols(),
                      DEBUG_ROWS, cudaMemcpyDeviceToHost);

      // free output storage
      cuda::free(g_out);
      break;
    }

    default:
      utility::ex() << "PointSet::distance_to_set: "
                       "Valid output rows is 1,2, or 24";
      break;
  }
}

template <typename Format_t>
void PointSet<Format_t>::distance_from_set(Format_t q[3],
                                           ResultBlock<Format_t>& out) {
  m_params.set_q(q);

  uint_t blocks, threads;
  computeGrid(blocks, threads);

  size_t pitchIn = m_pitchIn / sizeof(Format_t);
  size_t pitchOut = m_pitchOut / sizeof(Format_t);

  switch (out.rows()) {
    case 1: {
      // call the kernel
      kernels::distance_from_set<Format_t><<<blocks,threads>>>(
               m_params,
              m_g_in,
              pitchIn,
              m_g_out,
              pitchOut,
              m_dbSize
              );
      cuda::deviceSynchronize();

      // retrieve results
      cuda::memcpy2DT(
              out.ptr(),  out.pitch(),
              m_g_out,    m_pitchOut,
              out.cols(), 2,
              cudaMemcpyDeviceToHost );
      break;
  }

  case 2: {
    // call the kernel
    kernels::distance_from_set_with_id<Format_t><<<blocks,threads>>>(
             m_params,
            m_g_in,
            pitchIn,
            m_g_out,
            pitchOut,
            m_dbSize
            );
    cuda::deviceSynchronize();

    // retrieve results
    cuda::memcpy2DT(out.ptr(), out.pitch(), m_g_out, m_pitchOut, out.cols(), 2,
                    cudaMemcpyDeviceToHost);
    break;
  }

  case 24: {
    // allocate output storage
    size_t pitch = 1;
    Format_t* g_out = cuda::mallocPitchT<Format_t>(pitch, m_dbSize, 24);

    // call the kernel
    kernels::distance_from_set_debug<Format_t><<<blocks,threads>>>(
             m_params,
            m_g_in,
            pitchIn,
            g_out,
            pitch/sizeof(Format_t),
            m_dbSize
            );
    cuda::deviceSynchronize();

    // retrieve results
    cuda::memcpy2DT(out.ptr(), out.pitch(), g_out, pitch, out.cols(), 24,
                    cudaMemcpyDeviceToHost);

    // free output storage
    cuda::free(g_out);
    break;
  }

  default:
    utility::ex() << "PointSet::distance_from_set: "
                     "Valid output rows is 1,2, or 24";
    break;
  }
}

template <typename Format_t>
void PointSet<Format_t>::nearest_children(Format_t q[3],
                                          ResultBlock<Format_t>& out) {
  m_params.set_q(q);

  uint_t blocks, threads;
  computeGrid(blocks, threads);

  size_t pitchIn = m_pitchIn / sizeof(Format_t);
  size_t pitchOut = m_pitchOut / sizeof(Format_t);

  // call the kernel to calculate distances to children
  kernels::distance_to_set_with_id<Format_t><<<blocks,threads>>>(
      m_params,
      m_g_in,
      pitchIn,
      m_g_out,
      pitchOut,
      m_dbSize
      );
  cuda::deviceSynchronize();

  Format_t* unsortedKeys = m_g_out;
  Format_t* unsortedVals = m_g_out + pitchOut;
  Format_t* sortedKeys = m_g_sorted;
  Format_t* sortedVals = m_g_sorted + pitchOut;

  // call the kernel to sort the results
  m_sorter.sort(sortedKeys, sortedVals, unsortedKeys, unsortedVals, m_dbSize,
                cuda::bitonic::Ascending);
  cuda::deviceSynchronize();

  // fetch the k smallest
  cuda::memcpy2DT(out.ptr(), out.pitch(), m_g_sorted, m_pitchOut, out.cols(), 2,
                  cudaMemcpyDeviceToHost);
}

template <typename Format_t>
void PointSet<Format_t>::nearest_parents(Format_t q[3],
                                         ResultBlock<Format_t>& out) {
  m_params.set_q(q);

  uint_t blocks, threads;
  computeGrid(blocks, threads);

  size_t pitchIn = m_pitchIn / sizeof(Format_t);
  size_t pitchOut = m_pitchOut / sizeof(Format_t);

  // call the kernel to calculate distances to children
  kernels::distance_from_set_with_id<Format_t><<<blocks,threads>>>(
      m_params,
      m_g_in,
      pitchIn,
      m_g_out,
      pitchOut,
      m_dbSize
      );
  cuda::deviceSynchronize();

  Format_t* unsortedKeys = m_g_out;
  Format_t* unsortedVals = m_g_out + pitchOut;
  Format_t* sortedKeys = m_g_sorted;
  Format_t* sortedVals = m_g_sorted + pitchOut;

  // call the kernel to sort the results
  m_sorter.sort(sortedKeys, sortedVals, unsortedKeys, unsortedVals, m_dbSize,
                cuda::bitonic::Ascending);
  cuda::deviceSynchronize();

  // fetch the k smallest
  cuda::memcpy2DT(out.ptr(), out.pitch(), m_g_sorted, m_pitchOut, out.cols(), 2,
                  cudaMemcpyDeviceToHost);
}

template <typename Format_t>
void PointSet<Format_t>::group_distance_to_set(Format_t q[3],
                                               ResultBlock<Format_t>& out) {
  uint_t blocks, threads;
  computeGrid(blocks, threads);

  size_t pitchIn = m_pitchIn / sizeof(Format_t);
  size_t pitchOut = m_pitchOut / sizeof(Format_t);

  EuclideanParams<Format_t> params;
  params.set_q(q);

  // call the kernel to calculate distances to children
  kernels::group_distance_to_set<Format_t><<<blocks,threads>>>(
      params,
      m_g_in,
      pitchIn,
      m_g_out,
      pitchOut,
      m_dbSize
      );
  cuda::deviceSynchronize();

  // fetch the distances
  cuda::memcpy2DT(out.ptr(), out.pitch(), m_g_out, m_pitchOut, out.cols(), 1,
                  cudaMemcpyDeviceToHost);
}

template <typename Format_t>
void PointSet<Format_t>::group_distance_neighbors(Format_t q[3],
                                                  ResultBlock<Format_t>& out) {
  uint_t blocks, threads;
  computeGrid(blocks, threads);

  size_t pitchIn = m_pitchIn / sizeof(Format_t);
  size_t pitchOut = m_pitchOut / sizeof(Format_t);

  EuclideanParams<Format_t> params;
  params.set_q(q);

  // call the kernel to calculate distances to children
  kernels::group_distance_to_set_with_id<Format_t><<<blocks,threads>>>(
      params,
      m_g_in,
      pitchIn,
      m_g_out,
      pitchOut,
      m_dbSize
      );
  cuda::deviceSynchronize();

  Format_t* unsortedKeys = m_g_out;
  Format_t* unsortedVals = m_g_out + pitchOut;
  Format_t* sortedKeys = m_g_sorted;
  Format_t* sortedVals = m_g_sorted + pitchOut;

  // call the kernel to sort the results
  m_sorter.sort(sortedKeys, sortedVals, unsortedKeys, unsortedVals, m_dbSize,
                cuda::bitonic::Ascending);
  cuda::deviceSynchronize();

  // fetch the k smallest
  cuda::memcpy2DT(out.ptr(), out.pitch(), m_g_sorted, m_pitchOut, out.cols(), 2,
                  cudaMemcpyDeviceToHost);
}

template <typename Format_t>
void PointSet<Format_t>::get_fattr(fattrMap_t& map) {
  map["distance_to_set"].getFrom(&kernels::distance_to_set<Format_t>);
  map["distance_to_set_with_id"].getFrom(
      &kernels::distance_to_set_with_id<Format_t>);
  map["distance_to_set_debug"].getFrom(
      &kernels::distance_to_set_debug<Format_t>);
  map["distance_from_set"].getFrom(&kernels::distance_from_set<Format_t>);
  map["distance_from_set_with_id"].getFrom(
      &kernels::distance_from_set_with_id<Format_t>);
  map["distance_from_set_debug"].getFrom(
      &kernels::distance_from_set_debug<Format_t>);
  map["euclidean_to_set"].getFrom(&kernels::group_distance_to_set<Format_t>);
  map["euclidean_to_set_with_id"].getFrom(
      &kernels::group_distance_to_set<Format_t>);

  // Sorter_t::get_fattr(map);
}

} // curves
} // dubins
} // mpblocks

#endif // MPBLOCKS_DUBINS_CURVES_CUDA2_POINTSET_HPP_
