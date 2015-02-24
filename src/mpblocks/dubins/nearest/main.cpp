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
 *  \file   src/main.cpp
 *
 *  \date   Oct 24, 2012
 *  \author Josh Bialkowski (jbialk@mit.edu)
 *  \brief  
 */


#include <set>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <Eigen/Dense>
#include <mpblocks/dubins/curves_cuda.hpp>
#include <mpblocks/dubins/curves_eigen.hpp>
#include <mpblocks/dubins/kd_tree.hpp>
#include <mpblocks/utility/Timespec.h>
#include <boost/format.hpp>

using namespace Eigen;
using namespace mpblocks;
using namespace mpblocks::cuda;
using namespace mpblocks::dubins;
using namespace mpblocks::utility;

typedef unsigned int uint;

// test configuration
const uint  g_nSamples = 32768;
const uint  g_nTrials  = 10;
const uint  g_k        = 10;
const float g_r        = 30;    // dubins radius
const float g_w        = 100;   // worspace width
const float g_h        = 100;   // workspace height
const int   g_seed     = 0;

struct Solution {
  uint idx;
  int id;
  float dist;

  Solution() : idx(0), id(curves_eigen::INVALID), dist(0) {}

  Solution(uint idx, int id, float dist) : idx(idx), id(id), dist(dist) {}

  struct Compare {
    bool operator()(const Solution& a, const Solution& b) {
      if (a.dist < b.dist)
        return true;
      else if (a.dist == b.dist) {
        if (a.idx < b.idx)
          return true;
        else
          return false;
      } else
        return false;
    }
  };
};

typedef std::set<Solution, Solution::Compare> SortedSolns;

typedef curves_cuda::PackedIndex<float> CudaOutput;
typedef curves_cuda::ResultBlock<float> CudaResult;
typedef curves_cuda::PointSet<float> CudaPoints;

typedef dubins::kd_tree::Traits<float> KdTraits;
typedef dubins::kd_tree::KNearest<float> KdKNearest;
typedef mpblocks::kd_tree::Tree<KdTraits> KdTree;
typedef typename KdTraits::Node KdNode;

typedef boost::format fmt;

int main(int argc, char** argv) {
  // create the query point
  float    q_gpu[3] = {0, 0, 0};
  Vector3f q_cpu      (0, 0, 0);

  // the cuda point set
  CudaPoints  cudaPoints(g_nSamples,g_r);

  // the cpu point set
  Vector3f    cpuPoints[g_nSamples];

  // the kd-tree point set
  KdTree      cpuKdTree;

  // generate points
  srand(g_seed);
  for (int i = 0; i < g_nSamples; i++) {
    float x[3];
    x[0] = (rand() / (float)RAND_MAX) * g_w;
    x[1] = (rand() / (float)RAND_MAX) * g_h;
    x[2] = (rand() / (float)RAND_MAX) * 2 * M_PI - M_PI;

    cpuPoints[i] << x[0], x[1], x[2];
    cudaPoints.insert(x);

    KdNode* kdNode = new KdNode();
    kdNode->setPoint(cpuPoints[i]);
    kdNode->idx = i;

    cpuKdTree.insert(kdNode);
  }

  // for timing
  Timespec start, end;

  // we'll store the sorted output here
  SortedSolns cpuSolns;

  // brute force search
  // ------------------------------------------------------------------------
  clock_gettime(CLOCK_MONOTONIC, &start);

  for (int i = 0; i < g_nSamples; i++) {
    curves_eigen::Path<float> result =
        curves_eigen::solve(q_cpu, cpuPoints[i], g_r);
    Solution soln;
    soln.idx = i;
    soln.id = result.id;
    soln.dist = result.dist(g_r);

    cpuSolns.insert(soln);
    if (cpuSolns.size() > g_k) {
      SortedSolns::iterator iErase = cpuSolns.end();
      --iErase;
      cpuSolns.erase(iErase);
    }
  }

  clock_gettime(CLOCK_MONOTONIC, &end);

  double cpuTime = (end - start).milliseconds();
  double cpuThroughput = g_nSamples / cpuTime;

  // cuda brute force search
  // ------------------------------------------------------------------------
  // output buffer for cuda
  CudaResult cudaSoln(2, g_k);

  // brute force search
  clock_gettime(CLOCK_MONOTONIC, &start);

  cudaPoints.nearest_children(q_gpu, cudaSoln);

  clock_gettime(CLOCK_MONOTONIC, &end);

  double gpuTime = (end - start).milliseconds();
  double gpuThroughput = g_nSamples / gpuTime;

  // validate the results
  SortedSolns gpuSolns;

  for (int i = 0; i < g_k; i++) {
    float dist = cudaSoln(0, i);
    float out = cudaSoln(1, i);
    uint32_t pack = reinterpret_cast<uint32_t&>(out);

    Solution soln;
    soln.dist = dist;
    soln.idx = pack >> 4;
    soln.id = (pack & 0x0F);

    gpuSolns.insert(soln);
  }

  // kd-tree search
  // ------------------------------------------------------------------------
  KdKNearest kNearest(g_k, g_r);
  clock_gettime(CLOCK_MONOTONIC, &start);

  cpuKdTree.findNearest(q_cpu, kNearest);

  clock_gettime(CLOCK_MONOTONIC, &end);

  double cpuKdTime = (end - start).milliseconds();
  double cpuKdThroughput = g_nSamples / cpuKdTime;

  KdKNearest::PQueue_t& pqueue = kNearest.result();
  SortedSolns cpuKdSolns;

  while (pqueue.size() > 0) {
    Solution soln;
    soln.dist = pqueue.top().d2;
    soln.idx = pqueue.top().n->idx;
    soln.id = pqueue.top().id;

    cpuKdSolns.insert(soln);
    pqueue.pop();
  }

  std::cout << "Kd tree evaluated:\n"
            << "   " << kNearest.m_nodesEvaluated << " nodes ("
            << (100 * kNearest.m_nodesEvaluated / g_nSamples) << "%)\n"
            << "   " << kNearest.m_boxesEvaluated << " boxes\n\n";

  // result aggregation and output
  // ------------------------------------------------------------------------

  SortedSolns::iterator iCPU = cpuSolns.begin();
  SortedSolns::iterator iKD = cpuKdSolns.begin();
  SortedSolns::iterator iGPU = gpuSolns.begin();

  std::cout << "Validating solutions:\n"
            << "out_idx   cpu_idx   kd_idx    gpu_idx   cpu_dist     kd_dist      gpu_dist \n"
            << "-------   -------   -------   -------   ----------   ----------   ----------\n";

  for (int i = 0; i < g_k; i++) {
    if (iCPU == cpuSolns.end()) break;
    if (iGPU == gpuSolns.end()) break;
    if (iKD == cpuKdSolns.end()) break;

    std::cout << fmt("%7d   ") % i
              << fmt("%7d   ") % iCPU->idx
              << fmt("%7d   ") % iKD->idx
              << fmt("%7d   ") % iGPU->idx
              << fmt("%10.4f   ") % iCPU->dist
              << fmt("%10.4f   ") % iKD->dist
              << fmt("%10.4f   ") % iGPU->dist
              << "\n";
    ++iCPU;
    ++iGPU;
    ++iKD;
  }

  std::cout
  <<  "\n\nResults:\n"
      "              time (ms)      throughput           speedup   \n"
      "                             (1000 pairs/sec)     \n"
      "              -----------    ------------------   --------- \n"
      "        cpu:  " << fmt( "%11.4f") % cpuTime
                       << "    "
                       << fmt( "%15.4f") % cpuThroughput
                       << "   "
                       << fmt( "%11.4f") % 1.0
                       << "\n"
  <<  "         kd:  " << fmt( "%11.4f") % cpuKdTime
                       << "    "
                       << fmt( "%15.4f") % cpuKdThroughput
                       << "   "
                       << fmt( "%11.4f") % ( cpuKdThroughput / cpuThroughput )
                       << "\n"
  <<  "        gpu:  " << fmt( "%11.4f") % gpuTime
                       << "    "
                       << fmt( "%15.4f") % gpuThroughput
                       << "   "
                       << fmt ( "%11.4f" ) % ( gpuThroughput / cpuThroughput )
                       << "\n";
  return 0;
}








