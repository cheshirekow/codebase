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
#include <fstream>
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
const uint  g_twoPow   = 20;
const uint  g_twoPowBF = 16;    // dont plot cpu BF after this
const uint  g_nSamples = 0x01 << g_twoPow;
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

// create the query points
float q_gpu[3 * g_nTrials];
Vector3f q_cpu[g_nTrials];

// the cpu point set
Vector3f cpuPoints[g_nSamples];

int main(int argc, char** argv) {
  srand(g_seed + 1);
  for (int i = 0; i < g_nTrials; i++) {
    float* q_g = q_gpu + 3 * i;
    q_g[0] = (rand() / (float)RAND_MAX) * g_w;
    q_g[1] = (rand() / (float)RAND_MAX) * g_h;
    q_g[2] = (rand() / (float)RAND_MAX) * 2 * M_PI - M_PI;

    q_cpu[i] << q_g[0], q_g[1], q_g[2];
  }

  // the cuda point set
  CudaPoints cudaPoints(g_nSamples, g_r);

  // the kd-tree point set
  KdTree cpuKdTree;

  // the output arrays
  std::vector<double> cpuOut, gpuOut, kdOut;
  cpuOut.resize(g_twoPow, 0);
  gpuOut.resize(g_twoPow, 0);
  kdOut.resize(g_twoPow, 0);

  for (int exp = 4; exp <= g_twoPow; exp++) {
    // clear data sets
    cudaPoints.clear(true);
    cpuKdTree.clear();

    // two to the power of
    int nSamples = 0x01 << exp;

    std::cout << "For point set size (2^" << exp << "): " << nSamples
              << std::endl;

    // generate points
    srand(g_seed);
    for (int i = 0; i < nSamples; i++) {
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
    if (exp < g_twoPowBF) {
      for (int trial = 0; trial < g_nTrials; trial++) {
        clock_gettime(CLOCK_MONOTONIC, &start);

        for (int i = 0; i < nSamples; i++) {
          curves_eigen::Path<float> result =
              curves_eigen::solve(q_cpu[trial], cpuPoints[i], g_r);
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
        cpuOut[exp] += cpuTime / g_nTrials;
      }
    }

    // cuda brute force search
    // ------------------------------------------------------------------------
    // output buffer for cuda
    CudaResult cudaSoln(2, g_k);

    // brute force search
    for (int trial = 0; trial < g_nTrials; trial++) {
      clock_gettime(CLOCK_MONOTONIC, &start);

      cudaPoints.nearest_children(q_gpu + 3 * trial, cudaSoln);

      clock_gettime(CLOCK_MONOTONIC, &end);

      double gpuTime = (end - start).milliseconds();
      gpuOut[exp] += gpuTime / g_nTrials;
    }

    // kd-tree search
    // ------------------------------------------------------------------------
    for (int trial = 0; trial < g_nTrials; trial++) {
      KdKNearest kNearest(g_k, g_r);
      clock_gettime(CLOCK_MONOTONIC, &start);

      cpuKdTree.findNearest(q_cpu[trial], kNearest);

      clock_gettime(CLOCK_MONOTONIC, &end);

      double cpuKdTime = (end - start).milliseconds();
      double cpuKdThroughput = g_nSamples / cpuKdTime;

      kdOut[exp] += cpuKdTime / g_nTrials;
    }
  }

  // the output file
  std::ofstream timeOut("./queryTime.csv");
  timeOut << "\"set_size\",\"cpu\",\"kd\",\"gpu\"\n";
  for (int exp = 4; exp <= g_twoPow; exp++) {
    timeOut << (0x01 << exp) << "," << cpuOut[exp] << "," << kdOut[exp] << ","
            << gpuOut[exp] << "\n";
  }

  return 0;
}
