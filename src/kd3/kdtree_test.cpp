/*
 *  Copyright (C) 2015 Josh Bialkowski (josh.bialkowski@gmail.com)
 *
 *  This file is part of kd3.
 *
 *  kd3 is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  kd3 is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with kd3.  If not, see <http://www.gnu.org/licenses/>.
 */
/*
 *  \file   CSampler.cpp
 *
 *  \date   Mar 26, 2011
 *  \author Josh Bialkowski (jbialk@mit.edu)
 *  \brief
 */


#include <mpblocks/nearest_neighbor/kd_tree.hpp>
#include <mpblocks/nearest_neighbor/kd_tree/blocks.h>

#include "kdtree.h"
#include <cmath>
#include <iostream>
#include <vector>
#include <cstdio>
#include <cstdlib>

#define FMT_IS_FLOAT

#ifdef FMT_IS_FLOAT
#define TFMT float
#else
#define TFMT double
#endif

#define NDIM 3
#define NTEST 10
#define NQUERY 10

namespace kdns = mpblocks::nearest_neighbor::kd_tree;

typedef kdns::Node<TFMT, NDIM, void*> Node_t;
typedef kdns::Tree<TFMT, NDIM, Node_t> Tree_t;
typedef Eigen::Matrix<TFMT, NDIM, 1> Point_t;

struct Dummy {
  int dummy;
  Dummy(int idx) : dummy(idx) {}
};

int main() {
  using std::cout;
  using std::endl;

  srand(time(NULL));
  // srand(0);

  Tree_t tree;
  kdtree* kd_tree = kd_create(NDIM);

  void* data = 0;
  int idx = 0;
  TFMT point[NDIM];
  Point_t ePoint;

  for (unsigned int i = 0; i < NTEST; i++) {
    for (unsigned int j = 0; j < NDIM; j++) {
      point[j] = rand() / (TFMT)RAND_MAX;
      ePoint[j] = point[j];
    }

    data = static_cast<void*>(new Dummy(idx++));
    Node_t* node = tree.insert(ePoint);
    node->setData(data);
#ifdef FMT_IS_FLOAT
    kd_insertf(kd_tree, point, data);
#else
    kd_insert(kd_tree, point, data);
#endif
  }

  cout << "Nearest Node Test:\n-------------------\n";

  for (unsigned int i = 0; i < NTEST; i++) {
    for (unsigned int j = 0; j < NDIM; j++) {
      point[j] = rand() / (TFMT)RAND_MAX;
      ePoint[j] = point[j];
    }

    Node_t* cppResult = tree.findNearest(ePoint);
#ifdef FMT_IS_FLOAT
    kdres* kdResult = kd_nearestf(kd_tree, point);
#else
    kdres* kdResult = kd_nearest(kd_tree, point);
#endif
    void* cResult = kd_res_item_data(kdResult);
    kd_res_free(kdResult);

    if (cResult == cppResult->getData())
      printf("   %4d: passed\n", i);
    else
      printf("   %4d: failed\n", i);
  }

  cout << endl;
  cout << "Range Search Test:\n---------------------\n";

  for (unsigned int i = 0; i < NTEST; i++) {
    for (unsigned int j = 0; j < NDIM; j++) {
      point[j] = rand() / (TFMT)RAND_MAX;
      ePoint[j] = point[j];
    }

    TFMT range = rand() / (TFMT)RAND_MAX;

    Tree_t::RSearch_t::List_t& cppResult = tree.findNear(ePoint, range);
#ifdef FMT_IS_FLOAT
    kdres* kdResult = kd_nearest_rangef(kd_tree, point, range);
#else
    kdres* kdResult = kd_nearest_range(kd_tree, point, range);
#endif

    bool passed = true;
    if (kd_res_size(kdResult) != cppResult.size()) passed = false;

    unsigned int cResultSize = kd_res_size(kdResult);
    while (!kd_res_end(kdResult)) {
      void* cResult = kd_res_item_data(kdResult);

      bool found = false;
      for (unsigned int j = 0; j < cppResult.size(); j++) {
        if (cppResult[j]->getData() == cResult) {
          found = true;
          break;
        }
      }

      if (!found) {
        passed = false;
        break;
      }

      kd_res_next(kdResult);
    }

    kd_res_free(kdResult);

    if (passed)
      printf("   %4d: passed\n", i);
    else {
      printf("   %4d: failed\n", i);
      printf("          kdtree result size: %4d, this: %4d\n", cResultSize,
             (int)cppResult.size());
    }
  }

  cout << endl;
  cout << "K Nearest Search Test:\n---------------------\n";

  for (unsigned int i = 0; i < NTEST; i++) {
    for (unsigned int j = 0; j < NDIM; j++) {
      point[j] = rand() / (TFMT)RAND_MAX;
      ePoint[j] = point[j];
    }

    printf("   test %d: \n", i);

    for (unsigned int k = 2; k < NTEST; k += 2) {
      typename Tree_t::KSearch_t::PQueue_t& cppResult =
          tree.findNearest(ePoint, k);

      TFMT eps = 1e-6;
      TFMT range = std::sqrt(cppResult.rbegin()->d2) + eps;

#ifdef FMT_IS_FLOAT
      kdres* kdResult = kd_nearest_rangef(kd_tree, point, range);
#else
      kdres* kdResult = kd_nearest_range(kd_tree, point, range);
#endif

      bool passed = true;
      if (kd_res_size(kdResult) != cppResult.size()) passed = false;

      unsigned int cResultSize = kd_res_size(kdResult);
      while (!kd_res_end(kdResult)) {
        void* cResult = kd_res_item_data(kdResult);

        bool found = false;
        Tree_t::KSearch_t::PQueue_t::iterator iKey;

        for (iKey = cppResult.begin(); iKey != cppResult.end(); ++iKey) {
          if (iKey->node->getData() == cResult) {
            found = true;
            break;
          }
        }

        if (!found) {
          passed = false;
          break;
        }

        kd_res_next(kdResult);
      }

      kd_res_free(kdResult);

      if (passed)
        printf("   %4d nearest: passed\n", k);
      else {
        Tree_t::KSearch_t::PQueue_t::iterator iKey;

        printf("   %4d: failed\n", i);
        printf("          k-nearest: ");

        for (iKey = cppResult.begin(); iKey != cppResult.end(); ++iKey) {
          Dummy* dummy = static_cast<Dummy*>(iKey->node->getData());
          printf("[%0.3f, %d], ", iKey->d2, dummy->dummy);
        }
        printf("\n");
        printf("          kdtree result size: %4d, this: %4d\n", cResultSize,
               (int)cppResult.size());
      }
    }
  }

  kd_free(kd_tree);

  cout << endl;

  return 0;
}
