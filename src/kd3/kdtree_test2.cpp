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


#include <mpblocks/kd_tree.hpp>
#include <mpblocks/kd_tree/blocks.h>

#include "kdtree.h"
#include <cmath>
#include <iostream>
#include <vector>
#include <cstdio>
#include <cstdlib>

//#define TFMT    float
#define TFMT double

#define NDIM 3
#define NTEST 10
#define NQUERY 10

namespace kdns = mpblocks::kd_tree;

struct Traits {
  typedef TFMT Format_t;
  static const unsigned NDim = NDIM;

  typedef kdns::euclidean::HyperRect<Traits> HyperRect;

  class Node : public kdns::Node<Traits> {
   public:
    void* data;
  };
};

typedef Traits::Node Node_t;
typedef kdns::Tree<Traits> Tree_t;
typedef kdns::euclidean::Nearest<Traits> NNSearch_t;
typedef kdns::euclidean::KNearest<Traits> KNNSearch_t;
typedef kdns::euclidean::Ball<Traits> RangeSearch_t;
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
  std::list<Node_t> nodeStore;

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
    nodeStore.push_back(Node_t());
    Node_t* node = &(nodeStore.back());
    node->setPoint(ePoint);
    node->data = data;
    tree.insert(node);
#ifdef FMT_IS_FLOAT
    kd_insertf(kd_tree, point, data);
#else
    kd_insert(kd_tree, point, data);
#endif
  }

  cout << "Nearest Node Test:\n-------------------\n";

  NNSearch_t nnSearch;
  for (unsigned int i = 0; i < NTEST; i++) {
    for (unsigned int j = 0; j < NDIM; j++) {
      point[j] = rand() / (TFMT)RAND_MAX;
      ePoint[j] = point[j];
    }

    nnSearch.reset();
    tree.findNearest(ePoint, nnSearch);
#ifdef FMT_IS_FLOAT
    kdres* kdResult = kd_nearestf(kd_tree, point);
#else
    kdres* kdResult = kd_nearest(kd_tree, point);
#endif
    void* cResult = kd_res_item_data(kdResult);
    kd_res_free(kdResult);

    if (cResult == nnSearch.result()->data)
      printf("   %4d: passed\n", i);
    else
      printf("   %4d: failed\n", i);
  }

  cout << endl;
  cout << "Range Search Test:\n---------------------\n";

  RangeSearch_t rangeSearch;
  for (unsigned int i = 0; i < NTEST; i++) {
    for (unsigned int j = 0; j < NDIM; j++) {
      point[j] = rand() / (TFMT)RAND_MAX;
      ePoint[j] = point[j];
    }

    TFMT range = rand() / (TFMT)RAND_MAX;
    rangeSearch.reset(ePoint, range);
    tree.findRange(rangeSearch);

    const RangeSearch_t::List_t& cppResult = rangeSearch.result();
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
        if (cppResult[j]->data == cResult) {
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

  KNNSearch_t knnSearch;
  for (unsigned int i = 0; i < NTEST; i++) {
    for (unsigned int j = 0; j < NDIM; j++) {
      point[j] = rand() / (TFMT)RAND_MAX;
      ePoint[j] = point[j];
    }

    printf("   test %d: \n", i);

    for (unsigned int k = 2; k < NTEST; k += 2) {
      knnSearch.reset(k);
      tree.findNearest(ePoint, knnSearch);
      const typename KNNSearch_t::PQueue_t& cppResult = knnSearch.result();

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
        KNNSearch_t::PQueue_t::iterator iKey;

        for (iKey = cppResult.begin(); iKey != cppResult.end(); ++iKey) {
          if (iKey->n->data == cResult) {
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
        KNNSearch_t::PQueue_t::iterator iKey;

        printf("   %4d: failed\n", i);
        printf("          k-nearest: ");

        for (iKey = cppResult.begin(); iKey != cppResult.end(); ++iKey) {
          Dummy* dummy = static_cast<Dummy*>(iKey->n->data);
          printf("[%0.3f, %d], ", iKey->d2, dummy->dummy);
        }
        printf("\n");
        printf("          kdtree result size: %4d, this: %4d\n", cResultSize,
               (int)cppResult.size());
      }
    }
  }

  nodeStore.clear();
  kd_free(kd_tree);

  cout << endl;

  return 0;
}
