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
 *  @date   Aug 4, 2013
 *  @author Josh Bialkowski (jbialk@mit.edu)
 */

#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <list>
#include <map>
#include <set>
#include <vector>

#include <boost/format.hpp>
#include <gtest/gtest.h>
#include <mpblocks/btps/example_traits.h>
#include <mpblocks/btps/tree.h>
#include <mpblocks/util/timespec.h>
#include <mpblocks/util/range.hpp>

using namespace mpblocks;

typedef btps::ExampleTraits::Node Node;
typedef btps::Tree<btps::ExampleTraits> Tree;
typedef boost::format fmt;

TEST(btps_Test, RandomWeightsTest) {
  srand(0);
  Node nil(0);
  Tree tree(&nil);

  std::vector<Node> nodeStore;
  utility::Timespec start, finish;

  for (auto exp : util::Range(2u, 8u)) {
    unsigned supportSize = 2 << exp;

    std::cout << fmt("\n\nFor support size: %d\n") % supportSize;

    // clear out nil sample count in case we are accidentally sampling it
    nil.freq = 0;

    // clear out data, reserve space
    nodeStore.clear();
    nodeStore.reserve(supportSize);
    tree.clear();

    std::cout << fmt("Generating random weights:\n");
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (auto i : util::Range(0u, supportSize)) {
      nodeStore.emplace_back(rand() / (double)RAND_MAX);
      i += 0;  // prevent gcc warning
    }
    clock_gettime(CLOCK_MONOTONIC, &finish);
    std::cout << fmt("   Finished in %0.4f ms\n") %
                     (finish - start).Milliseconds();

    std::cout << fmt("Putting items in the tree:\n");
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (auto& node : nodeStore) tree.insert(&node);
    clock_gettime(CLOCK_MONOTONIC, &finish);
    std::cout << fmt("   Finished in %0.4f ms\n") %
                     (finish - start).Milliseconds();

    // perform experiment
    unsigned experimentSize = 1000u * supportSize;
    std::cout << fmt("Generating in %u samples\n") % experimentSize;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (auto s : util::Range(0u, experimentSize)) {
      tree.findInterval(rand() / (double)RAND_MAX)->freq++;
      s += 0;  // prevent gcc warning
    }
    clock_gettime(CLOCK_MONOTONIC, &finish);
    double ms = (finish - start).Milliseconds();
    std::cout << fmt("   Finished in %0.4f ms, %0.2f samples/seg\n") % ms %
                     (1000 * experimentSize / ms);

    // gather statistics
    double totalWeight = tree.sum();
    double sq_error_sum = 0;
    for (auto i : util::Range(0u, supportSize)) {
      double expected = nodeStore[i].weight / totalWeight;
      double actual = nodeStore[i].freq / (double)experimentSize;
      double error = actual - expected;
      sq_error_sum += error * error;
    }
    double rms_error = std::sqrt(sq_error_sum/supportSize);
    EXPECT_LT(rms_error/totalWeight, std::pow(10,-exp));

    // check on the tree depth
    std::cout << fmt("Computing depth counts, expected depth: %d \n") % exp;
    std::map<unsigned, std::list<Node*> > depthMap;
    depthMap[0].push_back(tree.root());
    for (unsigned i : util::Range(1u, 2u * exp)) {
      if (depthMap[i - 1].size() < 1) break;
      for (Node* node : depthMap[i - 1]) {
        if (node->left != &nil) depthMap[i].push_back(node->left);
        if (node->right != &nil) depthMap[i].push_back(node->right);
      }
    }

    int max_depth;
    for (auto& pair : depthMap) {
      int size = pair.second.size();
      if(size > 0) {
        max_depth = pair.first;
      }
      std::cout << fmt(" %4u : %4u \n") % pair.first % size;
    }
    EXPECT_LE(max_depth, exp+1); // allow one depth higher

    std::cout << "Testing removal:\n ";
    unsigned int mod = supportSize >> 2;
    for (std::size_t i = 0; i < supportSize; i += 1 + (rand() % mod)) {
      tree.remove(&nodeStore[i]);
    }
  }
}
