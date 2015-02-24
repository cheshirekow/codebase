/*
 *  Copyright (C) 2014 Josh Bialkowski (jbialk@mit.edu)
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
 *
 *  @date   Sept 17, 2014
 *  @author Josh Bialkowski (josh.bialkowski@gmail.com)
 *  @brief
 */

#include <iostream>
#include <set>
#include <sstream>
#include <gtest/gtest.h>
#include <mpblocks/util/set_operations.hpp>


TEST(SubsetTest, EmptySetTest) {
  std::set<int> empty_set;
  std::set<int> nonempty_set;
  for (int i = 0; i < 3; i++) {
    nonempty_set.insert(rand() % 10);
  }
  EXPECT_TRUE(set::Contains(nonempty_set.begin(), nonempty_set.end(),
                            empty_set.begin(), empty_set.end()));
  EXPECT_FALSE(set::Contains(empty_set.begin(), empty_set.end(),
                             nonempty_set.begin(), nonempty_set.end()));
}

TEST(SubsetTest, ResultsOfExamplesMatchesExpectedTest) {
  const int n_tests = 10;
  const int n_values = 20;
  const int n_subset = 5;

  // these should return true
  for (int i = 0; i < n_tests; i++) {
    std::set<int> superset;
    std::set<int> subset;

    for (int j = 0; j < n_values; j++) {
      int val = rand() % 10000;
      superset.insert(val);
      if (rand() % n_values < n_subset) {
        subset.insert(val);
      }
    }

    std::stringstream msg;
    msg << "superset: ";
    for (int v : superset) {
      msg << v << ", ";
    }
    msg << "\nsubset: ";
    for (int v : subset) {
      msg << v << ", ";
    }

    EXPECT_TRUE(set::Contains(superset.begin(), superset.end(), subset.begin(),
                              subset.end()))
        << msg.str();
  }

  // these should return false
  for (int i = 0; i < n_tests; i++) {
    std::set<int> superset;
    std::set<int> subset;

    for (int j = 0; j < n_values; j++) {
      int val = rand() % 10000;
      superset.insert(val);
      if (rand() % n_values < n_subset) {
        subset.insert(val);
      }
    }

    if (subset.size() < 1) {
      continue;
    }

    int idx_to_remove = rand() % subset.size();
    std::set<int>::iterator iter_to_remove = subset.begin();
    for (int i = 0; i < idx_to_remove; i++) {
      iter_to_remove++;
    }
    superset.erase(*iter_to_remove);

    std::stringstream msg;
    msg << "superset: ";
    for (int v : superset) {
      msg << v << ", ";
    }
    msg << "\nsubset: ";
    for (int v : subset) {
      msg << v << ", ";
    }

    EXPECT_FALSE(set::Contains(superset.begin(), superset.end(), subset.begin(),
                               subset.end()))
        << msg.str();
  }
}

TEST(SetOpsTest, ResultsOfExamplesMatchesExpectedTest) {
  std::set<int> set_a = {0,    2,    4, 5, 6,    8};
  std::set<int> set_b = {   1,    3,    5,    7, 8};
  std::set<int> only_a_expected = {0, 2, 4, 6};
  std::set<int> only_b_expected = {1, 3, 7};
  std::set<int> intersect_expected = {5, 8};

  std::set<int> only_a_computed;
  std::set<int> only_b_computed;
  std::set<int> intersect_computed;

  set::IntersectionAndDifference(
      set_a.begin(), set_a.end(), set_b.begin(), set_b.end(),
      std::inserter(only_a_computed, only_a_computed.begin()),
      std::inserter(only_b_computed, only_b_computed.begin()),
      std::inserter(intersect_computed, intersect_computed.begin()));

  EXPECT_EQ(only_a_expected, only_a_computed);
  EXPECT_EQ(only_b_expected, only_b_computed);
  EXPECT_EQ(intersect_expected, intersect_computed);
}
