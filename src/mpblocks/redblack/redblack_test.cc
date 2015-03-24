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
#include <set>
#include <vector>

#include <boost/format.hpp>
#include <gtest/gtest.h>
#include <mpblocks/redblack.h>
#include <mpblocks/util/timespec.h>

template <typename T>
struct Range {
  struct Iterator {
    T val;  ///< storage for the actual value

    Iterator(T val) : val(val) {}
    T operator*() { return val; }
    bool operator!=(T other) { return val != other; }
    Iterator& operator++() {
      ++val;
      return *this;
    }
    operator T() { return val; }
  };

 private:
  T m_begin;  ///< the first integral value
  T m_end;    ///< one past the last integral value

 public:
  Range(T begin, T end) : m_begin(begin), m_end(end) {}

  T size() { return m_end - m_begin; }

  Iterator begin() { return m_begin; }
  Iterator end() { return m_end; }
};

template <typename T>
Range<T> range(T begin, T end) {
  return Range<T>(begin, end);
}

template <typename C>
void applyToAll(const C& call) {}

template <typename C, typename Head, typename... Tail>
void applyToAll(const C& call, Head* h, Tail... tail) {
  call(h);
  applyToAll(call, tail...);
}

struct Clear {
  template <typename T>
  void operator()(T& list) const {
    list->clear();
  }
};

struct Reserve {
  size_t size;
  Reserve(size_t size) : size(size) {}

  template <typename T>
  void operator()(T& list) const {
    list->reserve(size);
  }
};

TEST(RedBlackTest, CompareWithStdSet) {
  using mpblocks::redblack::ExampleTraits;
  using mpblocks::utility::Timespec;
  typedef mpblocks::redblack::Tree<ExampleTraits> Tree;
  typedef ExampleTraits::Node Node;
  typedef boost::format fmt;

  Node Nil(0, 0);
  Tree rbTree(&Nil);

  std::vector<double> randomValues;
  std::vector<double> setSortedValues;
  std::vector<double> rbSortedValues;

  std::vector<Node> nodeStore;
  std::multiset<double> stdTree;

  Timespec start, finish;

  for (int exp : range(8, 16)) {
    int numValues = 2 << exp;

    std::cout << fmt("\n\nFor a value set of %d values:\n") % numValues;
    std::cout << "Reserving storage for experiment\n";
    applyToAll(Reserve(numValues), &randomValues, &nodeStore, &setSortedValues,
               &rbSortedValues);
    std::cout << "   ...Done\n";

    std::cout << "Clearing structures\n";
    applyToAll(Clear(), &randomValues, &nodeStore, &setSortedValues,
               &rbSortedValues, &stdTree, &rbTree);
    std::cout << "   ...Done\n";

    std::cout << fmt("Genrating %d random values\n") % numValues;
    for (int i = 0; i < numValues; i++)
      randomValues.push_back(rand() / (double)RAND_MAX);
    std::cout << "   ...Done\n";

    std::cout << "inserting into std::set\n";
    clock_gettime(CLOCK_MONOTONIC, &start);

    stdTree.insert(randomValues.begin(), randomValues.end());

    clock_gettime(CLOCK_MONOTONIC, &finish);
    std::cout << fmt("   ...done %d values in %0.4f ms\n") % stdTree.size() %
                     (finish - start).Milliseconds();

    std::cout << "inserting into mpblocks::redblack::Tree\n";
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (double val : randomValues) {
      nodeStore.emplace_back(val, &Nil);
      rbTree.insert(&nodeStore.back());
    }

    clock_gettime(CLOCK_MONOTONIC, &finish);
    std::cout << fmt("   ...done %d values in %0.4f ms\n") % rbTree.size() %
                     (finish - start).Milliseconds();

    std::cout << "copying out of std::multiset\n";
    clock_gettime(CLOCK_MONOTONIC, &start);
    std::copy(stdTree.begin(), stdTree.end(),
              std::back_inserter(setSortedValues));
    clock_gettime(CLOCK_MONOTONIC, &finish);
    std::cout << fmt("   ...done in %0.4f ms\n") %
                     (finish - start).Milliseconds();

    std::cout << "copying out of rbTree\n";
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (double val : rbTree) rbSortedValues.emplace_back(val);
    clock_gettime(CLOCK_MONOTONIC, &finish);
    std::cout << fmt("   ...done in %0.4f ms\n") %
                     (finish - start).Milliseconds();

    std::cout << "\nValidating results\n";
    bool sortedOrder = true;
    bool matchesSet = true;
    for (int i : range(size_t(1), rbSortedValues.size())) {
      if (rbSortedValues[i] < rbSortedValues[i - 1]) {
        sortedOrder = false;
      }
      if (rbSortedValues[i] != setSortedValues[i]) {
        matchesSet = false;
      }
    }
    EXPECT_TRUE(sortedOrder);
    EXPECT_TRUE(matchesSet);
    std::cout << "   ..done\n";
  }
}
