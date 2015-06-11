/*
 *  Copyright (C) 2015 Josh Bialkowski (josh.bialkowski@gmail.com)
 *
 *  This file is part of clarkson93.
 *
 *  clarkson93 is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  clarkson93 is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *(
 *  You should have received a copy of the GNU General Public License
 *  along with clarkson93.  If not, see <http://www.gnu.org/licenses/>.
 */
#include <gtest/gtest.h>

#include "clarkson93/simplex_impl.h"

struct TestTraits {
  static const int kDim = 4;
  typedef double Scalar;
  typedef int PointRef;
};

TEST(SimplexTest, ConstructionTest) {
  typedef clarkson93::Simplex<TestTraits> Simplex;
  Simplex simplex(0);
  for (int vertex_id : simplex.V) {
    EXPECT_EQ(0, vertex_id);
  }
}

TEST(SimplexTest, VertexSortTest) {
  typedef clarkson93::Simplex<TestTraits> Simplex;
  Simplex simplex(0);
  simplex.V = {3, 2, 1, 4, 5};

  for (int i = 0; i < 5; i++) {
    simplex.N[i] = reinterpret_cast<Simplex*>(simplex.V[i]);
  }
  clarkson93::SortVertices(&simplex);

  int prev_id = 0;
  for (int vertex_id : simplex.V) {
    EXPECT_LT(prev_id, vertex_id);
    prev_id = vertex_id;
  }

  Simplex* prev_ptr = 0;
  for (Simplex* neighbor_ptr : simplex.N) {
    EXPECT_LT(prev_ptr, neighbor_ptr);
    prev_ptr = neighbor_ptr;
  }
}

TEST(SimplexTest, GetIndexOfTest) {
  typedef clarkson93::Simplex<TestTraits> Simplex;
  Simplex simplex(0);
  simplex.V = {3, 2, 1, 4, 5};
  clarkson93::SortVertices(&simplex);

  EXPECT_EQ(0, simplex.GetIndexOf(1));
  EXPECT_EQ(1, simplex.GetIndexOf(2));
  EXPECT_EQ(2, simplex.GetIndexOf(3));
  EXPECT_EQ(3, simplex.GetIndexOf(4));
  EXPECT_EQ(4, simplex.GetIndexOf(5));
}

TEST(SimplexTest, NeighborAcrossTest) {
  typedef clarkson93::Simplex<TestTraits> Simplex;
  Simplex simplex(0);
  simplex.V = {3, 2, 1, 4, 5};
  clarkson93::SortVertices(&simplex);

  EXPECT_EQ(nullptr, simplex.GetNeighborAcross(1));
  simplex.SetNeighborAcross(1, reinterpret_cast<Simplex*>(1));
  EXPECT_EQ(reinterpret_cast<Simplex*>(1), simplex.GetNeighborAcross(1));

  EXPECT_EQ(nullptr, simplex.GetNeighborAcross(3));
  simplex.SetNeighborAcross(3, reinterpret_cast<Simplex*>(3));
  EXPECT_EQ(reinterpret_cast<Simplex*>(3), simplex.GetNeighborAcross(3));
}

TEST(SimplexTest, VsetSplitTest) {
  typedef clarkson93::Simplex<TestTraits> Simplex;
  Simplex simplex_a(0);
  Simplex simplex_b(0);

  simplex_a.V = {3, 2, 1, 4, 5};
  simplex_b.V = {9, 7, 1, 2, 5};
  clarkson93::SortVertices(&simplex_a);
  clarkson93::SortVertices(&simplex_b);

  std::vector<int> only_in_a;
  std::vector<int> only_in_b;
  std::vector<int> in_both;

  clarkson93::VsetSplit(simplex_a, simplex_b, std::back_inserter(only_in_a),
                        std::back_inserter(only_in_b),
                        std::back_inserter(in_both));
  EXPECT_EQ(only_in_a, std::vector<int>({3, 4}));
  EXPECT_EQ(only_in_b, std::vector<int>({7, 9}));
  EXPECT_EQ(in_both, std::vector<int>({1, 2, 5}));

  in_both.clear();
  clarkson93::VsetIntersection(simplex_a, simplex_b,
                               std::back_inserter(in_both));
  EXPECT_EQ(in_both, std::vector<int>({1, 2, 5}));
}
