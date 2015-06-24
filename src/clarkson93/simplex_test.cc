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
#include <Eigen/Dense>
#include <gtest/gtest.h>

#include "clarkson93/simplex_impl.h"

struct TestTraits {
  static const int kDim = 4;
  typedef double Scalar;
};

typedef Eigen::Matrix<TestTraits::Scalar, TestTraits::kDim, 1> Point;
typedef clarkson93::Simplex<TestTraits> Simplex;

static Point* const kNullPoint = nullptr;
static Simplex* const kNullSimplex = nullptr;

TEST(SimplexTest, ConstructionTest) {
  Simplex simplex;
  for (auto vertex_id : simplex.V) {
    EXPECT_EQ(nullptr, vertex_id);
  }
}

TEST(SimplexTest, VertexSortTest) {
  Simplex simplex;
  simplex.V = {kNullPoint + 3, kNullPoint + 2, kNullPoint + 1, kNullPoint + 4,
               kNullPoint + 5};

  for (int i = 0; i < 5; i++) {
    simplex.N[i] = reinterpret_cast<Simplex*>(simplex.V[i]);
  }
  clarkson93::SortVertices(&simplex);

  Point* prev_id = 0;
  for (Point* vertex_id : simplex.V) {
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
  Simplex simplex;
  simplex.V = {kNullPoint + 3, kNullPoint + 2, kNullPoint + 1, kNullPoint + 4,
               kNullPoint + 5};
  clarkson93::SortVertices(&simplex);

  EXPECT_EQ(0, simplex.GetIndexOf(kNullPoint + 1));
  EXPECT_EQ(1, simplex.GetIndexOf(kNullPoint + 2));
  EXPECT_EQ(2, simplex.GetIndexOf(kNullPoint + 3));
  EXPECT_EQ(3, simplex.GetIndexOf(kNullPoint + 4));
  EXPECT_EQ(4, simplex.GetIndexOf(kNullPoint + 5));
}

TEST(SimplexTest, NeighborAcrossTest) {
  Simplex simplex;
  simplex.V = {kNullPoint + 3, kNullPoint + 2, kNullPoint + 1, kNullPoint + 4,
               kNullPoint + 5};
  clarkson93::SortVertices(&simplex);

  EXPECT_EQ(nullptr, simplex.GetNeighborAcross(kNullPoint + 1));
  simplex.SetNeighborAcross(kNullPoint + 1, kNullSimplex + 1);
  EXPECT_EQ(kNullSimplex + 1, simplex.GetNeighborAcross(kNullPoint + 1));

  EXPECT_EQ(nullptr, simplex.GetNeighborAcross(kNullPoint + 3));
  simplex.SetNeighborAcross(kNullPoint + 3, kNullSimplex + 3);
  EXPECT_EQ(kNullSimplex + 3, simplex.GetNeighborAcross(kNullPoint + 3));
}

TEST(SimplexTest, VsetSplitTest) {
  Simplex simplex_a;
  Simplex simplex_b;

  simplex_a.V = {kNullPoint + 3, kNullPoint + 2, kNullPoint + 1, kNullPoint + 4,
                 kNullPoint + 5};
  simplex_b.V = {kNullPoint + 9, kNullPoint + 7, kNullPoint + 1, kNullPoint + 2,
                 kNullPoint + 5};
  clarkson93::SortVertices(&simplex_a);
  clarkson93::SortVertices(&simplex_b);

  std::vector<Point*> only_in_a;
  std::vector<Point*> only_in_b;
  std::vector<Point*> in_both;

  clarkson93::VsetSplit(simplex_a, simplex_b, std::back_inserter(only_in_a),
                        std::back_inserter(only_in_b),
                        std::back_inserter(in_both));
  EXPECT_EQ(only_in_a, std::vector<Point*>({kNullPoint + 3, kNullPoint + 4}));
  EXPECT_EQ(only_in_b, std::vector<Point*>({kNullPoint + 7, kNullPoint + 9}));
  EXPECT_EQ(in_both, std::vector<Point*>(
                         {kNullPoint + 1, kNullPoint + 2, kNullPoint + 5}));

  in_both.clear();
  clarkson93::VsetIntersection(simplex_a, simplex_b,
                               std::back_inserter(in_both));
  EXPECT_EQ(in_both, std::vector<Point*>(
                         {kNullPoint + 1, kNullPoint + 2, kNullPoint + 5}));
}

TEST(SimplexTest, GetNeighborsSharingTest) {
  Simplex simplex;
  simplex.V = {kNullPoint + 3, kNullPoint + 2, kNullPoint + 1, kNullPoint + 4,
               kNullPoint + 5};
  for (int i = 0; i < 5; i++) {
    simplex.N[i] = kNullSimplex + (simplex.V[i] - kNullPoint);
  }
  clarkson93::SortVertices(&simplex);

  std::vector<Simplex*> neighbors;
  std::vector<Simplex*> expected_neighbors;
  clarkson93::GetNeighborsSharing(
      simplex,
      std::vector<Point*>({kNullPoint + 1, kNullPoint + 4, kNullPoint + 5}),
      std::back_inserter(neighbors));

  expected_neighbors = {kNullSimplex + 2, kNullSimplex + 3};
  EXPECT_EQ(expected_neighbors, neighbors);

  neighbors.clear();
  clarkson93::GetNeighborsSharing(
      simplex, std::vector<Point*>({kNullPoint + 3, kNullPoint + 5}),
      std::back_inserter(neighbors));
  expected_neighbors = {kNullSimplex + 1, kNullSimplex + 2, kNullSimplex + 4};
  EXPECT_EQ(expected_neighbors, neighbors);
}

TEST(SimplexTest, GeometryTest) {
  std::vector<Point> points;
  points.resize(TestTraits::kDim + 1);
  for (int i = 0; i < TestTraits::kDim; i++) {
    points[i].fill(0);
    points[i][i] = 1;
  }
  points.back() = Point::Zero();

  Simplex simplex;
  simplex.V = {points.data() + 0, points.data() + 1, points.data() + 2,
               points.data() + 3, points.data() + 4};
  simplex.i_peak = TestTraits::kDim;

  clarkson93::ComputeBase(&simplex);
  clarkson93::OrientBase(&simplex, Point::Zero());
  EXPECT_LT(0, clarkson93::NormalProjection(simplex, Point::Zero()));
}
