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

#include <edelsbrunner96/edelsbrunner96.hpp>
#include "char_label_triangulation.h"

class InducedSubcomplexTest  : public CharLabelTriangulationTest {};

/**
 *  Ascii art for this test
 *  @verbatim
 *  lowercase: vertices
 *  uppercase: simplices
 *
 *
      \      F     /
       \          /
 c=(0,1)\ ______ /d=(1,1)
         |     /|
         |    / |
   B     |C  /  |     E
         |  /  D|
         | /    |
  a=(0,0)|/_____|b=(1,0)
        /        \
       /          \
      /     A      \
@endverbatim
 */
TEST_F(InducedSubcomplexTest, Convex2dTest) {
  typedef Traits::Simplex Simplex;
  typedef Traits::SimplexRef SimplexRef;
  typedef Traits::Point Point;
  typedef Traits::PointRef PointRef;

  // fill vertex storage
  storage_.points['a'] = Point(0,0);
  storage_.points['b'] = Point(1,0);
  storage_.points['c'] = Point(0,1);
  storage_.points['d'] = Point(1,1);

  // setup simplices and neighbor lists
  SetupSimplex('A', {'\0', 'a', 'b'}, {'D', 'E', 'B'});
  SetupSimplex('B', {'\0', 'a', 'c'}, {'C', 'F', 'A'});
  SetupSimplex('C', {'a', 'c', 'd'}, {'F', 'D', 'B'});
  SetupSimplex('D', {'a', 'b', 'd'}, {'E', 'C', 'A'});
  SetupSimplex('E', {'\0', 'b', 'd'}, {'D', 'F', 'A'});
  SetupSimplex('F', {'\0', 'c', 'd'}, {'C', 'E', 'B'});
  ASSERT_PRED_FORMAT1(AssertGraphConsistentFrom, 'A');

  edelsbrunner96::InducedSubcomplex<Traits> induced_subcomplex;
  induced_subcomplex.Init(storage_, 'C', 'D');
  induced_subcomplex.Build(storage_);
  EXPECT_EQ('a', induced_subcomplex.V[0]);
  EXPECT_EQ('b', induced_subcomplex.V[1]);
  EXPECT_EQ('c', induced_subcomplex.V[2]);
  EXPECT_EQ('d', induced_subcomplex.V[3]);

  EXPECT_EQ('a', induced_subcomplex.f[0]);
  EXPECT_EQ('b', induced_subcomplex.e[0]);
  EXPECT_EQ('c', induced_subcomplex.e[1]);
  EXPECT_EQ('d', induced_subcomplex.f[1]);

  EXPECT_EQ('\0', induced_subcomplex.S[0]);
  EXPECT_EQ('C', induced_subcomplex.S[1]);
  EXPECT_EQ('D', induced_subcomplex.S[2]);
  EXPECT_EQ('\0', induced_subcomplex.S[3]);
  ASSERT_TRUE(AssertMarksCleared());

// After the flip we should see
//        \            /
//         \    F     /
//   c=(0,1)\ ______ /d=(1,1)
//           |\     |
//           | \    |
//  B        |  \  H|   E
//           |G  \  |
//           |    \ |
//    a=(0,0)|_____\|b=(1,0)
//          /        \.
//         /    A     \.
//        /            \.
  ASSERT_TRUE(induced_subcomplex.IsFlippable(storage_));
  induced_subcomplex.Flip(storage_);
  std::vector<std::vector<PointRef>> expected_simplices = {
      { 'a', 'b', 'c' },
      { 'b', 'c', 'd' },
  };

  // verify that the two simplices we would expect to see are in the new
  // triangulation
  uint8_t n_simplices = 0;
  SimplexRef new_simplices[2] = { '\0', '\0' };
  for( SimplexRef s_ref : induced_subcomplex.S) {
    if (s_ref == storage_.NullSimplex() ) {
      continue;
    }
    n_simplices++;
    std::vector<PointRef> vertex_set(storage_[s_ref].V.begin(),
                                     storage_[s_ref].V.end());

    // try to match the new simplex with one of the ones we expect
    bool matched_simplex = false;
    for (int i = 0; i < 2; i++) {
      auto& expected_set = expected_simplices[i];
      if (vertex_set == expected_set) {
        new_simplices[i] = s_ref;
        matched_simplex = true;
      }
    }
    EXPECT_TRUE(matched_simplex) << "For simplex consisting of vertices "
                                 << vertex_set[0] << ", "
                                 << vertex_set[1] << ", "
                                 << vertex_set[2];
  }
  // There should only be two new simplices
  ASSERT_EQ(2, n_simplices);

  // Verify the neighborhood of new simplices
  ASSERT_PRED_FORMAT2(AssertNeighborhoodIs, new_simplices[0],
                      std::vector<char>({new_simplices[1], 'B', 'A'}));
  ASSERT_PRED_FORMAT2(AssertNeighborhoodIs, new_simplices[1],
                      std::vector<char>({'F', 'E', new_simplices[0]}));

  // Verify neighborhood of edge simplices
  ASSERT_PRED_FORMAT2(AssertNeighborhoodIs, 'A',
                      std::vector<char>({new_simplices[0], 'E', 'B'}));
  ASSERT_PRED_FORMAT2(AssertNeighborhoodIs, 'B',
                      std::vector<char>({new_simplices[0], 'F', 'A'}));
  ASSERT_PRED_FORMAT2(AssertNeighborhoodIs, 'E',
                      std::vector<char>({new_simplices[1], 'F', 'A'}));
  ASSERT_PRED_FORMAT2(AssertNeighborhoodIs, 'F',
                      std::vector<char>({new_simplices[1], 'E', 'B'}));
}

/**
 *  Ascii art for this test
 *  @verbatim

      \
       \
        \
         \
          \
   d=(0,2)|\\.
          | \   \.      E
          |  \     \.
     F    |   \       \.
          |    \  B      \.
         b=(1,1)\___________\ ________________________
          |     /          ./c=(2,1)
          | C  /  A     ./
          |   /      ./
          |  /    ./
          | /  ./     D
   a=(0,0)|//
          /
         /
        /
       /
      /


@endverbatim
 */
TEST_F(InducedSubcomplexTest, Reflex2dTest) {
  typedef Traits::Simplex Simplex;
  typedef Traits::SimplexRef SimplexRef;
  typedef Traits::Point Point;
  typedef Traits::PointRef PointRef;

  // fill vertex storage
  storage_.points['a'] = Point(0,0);
  storage_.points['b'] = Point(1,1);
  storage_.points['c'] = Point(2,1);
  storage_.points['d'] = Point(0,2);

  // setup simplices and neighbor lists
  SetupSimplex('A', {'a', 'b', 'c'}, {'B', 'D', 'C'});
  SetupSimplex('B', {'b', 'c', 'd'}, {'E', 'C', 'A'});
  SetupSimplex('C', {'a', 'b', 'd'}, {'B', 'F', 'A'});
  SetupSimplex('D', {'\0', 'a', 'c'}, {'A', 'E', 'F'});
  SetupSimplex('E', {'\0', 'c', 'd'}, {'B', 'F', 'D'});
  SetupSimplex('F', {'\0', 'a', 'd'}, {'C', 'E', 'D'});
  ASSERT_PRED_FORMAT1(AssertGraphConsistentFrom, 'A');

  for(auto& pair: storage_.simplices) {
    pair.second.ComputeCenter(storage_);
  }
  edelsbrunner96::InducedSubcomplex<Traits> induced_subcomplex;
  induced_subcomplex.Init(storage_, 'A', 'B');
  EXPECT_TRUE(induced_subcomplex.IsLocallyRegular(storage_));
  induced_subcomplex.Build(storage_);
  EXPECT_TRUE(induced_subcomplex.IsFlippable(storage_));

  EXPECT_EQ('a', induced_subcomplex.V[0]);
  EXPECT_EQ('b', induced_subcomplex.V[1]);
  EXPECT_EQ('c', induced_subcomplex.V[2]);
  EXPECT_EQ('d', induced_subcomplex.V[3]);

  EXPECT_EQ('a', induced_subcomplex.e[1]);
  EXPECT_EQ('b', induced_subcomplex.f[0]);
  EXPECT_EQ('c', induced_subcomplex.f[1]);
  EXPECT_EQ('d', induced_subcomplex.e[0]);

  EXPECT_EQ('B', induced_subcomplex.S[0]);
  EXPECT_EQ('\0', induced_subcomplex.S[1]);
  EXPECT_EQ('C', induced_subcomplex.S[2]);
  EXPECT_EQ('A', induced_subcomplex.S[3]);

  ASSERT_TRUE(AssertMarksCleared());

  // Here we test the same facet as we tested above, but we test it with the
  // order of the simplices reversed. The only difference in the expected
  // outcomes is that e[0] and e[1] are flipped.
  induced_subcomplex.Init(storage_, 'B', 'A');
  EXPECT_TRUE(induced_subcomplex.IsLocallyRegular(storage_));
  induced_subcomplex.Build(storage_);
  EXPECT_TRUE(induced_subcomplex.IsFlippable(storage_));

  EXPECT_EQ('a', induced_subcomplex.V[0]);
  EXPECT_EQ('b', induced_subcomplex.V[1]);
  EXPECT_EQ('c', induced_subcomplex.V[2]);
  EXPECT_EQ('d', induced_subcomplex.V[3]);

  EXPECT_EQ('a', induced_subcomplex.e[0]);
  EXPECT_EQ('b', induced_subcomplex.f[0]);
  EXPECT_EQ('c', induced_subcomplex.f[1]);
  EXPECT_EQ('d', induced_subcomplex.e[1]);

  EXPECT_EQ('B', induced_subcomplex.S[0]);
  EXPECT_EQ('\0', induced_subcomplex.S[1]);
  EXPECT_EQ('C', induced_subcomplex.S[2]);
  EXPECT_EQ('A', induced_subcomplex.S[3]);

  ASSERT_TRUE(AssertMarksCleared());

  induced_subcomplex.Init(storage_, 'B', 'C');
  EXPECT_TRUE(induced_subcomplex.IsLocallyRegular(storage_));
  induced_subcomplex.Build(storage_);
  EXPECT_TRUE(induced_subcomplex.IsFlippable(storage_));

  EXPECT_EQ('a', induced_subcomplex.V[0]);
  EXPECT_EQ('b', induced_subcomplex.V[1]);
  EXPECT_EQ('c', induced_subcomplex.V[2]);
  EXPECT_EQ('d', induced_subcomplex.V[3]);

  EXPECT_EQ('a', induced_subcomplex.e[0]);
  EXPECT_EQ('b', induced_subcomplex.f[0]);
  EXPECT_EQ('d', induced_subcomplex.f[1]);
  EXPECT_EQ('c', induced_subcomplex.e[1]);

  EXPECT_EQ('B', induced_subcomplex.S[0]);
  EXPECT_EQ('\0', induced_subcomplex.S[1]);
  EXPECT_EQ('C', induced_subcomplex.S[2]);
  EXPECT_EQ('A', induced_subcomplex.S[3]);

  ASSERT_TRUE(AssertMarksCleared());

  induced_subcomplex.Init(storage_, 'C', 'B');
  EXPECT_TRUE(induced_subcomplex.IsLocallyRegular(storage_));
  induced_subcomplex.Build(storage_);
  EXPECT_TRUE(induced_subcomplex.IsFlippable(storage_));

  EXPECT_EQ('a', induced_subcomplex.V[0]);
  EXPECT_EQ('b', induced_subcomplex.V[1]);
  EXPECT_EQ('c', induced_subcomplex.V[2]);
  EXPECT_EQ('d', induced_subcomplex.V[3]);

  EXPECT_EQ('a', induced_subcomplex.e[1]);
  EXPECT_EQ('b', induced_subcomplex.f[0]);
  EXPECT_EQ('d', induced_subcomplex.f[1]);
  EXPECT_EQ('c', induced_subcomplex.e[0]);

  EXPECT_EQ('B', induced_subcomplex.S[0]);
  EXPECT_EQ('\0', induced_subcomplex.S[1]);
  EXPECT_EQ('C', induced_subcomplex.S[2]);
  EXPECT_EQ('A', induced_subcomplex.S[3]);

  ASSERT_TRUE(AssertMarksCleared());

  induced_subcomplex.Init(storage_, 'A', 'C');
  EXPECT_TRUE(induced_subcomplex.IsLocallyRegular(storage_));
  induced_subcomplex.Build(storage_);
  EXPECT_TRUE(induced_subcomplex.IsFlippable(storage_));

  EXPECT_EQ('a', induced_subcomplex.V[0]);
  EXPECT_EQ('b', induced_subcomplex.V[1]);
  EXPECT_EQ('c', induced_subcomplex.V[2]);
  EXPECT_EQ('d', induced_subcomplex.V[3]);

  EXPECT_EQ('c', induced_subcomplex.e[1]);
  EXPECT_EQ('a', induced_subcomplex.f[0]);
  EXPECT_EQ('b', induced_subcomplex.f[1]);
  EXPECT_EQ('d', induced_subcomplex.e[0]);

  EXPECT_EQ('B', induced_subcomplex.S[0]);
  EXPECT_EQ('\0', induced_subcomplex.S[1]);
  EXPECT_EQ('C', induced_subcomplex.S[2]);
  EXPECT_EQ('A', induced_subcomplex.S[3]);

  ASSERT_TRUE(AssertMarksCleared());

  induced_subcomplex.Init(storage_, 'C', 'A');
  EXPECT_TRUE(induced_subcomplex.IsLocallyRegular(storage_));
  induced_subcomplex.Build(storage_);
  EXPECT_TRUE(induced_subcomplex.IsFlippable(storage_));

  EXPECT_EQ('a', induced_subcomplex.V[0]);
  EXPECT_EQ('b', induced_subcomplex.V[1]);
  EXPECT_EQ('c', induced_subcomplex.V[2]);
  EXPECT_EQ('d', induced_subcomplex.V[3]);

  EXPECT_EQ('d', induced_subcomplex.e[1]);
  EXPECT_EQ('a', induced_subcomplex.f[0]);
  EXPECT_EQ('b', induced_subcomplex.f[1]);
  EXPECT_EQ('c', induced_subcomplex.e[0]);

  EXPECT_EQ('B', induced_subcomplex.S[0]);
  EXPECT_EQ('\0', induced_subcomplex.S[1]);
  EXPECT_EQ('C', induced_subcomplex.S[2]);
  EXPECT_EQ('A', induced_subcomplex.S[3]);

  ASSERT_TRUE(AssertMarksCleared());

// After the flip we should see:
//      \.
//       \.
//        \.
//         \.
//          \.
//    d=(0,2)| \.
//           |     \.      E
//           |        \.
//      F    |           \.
//           |              \.
//         b=(1,1) O           \ ________________________
//           |                ./ c=(2,1)
//           |       G     ./
//           |          ./
//           |       ./
//           |    ./       D
//    a=(0,0)|./
//           /
//          /
//         /
//        /
//       /
//
//

  ASSERT_TRUE(induced_subcomplex.IsFlippable(storage_));
  induced_subcomplex.Flip(storage_);
  std::vector<PointRef> expected_simplex = {
      { 'a', 'c', 'd' }
  };

  // verify that the new simplex we would expect to see is in the new
  // triangulation
  uint8_t n_simplices = 0;
  SimplexRef new_simplex = '\0';
  for( SimplexRef s_ref : induced_subcomplex.S) {
    if (s_ref == '\0') {
      continue;
    }
    n_simplices++;
    std::vector<PointRef> vertex_set(storage_[s_ref].V.begin(),
                                     storage_[s_ref].V.end());

    // try to match the new simplex with the one we expect
    bool matched_simplex = false;
    if (vertex_set == expected_simplex) {
      new_simplex = s_ref;
      matched_simplex = true;
    }
    EXPECT_TRUE(matched_simplex) << "For simplex consisting of vertices "
                                 << vertex_set[0] << ", "
                                 << vertex_set[1] << ", "
                                 << vertex_set[2];
  }
  // There should only be one new simplices
  ASSERT_EQ(1, n_simplices);

  // Verify the neighborhood
  // neighborhood of new_simplex
  ASSERT_PRED_FORMAT2(AssertNeighborhoodIs, 'G',
                      std::vector<char>({'E', 'F', 'D'}));
  ASSERT_PRED_FORMAT2(AssertNeighborhoodIs, 'D',
                      std::vector<char>({'G', 'E', 'F'}));
  ASSERT_PRED_FORMAT2(AssertNeighborhoodIs, 'E',
                      std::vector<char>({'G', 'F', 'D'}));
  ASSERT_PRED_FORMAT2(AssertNeighborhoodIs, 'F',
                      std::vector<char>({'G', 'E', 'D'}));
}


/**
 *  Ascii art for this test
 *  @verbatim
           \
            \
             \
              \
               \d=(0,2)
               ./\\.
       H     ./   \   \.       G
           ./      \     \.
         ./    C    \       \.
e=     ./            \  B      \.
(-1,1)/_______________\___________\ ________________________
      \.       b=(1,1)/          ./ c=(2,1)
        \.           /  A     ./
          \.   D    /      ./
            \.     /    ./     F
       E      \.  /  ./
                \//a=(0,0)
                /
               /
              /
             /
            /


@endverbatim
 */
TEST_F(InducedSubcomplexTest, NotFlippable2dTest) {
  typedef Traits::Simplex Simplex;
  typedef Traits::SimplexRef SimplexRef;
  typedef Traits::Point Point;
  typedef Traits::PointRef PointRef;

  // fill vertex storage
  storage_.points['a'] = Point(0,0);
  storage_.points['b'] = Point(1,1);
  storage_.points['c'] = Point(2,1);
  storage_.points['d'] = Point(0,2);
  storage_.points['e'] = Point(-1,1);

  // setup simplices and neighbor lists
  SetupSimplex('A', {'a', 'b', 'c'}, {'B', 'F', 'D'});
  SetupSimplex('B', {'b', 'c', 'd'}, {'G', 'C', 'A'});
  SetupSimplex('C', {'b', 'd', 'e'}, {'H', 'D', 'B'});
  SetupSimplex('D', {'a', 'b', 'e'}, {'C', 'E', 'A'});
  SetupSimplex('E', {'\0', 'a', 'e'}, {'D', 'H', 'F'});
  SetupSimplex('F', {'\0', 'a', 'c'}, {'A', 'G', 'E'});
  SetupSimplex('G', {'\0', 'c', 'd'}, {'B', 'H', 'F'});
  SetupSimplex('H', {'\0', 'd', 'e'}, {'C', 'E', 'G'});

  ASSERT_PRED_FORMAT1(AssertGraphConsistentFrom, 'A');

  for(auto& pair : storage_.simplices) {
    pair.second.ComputeCenter(storage_);
  }

  edelsbrunner96::InducedSubcomplex<Traits> induced_subcomplex;
  induced_subcomplex.Init(storage_, 'A', 'B');
  induced_subcomplex.Build(storage_);
  EXPECT_EQ('a', induced_subcomplex.V[0]);
  EXPECT_EQ('b', induced_subcomplex.V[1]);
  EXPECT_EQ('c', induced_subcomplex.V[2]);
  EXPECT_EQ('d', induced_subcomplex.V[3]);

  EXPECT_EQ('a', induced_subcomplex.e[1]);
  EXPECT_EQ('b', induced_subcomplex.f[0]);
  EXPECT_EQ('c', induced_subcomplex.f[1]);
  EXPECT_EQ('d', induced_subcomplex.e[0]);

  EXPECT_EQ('B', induced_subcomplex.S[0]);
  EXPECT_EQ('\0', induced_subcomplex.S[1]);
  EXPECT_EQ('\0',  induced_subcomplex.S[2]);
  EXPECT_EQ('A',  induced_subcomplex.S[3]);

  ASSERT_TRUE(AssertMarksCleared());

  ASSERT_TRUE(induced_subcomplex.IsLocallyRegular(storage_));
  ASSERT_FALSE(induced_subcomplex.IsFlippable(storage_));

  induced_subcomplex.Init(storage_, 'C', 'D');
  induced_subcomplex.Build(storage_);
  EXPECT_EQ('C', induced_subcomplex.s[0]);
  EXPECT_EQ('D', induced_subcomplex.s[1]);

  EXPECT_EQ('a', induced_subcomplex.V[0]);
  EXPECT_EQ('b', induced_subcomplex.V[1]);
  EXPECT_EQ('d', induced_subcomplex.V[2]);
  EXPECT_EQ('e', induced_subcomplex.V[3]);

  EXPECT_EQ('a', induced_subcomplex.e[0]);
  EXPECT_EQ('b', induced_subcomplex.f[0]);
  EXPECT_EQ('e', induced_subcomplex.f[1]);
  EXPECT_EQ('d', induced_subcomplex.e[1]);

  EXPECT_EQ('C', induced_subcomplex.S[0]);
  EXPECT_EQ('\0', induced_subcomplex.S[1]);
  EXPECT_EQ('D', induced_subcomplex.S[2]);
  EXPECT_EQ('\0', induced_subcomplex.S[3]);

  ASSERT_TRUE(AssertMarksCleared());

  ASSERT_TRUE(induced_subcomplex.IsFlippable(storage_));
  ASSERT_TRUE(induced_subcomplex.IsFlippable(storage_));
}

