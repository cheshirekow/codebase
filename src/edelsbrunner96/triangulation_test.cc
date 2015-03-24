#include <iostream>
#include <set>
#include <sstream>
#include <gtest/gtest.h>

#include <edelsbrunner96/edelsbrunner96.hpp>
#include "char_label_triangulation.h"

class TriangulationTest : public CharLabelTriangulationTest {};


/**
 *  Ascii art for this test
 *  @verbatim
     \
      \
       \
        \
         \
   c=(0,2)|\\.
          |     \.     B
          |        \.
     C    |           \.
          |              \.
          |     A           \ ________________________
          |                ./ b=(2,1)
          |             ./
          |          ./
          |       ./
          |    ./       D
   a=(0,0)| /
          /
         /
        /
       /
      /


@endverbatim
 */
TEST_F(TriangulationTest, TriangulateTest) {
  typedef Traits::Simplex Simplex;
  typedef Traits::SimplexRef SimplexRef;
  typedef Traits::Point Point;
  typedef Traits::PointRef PointRef;

  // vertex buffer.
  storage_.points['a'] = Point(0, 0);
  storage_.points['b'] = Point(2, 1);
  storage_.points['c'] = Point(0, 2);
  storage_.points['d'] = Point(1, 1);

  ASSERT_EQ('A', edelsbrunner96::Triangulate<Traits>(storage_, {'a', 'b', 'c'}));
  ASSERT_PRED_FORMAT1(AssertGraphConsistentFrom, 'A');

  ASSERT_PRED_FORMAT2(AssertNeighborhoodIs, 'A',
                      std::vector<char>({'B', 'C', 'D'}));
  ASSERT_PRED_FORMAT2(AssertNeighborhoodIs, 'B',
                      std::vector<char>({'A', 'C', 'D'}));
  ASSERT_PRED_FORMAT2(AssertNeighborhoodIs, 'C',
                      std::vector<char>({'A', 'B', 'D'}));
  ASSERT_PRED_FORMAT2(AssertNeighborhoodIs, 'D',
                      std::vector<char>({'A', 'B', 'C'}));

  // After insertion:
  //       \.
  //        \.
  //         \.
  //          \.
  //           \.
  //    c=(0,2)|\\.
  //           | \   \.      B
  //           |  \     \.
  //      C    |   \       \.
  //           |    \  E      \.
  //          d=(1,1)\___________\ ________________________
  //           |     /          ./b=(2,1)
  //           | F  /  G     ./
  //           |   /      ./
  //           |  /    ./
  //           | /  ./     D
  //    a=(0,0)|//
  //           /
  //          /
  //         /
  //        /
  //       /
  SimplexRef arbitrary_ref =
      edelsbrunner96::InsertInside<Traits>(storage_, 'A', 'd');
  ASSERT_NE(storage_.NullSimplex(), arbitrary_ref);

  ASSERT_PRED_FORMAT2(AssertNeighborhoodIs, 'B',
                      std::vector<char>({'E', 'C', 'D'}));
  ASSERT_PRED_FORMAT2(AssertNeighborhoodIs, 'C',
                      std::vector<char>({'F', 'B', 'D'}));
  ASSERT_PRED_FORMAT2(AssertNeighborhoodIs, 'D',
                      std::vector<char>({'G', 'B', 'C'}));
  ASSERT_PRED_FORMAT2(AssertNeighborhoodIs, 'E',
                      std::vector<char>({'F', 'G', 'B'}));
  ASSERT_PRED_FORMAT2(AssertNeighborhoodIs, 'F',
                      std::vector<char>({'E', 'G', 'C'}));
  ASSERT_PRED_FORMAT2(AssertNeighborhoodIs, 'G',
                      std::vector<char>({'E', 'F', 'D'}));

  std::list<SimplexRef> all_simplices;
  edelsbrunner96::BreadthFirstSearch<Traits>(storage_, arbitrary_ref,
                                             std::back_inserter(all_simplices));

  // Verify a consistent graph
  for (SimplexRef simplex_ref : all_simplices) {
    Simplex& simplex = storage_[simplex_ref];
    for (int i = 0; i < Traits::NDim + 1; i++) {
      SimplexRef neighbor_ref = simplex.N[i];
      Simplex& neighbor_simplex = storage_[neighbor_ref];
      bool found_self_in_neighbor = false;
      for (int j = 0; j < Traits::NDim + 1; j++) {
        if (neighbor_simplex.N[j] == simplex_ref) {
          found_self_in_neighbor = true;
        }
      }
      EXPECT_TRUE(found_self_in_neighbor) << "Failed to find self "
                                          << simplex_ref << " in neighbor "
                                          << neighbor_ref;
    }
  }
}
