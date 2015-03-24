#include <iostream>
#include <set>
#include <sstream>
#include <gtest/gtest.h>

#include <edelsbrunner96/edelsbrunner96.hpp>
#include "char_label_triangulation.h"

/**
 *  Ascii art for the initial triangulation
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
class InsertTest : public CharLabelTriangulationTest {
  virtual void SetUp() {
    typedef Traits::Simplex Simplex;
    typedef Traits::SimplexRef SimplexRef;
    typedef Traits::Point Point;
    typedef Traits::PointRef PointRef;

    // vertex buffer.
    storage_.points['a'] = Point(0, 0);
    storage_.points['b'] = Point(2, 1);
    storage_.points['c'] = Point(0, 2);
    storage_.points['d'] = Point(1, 1);
    storage_.points['e'] = Point(1, 1.5);
    storage_.points['f'] = Point(2 + 1e-4, 1);

    ASSERT_EQ('A',
              edelsbrunner96::Triangulate<Traits>(storage_, {'a', 'b', 'c'}));
    ASSERT_PRED_FORMAT1(AssertGraphConsistentFrom, 'A');

    ASSERT_PRED_FORMAT2(AssertNeighborhoodIs, 'A',
                        std::vector<char>({'B', 'C', 'D'}));
    ASSERT_PRED_FORMAT2(AssertNeighborhoodIs, 'B',
                        std::vector<char>({'A', 'C', 'D'}));
    ASSERT_PRED_FORMAT2(AssertNeighborhoodIs, 'C',
                        std::vector<char>({'A', 'B', 'D'}));
    ASSERT_PRED_FORMAT2(AssertNeighborhoodIs, 'D',
                        std::vector<char>({'A', 'B', 'C'}));
  }
};



TEST_F(InsertTest, InsertInterior) {
  typedef Traits::Simplex Simplex;
  typedef Traits::SimplexRef SimplexRef;
  typedef Traits::Point Point;
  typedef Traits::PointRef PointRef;

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
  edelsbrunner96::FuzzyWalkInsert<Traits>(storage_, 'A', 'd', 1e-9);
  EXPECT_PRED_FORMAT2(AssertNeighborhoodIs, 'B',
                      std::vector<char>({'C', 'D', 'E'}));
  EXPECT_PRED_FORMAT2(AssertNeighborhoodIs, 'C',
                      std::vector<char>({'B', 'D', 'F'}));
  EXPECT_PRED_FORMAT2(AssertNeighborhoodIs, 'D',
                      std::vector<char>({'B', 'C', 'G'}));
  EXPECT_PRED_FORMAT2(AssertNeighborhoodIs, 'E',
                      std::vector<char>({'B', 'F', 'G'}));
  EXPECT_PRED_FORMAT2(AssertNeighborhoodIs, 'F',
                      std::vector<char>({'C', 'E', 'G'}));
  EXPECT_PRED_FORMAT2(AssertNeighborhoodIs, 'G',
                      std::vector<char>({'D', 'E', 'F'}));
  EXPECT_PRED_FORMAT1(AssertGraphConsistentFrom, 'B');
}

TEST_F(InsertTest, InsertEdge) {
  typedef Traits::Simplex Simplex;
  typedef Traits::SimplexRef SimplexRef;
  typedef Traits::Point Point;
  typedef Traits::PointRef PointRef;

  // After insertion:
  //       \.
  //        \.
  //         \.     E         /
  //          \.             /
  //           \.           /
  //    c=(0,2)| \.        /              F
  //           |     \.   /
  //           |        \/e=(1,1.5)
  //      C    |        /  \.
  //           |  G    /      \.
  //           |      /         _\ ________________________
  //           |     /   H      ./b=(2,1)
  //           |    /        ./
  //           |   /      ./
  //           |  /    ./
  //           | /  ./     D
  //    a=(0,0)|//
  //           /
  //          /
  //         /
  //        /
  //       /
  edelsbrunner96::FuzzyWalkInsert<Traits>(storage_, 'A', 'e', 1e-9);
  EXPECT_PRED_FORMAT2(AssertNeighborhoodIs, 'C',
                      std::vector<char>({'D', 'E', 'G'}));
  EXPECT_PRED_FORMAT2(AssertNeighborhoodIs, 'D',
                      std::vector<char>({'C', 'F', 'H'}));
  EXPECT_PRED_FORMAT2(AssertNeighborhoodIs, 'E',
                      std::vector<char>({'C', 'F', 'G'}));
  EXPECT_PRED_FORMAT2(AssertNeighborhoodIs, 'F',
                      std::vector<char>({'D', 'E', 'H'}));
  EXPECT_PRED_FORMAT2(AssertNeighborhoodIs, 'G',
                      std::vector<char>({'C', 'E', 'H'}));
  EXPECT_PRED_FORMAT2(AssertNeighborhoodIs, 'H',
                        std::vector<char>({'D', 'F', 'G'}));
  EXPECT_PRED_FORMAT1(AssertGraphConsistentFrom, 'C');
}

TEST_F(InsertTest, InsertVertex) {
  typedef Traits::Simplex Simplex;
  typedef Traits::SimplexRef SimplexRef;
  typedef Traits::Point Point;
  typedef Traits::PointRef PointRef;

  // After insertion:
  //       \.
  //        \.
  //         \.
  //          \.
  //           \.
  //    c=(0,2)| \.         F
  //           |     \.
  //           |        \.
  //      C    |           \.
  //           |              \.
  //           |      E          \ ________________________
  //           |                ./f=(2.0001,1)
  //           |             ./
  //           |          ./
  //           |       ./
  //           |    ./
  //    a=(0,0)| /           G
  //           /
  //          /
  //         /
  //        /
  //       /
  edelsbrunner96::FuzzyWalkInsert<Traits>(storage_, 'A', 'f', 1e-3);
  EXPECT_PRED_FORMAT2(AssertNeighborhoodIs, 'C',
                      std::vector<char>({'E', 'F', 'G'}));
  EXPECT_PRED_FORMAT2(AssertNeighborhoodIs, 'E',
                      std::vector<char>({'C', 'F', 'G'}));
  EXPECT_PRED_FORMAT2(AssertNeighborhoodIs, 'F',
                      std::vector<char>({'C', 'E', 'G'}));
  EXPECT_PRED_FORMAT2(AssertNeighborhoodIs, 'G',
                        std::vector<char>({'C', 'E', 'F'}));
  EXPECT_PRED_FORMAT1(AssertGraphConsistentFrom, 'C');
}
