#include <iostream>
#include <set>
#include <sstream>
#include <gtest/gtest.h>

#include <edelsbrunner96/edelsbrunner96.hpp>
#include "char_label_triangulation.h"

class VisibleHullTest : public CharLabelTriangulationTest{};

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
          |    \   B     \.
         b=(1,1)\___________\ ________________________
          |     /          ./ c=(2,1)
          | C  /   A    ./
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
TEST_F(VisibleHullTest, LineWalkReachesExpected2d) {
  typedef Traits::Simplex Simplex;
  typedef Traits::SimplexRef SimplexRef;
  typedef Traits::Point Point;
  typedef Traits::PointRef PointRef;

  // fill the vertex storage
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

  std::list<SimplexRef> visible_list;
  std::set<SimplexRef> visible_set;
  edelsbrunner96::GetVisibleHull<Traits>(storage_, 'D', Point(4, 1),
                                         std::back_inserter(visible_list));
  visible_set.clear();
  visible_set.insert(visible_list.begin(), visible_list.end());
  EXPECT_EQ(2, visible_list.size());
  EXPECT_NE(visible_set.end(), visible_set.find('D'));
  EXPECT_NE(visible_set.end(), visible_set.find('E'));

  visible_list.clear();
  edelsbrunner96::GetVisibleHull<Traits>(storage_, 'D', Point(1, 0),
                                         std::back_inserter(visible_list));
  visible_set.clear();
  visible_set.insert(visible_list.begin(), visible_list.end());
  EXPECT_EQ(1, visible_list.size());
  EXPECT_NE(visible_set.end(), visible_set.find('D'));

  visible_list.clear();
  edelsbrunner96::GetVisibleHull<Traits>(storage_, 'F', Point(0, -1),
                                         std::back_inserter(visible_list));
  visible_set.clear();
  visible_set.insert(visible_list.begin(), visible_list.end());
  EXPECT_EQ(2, visible_list.size());
  EXPECT_NE(visible_set.end(), visible_set.find('D'));
  EXPECT_NE(visible_set.end(), visible_set.find('F'));
}

