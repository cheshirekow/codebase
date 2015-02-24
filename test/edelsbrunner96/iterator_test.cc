#include <iostream>
#include <set>
#include <sstream>
#include <gtest/gtest.h>

#include <edelsbrunner96.hpp>
#include "char_label_triangulation.h"

/*
       \.
        \.
         \.
          \.
           \.
    c=(0,2)|\\.
           | \   \.      D
           |  \     \.
      E    |   \       \.
           |    \  A      \.
          d=(1,1)\___________\ ________________________
           |     /          ./b=(2,1)
           | B  /  C     ./
           |   /      ./
           |  /    ./
           | /  ./     F
    a=(0,0)|//
           /
          /
         /
        /
       /
 */
class IteratorTest : public CharLabelTriangulationTest {
  virtual void SetUp() {
    typedef Traits::Simplex Simplex;
    typedef Traits::SimplexRef SimplexRef;
    typedef Traits::Point Point;
    typedef Traits::PointRef PointRef;

    storage_.points['a'] = Point(0, 0);
    storage_.points['b'] = Point(2, 1);
    storage_.points['c'] = Point(0, 2);
    storage_.points['d'] = Point(1, 1);

    SetupSimplex('A', {'b', 'c', 'd'}, {'B', 'C', 'D'});
    SetupSimplex('B', {'a', 'c', 'd'}, {'A', 'C', 'E'});
    SetupSimplex('C', {'a', 'b', 'd'}, {'A', 'B', 'F'});
    SetupSimplex('D', {'\0', 'b', 'c'}, {'A', 'E', 'F'});
    SetupSimplex('E', {'\0', 'a', 'c'}, {'B', 'D', 'F'});
    SetupSimplex('F', {'\0', 'a', 'b'}, {'C', 'D', 'E'});
  }
};

TEST_F(IteratorTest, BreadthFirstTest) {
  std::list<SimplexRef> bfs_walk;
  for (SimplexRef s_ref : edelsbrunner96::BreadthFirst<Traits>(storage_, 'A')) {
    bfs_walk.push_back(s_ref);
  }

  EXPECT_PRED_FORMAT2(AssertCharSetEquals, bfs_walk,
                      std::vector<char>({'A', 'B', 'C', 'D', 'E', 'F'}));
}
