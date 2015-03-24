#include <iostream>
#include <set>
#include <sstream>
#include <gtest/gtest.h>

#include <edelsbrunner96/edelsbrunner96.hpp>
#include "char_label_triangulation.h"

class LineWalkerTest : public CharLabelTriangulationTest{};

/**
 *  Ascii art for this test
 *  @verbatim
      \
       \
        \
         \
          \
   d=(0,2)|\\.
          | \   \.       E
          |  \     \.
     F    |   \       \.
          |    \  B      \.
         b=(1,1)\___________\ ________________________
          |     /          ./ c=(2,1)
          | C  /  A     ./
          |   /      ./
          |  /    ./
          | /  ./       D
   a=(0,0)|//
          /
         /
        /
       /
      /

@endverbatim
 */
TEST_F(LineWalkerTest, LineWalkReachesExpected2d) {
  typedef Traits::Simplex Simplex;
  typedef Traits::SimplexRef SimplexRef;
  typedef Traits::Point Point;
  typedef Traits::PointRef PointRef;

  // vertices
  storage_.points['a'] = Point(0, 0);
  storage_.points['b'] = Point(1, 1);
  storage_.points['c'] = Point(2, 1);
  storage_.points['d'] = Point(0, 2);

  // setup simplices and neighbor lists
  SetupSimplex('A', {'a', 'b', 'c'}, {'B', 'D', 'C'});
  SetupSimplex('B', {'b', 'c', 'd'}, {'E', 'C', 'A'});
  SetupSimplex('C', {'a', 'b', 'd'}, {'B', 'F', 'D'});
  SetupSimplex('D', {'\0', 'a', 'c'}, {'A', 'E', 'F'});
  SetupSimplex('E', {'\0', 'c', 'd'}, {'B', 'F', 'D'});
  SetupSimplex('F', {'\0', 'a', 'd'}, {'C', 'E', 'D'});

  EXPECT_EQ('B',
            edelsbrunner96::LineWalk<Traits>(storage_, 'A', Point(1.5, 1.1)));
  EXPECT_EQ('E',
            edelsbrunner96::LineWalk<Traits>(storage_, 'A', Point(1.5, 2.0)));
  EXPECT_EQ('F',
            edelsbrunner96::LineWalk<Traits>(storage_, 'A', Point(-1.0, 0.0)));
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
          | \   \.       E
          |  \     \.
     F    |   \       \.
          |    \  B      \.
         b=(1,1)\___________\ ________________________
          |     /          ./ c=(2,1)
          | C  /  A     ./
          |   /      ./
          |  /    ./
          | /  ./       D
   a=(0,0)|//
          /
         /
        /
       /
      /

@endverbatim
 */
class FuzzyWalkTest : public CharLabelTriangulationTest{
 public:
  typedef Traits::Scalar Scalar;
  typedef Traits::Simplex Simplex;
  typedef Traits::SimplexRef SimplexRef;
  typedef Traits::Point Point;
  typedef Traits::PointRef PointRef;

  virtual void SetUp() {
    // vertices
    storage_.points['a'] = Point(0, 0);
    storage_.points['b'] = Point(1, 1);
    storage_.points['c'] = Point(2, 1);
    storage_.points['d'] = Point(0, 2);

    // setup simplices and neighbor lists
    SetupSimplex('A', {'a', 'b', 'c'}, {'B', 'D', 'C'});
    SetupSimplex('B', {'b', 'c', 'd'}, {'E', 'C', 'A'});
    SetupSimplex('C', {'a', 'b', 'd'}, {'B', 'F', 'A'});
    SetupSimplex('D', {'\0', 'a', 'c'}, {'A', 'E', 'F'});
    SetupSimplex('E', {'\0', 'c', 'd'}, {'B', 'F', 'D'});
    SetupSimplex('F', {'\0', 'a', 'd'}, {'C', 'E', 'D'});
    ASSERT_PRED_FORMAT1(AssertGraphConsistentFrom, 'A');
  }

  ::testing::AssertionResult Execute() {
    std::list<SimplexRef> walk_list;
    std::list<SimplexRef> result_list;
    edelsbrunner96::FuzzyWalk_<Traits>(storage_, start_at_, x_query_, epsilon_,
                                       walk_list,
                                       std::back_inserter(result_list));
    if (walk_list != expected_walk_ || expected_result_ != result_list) {
      Point x_proj;
      std::vector<PointRef> V_feature;
      std::stringstream msg;
      msg << "\nDistance from : ";
      for(SimplexRef s_ref : walk_list) {
        msg << "\n   " << s_ref << " : "
            << edelsbrunner96::SimplexDistance<Traits>(
                   storage_, x_query_, s_ref, &x_proj,
                   std::back_inserter(V_feature));
      }
      return ::testing::AssertionFailure()
             << "For test query " << x_query_.transpose()
             << "\n   starting from: " << start_at_
             << "\n    with epsilon: " << epsilon_
             << "\n   expected walk: " << FormatSet(expected_walk_)
             << "\n     actual walk: " << FormatSet(walk_list)
             << "\n expected result: " << FormatSet(expected_result_)
             << "\n   actual result: " << FormatSet(result_list)
             << msg.str();
    } else {
      return ::testing::AssertionSuccess();
    }
  }

 protected:
  Point x_query_;
  SimplexRef start_at_;
  Scalar epsilon_;
  std::list<SimplexRef> expected_walk_;
  std::list<SimplexRef> expected_result_;
};


TEST_F(FuzzyWalkTest, q0) {
  x_query_ = {1.5, 1.1};
  start_at_ = 'A';
  epsilon_ = 1e-9;
  expected_walk_ = {'A', 'B', 'E', 'C'};
  expected_result_ = {'B'};
  EXPECT_TRUE(Execute());
}

TEST_F(FuzzyWalkTest, q1) {
  x_query_ = {1.5, 2.0};
  start_at_ = 'A';
  epsilon_ = 1e-9;
  expected_walk_ = {'A', 'B', 'E', 'F', 'D'};
  expected_result_ = {'E'};
  EXPECT_TRUE(Execute());
}

TEST_F(FuzzyWalkTest, q2) {
  x_query_ = {-0.5, 0.9};
  start_at_ = 'A';
  epsilon_ = 1e-9;
  expected_walk_ = {'A', 'C', 'F', 'E', 'D'};
  expected_result_ = {'F'};
  EXPECT_TRUE(Execute());
}

TEST_F(FuzzyWalkTest, q3) {
  x_query_ = {-1, 0.0};
  start_at_ = 'A';
  epsilon_ = 1e-9;
  expected_walk_ = {'A', 'D', 'C', 'F', 'E'};
  expected_result_ = {'F'};
  EXPECT_TRUE(Execute());
}

// Degenerate cases where query falls on an edge

TEST_F(FuzzyWalkTest, q4) {
  x_query_ = {1.5, 1.0};
  start_at_ = 'A';
  epsilon_ = 0.1;
  expected_walk_ = {'A', 'B', 'D', 'C', 'E'};
  expected_result_ = {'A', 'B'};
  EXPECT_TRUE(Execute());
}

TEST_F(FuzzyWalkTest, q5) {
  x_query_ = {1, 1.5};
  start_at_ = 'A';
  epsilon_ = 0.1;
  expected_walk_ = {'A', 'B', 'E', 'C', 'F', 'D'};
  expected_result_ = {'B', 'E'};
  EXPECT_TRUE(Execute());
}

