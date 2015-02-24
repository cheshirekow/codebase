#include <iostream>
#include <sstream>
#include <vector>
#include <boost/format.hpp>
#include <gtest/gtest.h>

#include <edelsbrunner96.hpp>
#include "char_label_triangulation.h"

class SimplexDistanceTest : public CharLabelTriangulationTest {
 public:
  typedef Traits::Simplex Simplex;
  typedef Traits::SimplexRef SimplexRef;
  typedef Traits::Point Point;
  typedef Traits::PointRef PointRef;
  typedef Traits::Scalar Scalar;

  std::string FormatFeature(const std::vector<Traits::PointRef>& V,
                            const Traits::PointRef V0) {
    std::stringstream stream;
    stream << boost::format("(%d) :[") % V.size();
    auto last = V.end();
    --last;
    for (auto iter = V.begin(); iter != last; iter++) {
      stream << boost::format("%d, ") % (storage_[*iter] - storage_[V0]);
    }
    stream << boost::format("%d") % (storage_[*last] - storage_[V0]);
    stream << "]";
    return stream.str();
  }

  ::testing::AssertionResult TestGJK() {
    Point x_proj;
    std::vector<PointRef> V_feature;
    Scalar dist = edelsbrunner96::SimplexDistance<Traits>(
        storage_, x_query_, simplex_, &x_proj, std::back_inserter(V_feature));
    Scalar proj_error = (x_proj - expected_proj_).norm();
    Scalar dist_error = std::abs(dist - expected_dist_);
    std::sort(V_feature.begin(), V_feature.end());
    if(proj_error > 1e-6 || dist_error > 1e-6 || V_feature != expected_feature_) {
      return ::testing::AssertionFailure()
        << "For GJK test query " << x_query_.transpose()
        << "\n where computed projection is " << x_proj.transpose()
        << "\n and expected projection is   " << expected_proj_.transpose()
        << "\n with computed feature : " << FormatSet(V_feature)
        << "\n and expected feature  : " << FormatSet(expected_feature_)
        << "\n computed distance     : " << dist
        << "\n and expected distance : " << expected_dist_;
    } else {
      return ::testing::AssertionSuccess();
    }
  }

  ::testing::AssertionResult TestExhaustive() {
    Scalar dist;
    Point x_proj;
    std::vector<PointRef> V_feature;
    edelsbrunner96::ExhaustiveSimplexDistance<Traits>()
        .Compute(storage_, x_query_, storage_[simplex_].V.begin(),
                 storage_[simplex_].V.end())
        .GetResult(&dist, &x_proj, std::back_inserter(V_feature));
    Scalar proj_error = (x_proj - expected_proj_).norm();
    Scalar dist_error = std::abs(dist - expected_dist_);
    std::sort(V_feature.begin(), V_feature.end());
    if (proj_error > 1e-6 || dist_error > 1e-6 ||
        V_feature != expected_feature_) {
      return ::testing::AssertionFailure()
             << "For exhaustive test query " << x_query_.transpose()
             << "\n where computed projection is " << x_proj.transpose()
             << "\n and expected projection is   " << expected_proj_.transpose()
             << "\n with computed feature : " << FormatSet(V_feature)
             << "\n and expected feature  : " << FormatSet(expected_feature_)
             << "\n computed distance     : " << dist
             << "\n and expected distance : " << expected_dist_;
    } else {
      return ::testing::AssertionSuccess();
    }
  }

 protected:
  SimplexRef simplex_;
  Point x_query_;
  Point expected_proj_;
  Scalar expected_dist_;
  std::vector<PointRef> expected_feature_;
};

/**
 *  Ascii art for this test
 *  @verbatim
 *
 *                |
 *                |
 *                |c=(0,1)
 *                |\                    lowercase: vertices
 *                | \                   upercase: simplices
 *            B   |  \   A
 *                | D \
 *                |    \
 *         a=(0,0)|_____\b=(1,0)
 *               /       \
 *              /         \
 *             /    C      \
 *            /             \
 *
 *               q5
 *                |
 *                |       q4
 *                p5     /
 *                |\    /               qi : query point
 *                | \  /                pi : projected point
 *                |  \p4
 *       x--------|   \
 *      q0      p0| q6 \
 *                |_____\
 *              p1   p2  p3
 *              /    |    \
 *             /     |     \
 *          q1/      x q2   \q3
 *
@endverbatim
 */
class SingleSimplexTest : public SimplexDistanceTest {
  virtual void SetUp() {
    typedef Traits::Simplex Simplex;
    typedef Traits::SimplexRef SimplexRef;
    typedef Traits::Point Point;
    typedef Traits::PointRef PointRef;

    // vertices
    storage_.points['a'] = Point(0, 0);
    storage_.points['b'] = Point(1, 0);
    storage_.points['c'] = Point(0, 1);

    // simplices
    SetupSimplex('A', {'\0', 'b', 'c'}, {'D', 'B', 'C'});
    SetupSimplex('B', {'\0', 'a', 'c'}, {'D', 'A', 'C'});
    SetupSimplex('C', {'\0', 'a', 'b'}, {'D', 'A', 'B'});
    SetupSimplex('D', {'a', 'b', 'c'}, {'A', 'B', 'C'});
    ASSERT_PRED_FORMAT1(AssertGraphConsistentFrom, 'A');
  }
};

TEST_F(SingleSimplexTest, q0) {
  simplex_ = 'D';
  x_query_ = {-1, 0.5};
  expected_proj_ = {0, 0.5};
  expected_dist_ = 1;
  expected_feature_ = {'a', 'c'};
  EXPECT_TRUE(TestGJK());
  EXPECT_TRUE(TestExhaustive());
}

TEST_F(SingleSimplexTest, q1) {
  simplex_ = 'D';
  x_query_ = {-1, -1};
  expected_proj_ = {0, 0};
  expected_dist_ = std::sqrt(2.0);
  expected_feature_ = {'a'};
  EXPECT_TRUE(TestGJK());
  EXPECT_TRUE(TestExhaustive());
}

TEST_F(SingleSimplexTest, q2) {
  simplex_ = 'D';
  x_query_ = {0.5, -1};
  expected_proj_ = {0.5, 0};
  expected_dist_ = 1;
  expected_feature_ = {'a','b'};
  EXPECT_TRUE(TestGJK());
  EXPECT_TRUE(TestExhaustive());
}

TEST_F(SingleSimplexTest, q3) {
  simplex_ = 'D';
  x_query_ = {2, -1};
  expected_proj_ = {1, 0};
  expected_dist_ = std::sqrt(2.0);
  expected_feature_ = {'b'};
  EXPECT_TRUE(TestGJK());
  EXPECT_TRUE(TestExhaustive());
}

TEST_F(SingleSimplexTest, q4) {
  simplex_ = 'D';
  x_query_ = {1.5, 1.5};
  expected_proj_ = {0.5, 0.5};
  expected_dist_ = std::sqrt(2.0);
  expected_feature_ = {'b', 'c'};
  EXPECT_TRUE(TestGJK());
  EXPECT_TRUE(TestExhaustive());
}

TEST_F(SingleSimplexTest, q5) {
  simplex_ = 'D';
  x_query_ = {0, 2};
  expected_proj_ = {0, 1};
  expected_dist_ = 1;
  expected_feature_ = {'c'};
  EXPECT_TRUE(TestGJK());
  EXPECT_TRUE(TestExhaustive());
}

TEST_F(SingleSimplexTest, q6) {
  simplex_ = 'D';
  x_query_ = {0.4, 0.4};
  expected_proj_ = {0.4, 0.4};
  expected_dist_ = 0;
  expected_feature_ = {'a', 'b', 'c'};
  EXPECT_TRUE(TestGJK());
  EXPECT_TRUE(TestExhaustive());
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
class TripleSimplexTest : public SimplexDistanceTest {
  virtual void SetUp() {
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
    SetupSimplex('C', {'a', 'b', 'd'}, {'B', 'F', 'A'});
    SetupSimplex('D', {'\0', 'a', 'c'}, {'A', 'E', 'F'});
    SetupSimplex('E', {'\0', 'c', 'd'}, {'B', 'F', 'D'});
    SetupSimplex('F', {'\0', 'a', 'd'}, {'C', 'E', 'D'});
    ASSERT_PRED_FORMAT1(AssertGraphConsistentFrom, 'A');
  }
};

TEST_F(TripleSimplexTest, q0) {
  simplex_ = 'B';
  x_query_ = {1.5, 1.1};
  expected_proj_ = {1.5, 1.1};
  expected_dist_ = 0;
  expected_feature_ = {'b', 'c', 'd'};
  EXPECT_TRUE(TestGJK());
  EXPECT_TRUE(TestExhaustive());
}

TEST_F(TripleSimplexTest, q1) {
  simplex_ = 'B';
  x_query_ = {1.5, 2.0};
  expected_proj_ = {1.2, 1.4};
  expected_dist_ = 0.67082039;
  expected_feature_ = {'c', 'd'};
  EXPECT_TRUE(TestGJK());
  EXPECT_TRUE(TestExhaustive());
}

TEST_F(TripleSimplexTest, q2) {
  simplex_ = 'A';
  x_query_ = {-0.5, 0.9};
  expected_proj_ = {0.2, 0.2};
  expected_dist_ = 0.98994949366116647;
  expected_feature_ = {'a', 'b'};
  EXPECT_TRUE(TestGJK());
  EXPECT_TRUE(TestExhaustive());
}


/**
 *  This query fails in the gui
 */
TEST_F(SimplexDistanceTest, SpecificFailure0) {
  typedef Traits::Simplex Simplex;
  typedef Traits::SimplexRef SimplexRef;
  typedef Traits::Point Point;
  typedef Traits::PointRef PointRef;

  // vertices
  storage_.points['a'] = Point(-0.17, 2.71);
  storage_.points['b'] = Point(6.84, 2.77);
  storage_.points['c'] = Point(4.92, 0.45);

  // simplices
  SetupSimplex('A', {'\0', 'b', 'c'}, {'D', 'B', 'C'});
  SetupSimplex('B', {'\0', 'a', 'c'}, {'D', 'A', 'C'});
  SetupSimplex('C', {'\0', 'a', 'b'}, {'D', 'A', 'B'});
  SetupSimplex('D', {'a', 'b', 'c'}, {'A', 'B', 'C'});
  ASSERT_PRED_FORMAT1(AssertGraphConsistentFrom, 'A');

  simplex_ = 'D';
  x_query_ = {3.11, 0.13};
  expected_proj_ = {3.5267516451345609, 1.0686132184667765};
  expected_dist_ = 1.026974540873786;
  expected_feature_ = {'a', 'c'};
  EXPECT_TRUE(TestGJK());
  EXPECT_TRUE(TestExhaustive());
}
