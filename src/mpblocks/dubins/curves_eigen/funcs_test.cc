#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <mpblocks/dubins/curves_eigen/funcs.hpp>

typedef Eigen::Matrix<double,2,1> Vector2d;
typedef Eigen::Matrix<double,3,1> Vector3d;
namespace dubins = mpblocks::dubins::curves_eigen;

TEST(ccwCenterTest, CanonicalCases) {
  std::vector<Vector3d> states = {
      {0, 0, 0},
      {0, 0, M_PI/2},
      {0, 0, M_PI},
      {0, 0, -M_PI/2},
  };

  std::vector<Vector2d> expect_centers = {
      {0, 1},
      {-1, 0},
      {0, -1},
      {1, 0}
  };

  for (std::size_t i = 0; i < states.size(); i++) {
    Vector3d state = states[i];
    Vector2d center = dubins::ccwCenter(state, 1.0);
    Vector2d expected = expect_centers[i];
    EXPECT_NEAR(expected[0], center[0], 1e-10) << " for state " << i << ": "
                                               << state.transpose();
    EXPECT_NEAR(expected[1], center[1], 1e-10) << " for state " << i << ": "
                                               << state.transpose();
  }
}

TEST(cwCenterTest, CanonicalCases) {
  std::vector<Vector3d> states = {
      {0, 0, 0},
      {0, 0, M_PI/2},
      {0, 0, M_PI},
      {0, 0, -M_PI/2},
  };

  std::vector<Vector2d> expect_centers = {
      {0, -1},
      {1, 0},
      {0, 1},
      {-1, 0}
  };

  for (std::size_t i = 0; i < states.size(); i++) {
    Vector3d state = states[i];
    Vector2d center = dubins::cwCenter(state, 1.0);
    Vector2d expected = expect_centers[i];
    EXPECT_NEAR(expected[0], center[0], 1e-10) << " for state " << i << ": "
                                               << state.transpose();
    EXPECT_NEAR(expected[1], center[1], 1e-10) << " for state " << i << ": "
                                               << state.transpose();
  }
}

TEST(ccwAngleOf, CanonicalCases) {
  std::vector<Vector2d> tests = {
      {0, -M_PI/2},
      {M_PI/2, 0},
      {M_PI, M_PI/2},
      {-M_PI/2, -M_PI}
  };

  for (std::size_t i = 0; i < tests.size(); i++) {
    EXPECT_NEAR(tests[i][1], dubins::ccwAngleOf(tests[i][0]), 1e-10)
        << "for test " << i << ": " << tests[i][0];
  }
}

TEST(cwAngleOf, CanonicalCases) {
  std::vector<Vector2d> tests = {
      {0, M_PI/2},
      {M_PI/2, M_PI},
      {M_PI, -M_PI/2},
      {-M_PI/2, 0}
  };

  for (std::size_t i = 0; i < tests.size(); i++) {
    EXPECT_NEAR(tests[i][1], dubins::cwAngleOf(tests[i][0]), 1e-10)
        << "for test " << i << ": " << tests[i][0];
  }
}
