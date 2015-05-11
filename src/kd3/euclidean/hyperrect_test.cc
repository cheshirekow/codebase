#include <gtest/gtest.h>
#include <kd3/euclidean/hyperrect.h>

struct Traits {
  typedef double Scalar;
  static const unsigned int NDim = 2;
};

TEST(HyperRect, SimpleTest) {
  kd3::euclidean::HyperRect<Traits> hrect;
  hrect.max_ext = Eigen::Vector2d(1, 1);

  EXPECT_EQ(1.0, hrect.GetSquaredDistanceTo(Eigen::Vector2d(2.0, 0.0)));
  EXPECT_EQ(1.0, hrect.GetSquaredDistanceTo(Eigen::Vector2d(0.0, 2.0)));
  EXPECT_EQ(1.0, hrect.GetMeasure());
}
