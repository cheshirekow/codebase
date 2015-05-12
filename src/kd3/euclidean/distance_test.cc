#include <gtest/gtest.h>
#include <kd3/hyperrect.h>
#include <kd3/euclidean/distance.h>

struct Traits {
  typedef double Scalar;
  static const unsigned int NDim = 2;
};

TEST(HyperRect, SimpleTest) {
  kd3::HyperRect<Traits> hrect;
  hrect.max_ext = Eigen::Vector2d(1, 1);

  kd3::euclidean::SquaredDistance<Traits> sq_dist;
  EXPECT_EQ(1.0, sq_dist(Eigen::Vector2d(2.0, 0.0), Eigen::Vector2d(1.0, 0.0)));
  EXPECT_EQ(1.0, sq_dist(Eigen::Vector2d(2.0, 0.0), hrect));
  EXPECT_EQ(1.0, sq_dist(Eigen::Vector2d(0.0, 2.0), hrect));

  kd3::euclidean::Distance<Traits> dist;
  EXPECT_EQ(1.0, dist(Eigen::Vector2d(2.0, 0.0), Eigen::Vector2d(1.0, 0.0)));
  EXPECT_EQ(1.0, dist(Eigen::Vector2d(2.0, 0.0), hrect));
  EXPECT_EQ(1.0, dist(Eigen::Vector2d(0.0, 2.0), hrect));

  EXPECT_EQ(1.0, hrect.GetMeasure());
}
