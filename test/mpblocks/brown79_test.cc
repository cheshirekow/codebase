/*
 *  Copyright (C) 2012 Josh Bialkowski (jbialk@mit.edu)
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

#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <mpblocks/brown79.hpp>

TEST(Brown79Test, SomeCanonicalCases2d) {
  struct Traits {
    typedef double Scalar;
    typedef Eigen::Matrix<Scalar, 2, 1> Point;
  };

  typedef mpblocks::brown79::Inversion<Traits> Inversion;
  typedef Inversion::Point Point;

  Point center(0, 0);
  Inversion inversion(center);

  /// Ensures that inversion of a point on the surface of the ball is involute
  std::vector<Point> test_points = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
  for (const Point& test_point : test_points) {
    Point inverted = inversion(test_point);
    EXPECT_GT(1e-16, (inverted - test_point).norm())
        << "For test point " << test_point.transpose()
        << " with computed inversion " << inverted.transpose();
  }

  /// Ensures that inversion of a point outside the ball falls inside the ball
  test_points = {{2, 0}, {0, 2}, {-2, 0}, {0, -2}};
  for (const Point& test_point : test_points) {
    Point inverted = inversion(test_point);
    EXPECT_LT(1, (test_point - center).norm())
        << "For test point " << test_point.transpose()
        << " with computed inversion " << inverted.transpose();
    EXPECT_GT(1, (inverted - center).norm())
        << "For test point " << test_point.transpose()
        << " with computed inversion " << inverted.transpose();
  }

  /// Ensures that inversion of a point inside the ball falls outside the ball
  test_points = {{0.5, 0}, {0, 0.5}, {-0.5, 0}, {0, -0.5}};
  for (const Point& test_point : test_points) {
    Point inverted = inversion(test_point);
    EXPECT_GT(1, (test_point - center).norm())
        << "For test point " << test_point.transpose()
        << " with computed inversion " << inverted.transpose();
    EXPECT_LT(1, (inverted - center).norm())
        << "For test point " << test_point.transpose()
        << " with computed inversion " << inverted.transpose();
  }
}

TEST(Brown79Test, SomeCanonicalCases3d) {
  struct Traits {
    typedef double Scalar;
    typedef Eigen::Matrix<Scalar,3,1> Point;
  };

  typedef mpblocks::brown79::Inversion<Traits> Inversion;
  typedef Inversion::Point Point;

  Point center(0,0,0);
  Inversion inversion(center);

  /// Ensures that inversion of a point on the surface of the ball is involute
  std::vector<Point> test_points = {
      {1, 0, 0},  {0, 1, 0},  {0, 0, 1},
      {-1, 0, 0}, {0, -1, 0}, {0, 0, -1}};
  for (const Point& test_point : test_points) {
    Point inverted = inversion(test_point);
    EXPECT_GT(1e-16, (inverted - test_point).norm())
        << "For test point " << test_point.transpose()
        << " with computed inversion " << inverted.transpose();
  }

  /// Ensures that inversion of a point outside the ball falls inside the ball
  test_points = {
        {2, 0, 0},  {0, 2, 0},  {0, 0, 2},
        {-2, 0, 0}, {0, -2, 0}, {0, 0, -2}};
  for (const Point& test_point : test_points) {
    Point inverted = inversion(test_point);
    EXPECT_LT(1, (test_point - center).norm())
        << "For test point " << test_point.transpose()
        << " with computed inversion " << inverted.transpose();
    EXPECT_GT(1, (inverted - center).norm())
        << "For test point " << test_point.transpose()
        << " with computed inversion " << inverted.transpose();
  }

  /// Ensures that inversion of a point inside the ball falls outside the ball
  test_points = {
          {0.5, 0, 0},  {0, 0.5, 0},  {0, 0, 0.5},
          {-0.5, 0, 0}, {0, -0.5, 0}, {0, 0, -0.5}};
  for (const Point& test_point : test_points) {
    Point inverted = inversion(test_point);
    EXPECT_GT(1, (test_point - center).norm())
        << "For test point " << test_point.transpose()
        << " with computed inversion " << inverted.transpose();
    EXPECT_LT(1, (inverted - center).norm())
        << "For test point " << test_point.transpose()
        << " with computed inversion " << inverted.transpose();
  }
}
