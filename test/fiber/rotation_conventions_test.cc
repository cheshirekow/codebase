/*
 *  Copyright (C) 2014 Josh Bialkowski (jbialk@mit.edu)
 *
 *  This file is part of fiber.
 *
 *  fiber is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  fiber is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with fiber.  If not, see <http://www.gnu.org/licenses/>.
 */
/**
 *  @file
 *  @date   Dec 12, 2014
 *  @author Josh Bialkowski (jbialk@mit.edu)
 */


#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <fiber/fiber.h>

#include "assert_equals.h"

using namespace fiber;

TEST(RotationConventionsTest, RotationAboutZTest) {
  fiber::Vector3d v, r_original, r_expected, r_actual;
  fiber::Quaterniond q;
  fiber::Matrix3d R;

  // create a rotation about the z-axis
  double theta = M_PI/10;
  fiber::CoordinateAxisAngleZd caa(theta);
  fiber::CoordinateAxisAngleToQuaternion(caa,&q);
  fiber::CoordinateAxisAngleToRotationMatrix(caa,&R);

  // we will rotate the x-axis about the z-axis, and w expect:
  r_original = fiber::AxisXd();
  r_expected << std::cos(theta), std::sin(theta), 0;
  r_actual = q.Rotate(r_original);
  EXPECT_PRED_FORMAT3(AssertMatrixApproxEquality, r_expected, r_actual, 1e-9);
  r_actual = R*r_original;
  EXPECT_PRED_FORMAT3(AssertMatrixApproxEquality, r_expected, r_actual, 1e-9);

  fiber::AxisAngled aa( fiber::AxisZd(), theta);
  fiber::AxisAngleToQuaternion(aa,&q);
  fiber::AxisAngleToRotationMatrix(aa, &R);
  r_actual = q.Rotate(r_original);
  EXPECT_PRED_FORMAT3(AssertMatrixApproxEquality, r_expected, r_actual, 1e-9);
  r_actual = R*r_original;
  EXPECT_PRED_FORMAT3(AssertMatrixApproxEquality, r_expected, r_actual, 1e-9);
}
