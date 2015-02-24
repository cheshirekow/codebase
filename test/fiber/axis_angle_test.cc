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
 *  @date   Dec 09, 2014
 *  @author Josh Bialkowski (jbialk@mit.edu)
 */

#include <gtest/gtest.h>
#include <fiber/fiber.h>

#include "assert_equals.h"

using namespace fiber;

TEST(AxisAngleTest,ManualTests) {

}

TEST(CoordinateAxisAngleTest,ManualTests) {
  EXPECT_PRED_FORMAT2(AssertMatrixEquality,
                      (fiber::Matrix<double, 3, 1>(1, 0, 0)),
                      (fiber::CoordinateAxis<double, 0>()));
  EXPECT_PRED_FORMAT2(AssertMatrixEquality,
                      (fiber::Matrix<double, 3, 1>(0, 1, 0)),
                      (fiber::CoordinateAxis<double, 1>()));
  EXPECT_PRED_FORMAT2(AssertMatrixEquality,
                      (fiber::Matrix<double, 3, 1>(0, 0, 1)),
                      (fiber::CoordinateAxis<double, 2>()));

  EXPECT_PRED_FORMAT3(
      AssertMatrixApproxEquality,
      (fiber::Quaternion<double>(0, 1, 0, 0)),
      (fiber::CoordinateAxisAngle<double, 0>(M_PI).ToQuaternion()),
      1e-16);
  EXPECT_PRED_FORMAT3(
      AssertMatrixApproxEquality, (fiber::Quaternion<double>(0, 0, 1, 0)),
      (fiber::CoordinateAxisAngle<double, 1>(M_PI).ToQuaternion()),
      1e-16);
  EXPECT_PRED_FORMAT3(
      AssertMatrixApproxEquality, (fiber::Quaternion<double>(0, 0, 0, 1)),
      (fiber::CoordinateAxisAngle<double, 2>(M_PI).ToQuaternion()),
      1e-16);
}
