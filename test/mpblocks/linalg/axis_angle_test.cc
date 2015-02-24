/*
 *  Copyright (C) 2014 Josh Bialkowski (jbialk@mit.edu)
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
/**
 *  @file
 *  @date   Dec 09, 2014
 *  @author Josh Bialkowski (jbialk@mit.edu)
 */

#include <gtest/gtest.h>
#include <mpblocks/linalg.h>

#include "assert_equals.h"

using namespace mpblocks;

TEST(AxisAngleTest,ManualTests) {

}

TEST(CoordinateAxisAngleTest,ManualTests) {
  EXPECT_PRED_FORMAT2(AssertMatrixEquality,
                      (linalg::Matrix<double, 3, 1>(1, 0, 0)),
                      (linalg::CoordinateAxis<double, 0>()));
  EXPECT_PRED_FORMAT2(AssertMatrixEquality,
                      (linalg::Matrix<double, 3, 1>(0, 1, 0)),
                      (linalg::CoordinateAxis<double, 1>()));
  EXPECT_PRED_FORMAT2(AssertMatrixEquality,
                      (linalg::Matrix<double, 3, 1>(0, 0, 1)),
                      (linalg::CoordinateAxis<double, 2>()));

  EXPECT_PRED_FORMAT3(
      AssertMatrixApproxEquality,
      (linalg::Quaternion<double>(0, 1, 0, 0)),
      (linalg::CoordinateAxisAngle<double, 0>(M_PI).ToQuaternion()),
      1e-16);
  EXPECT_PRED_FORMAT3(
      AssertMatrixApproxEquality, (linalg::Quaternion<double>(0, 0, 1, 0)),
      (linalg::CoordinateAxisAngle<double, 1>(M_PI).ToQuaternion()),
      1e-16);
  EXPECT_PRED_FORMAT3(
      AssertMatrixApproxEquality, (linalg::Quaternion<double>(0, 0, 0, 1)),
      (linalg::CoordinateAxisAngle<double, 2>(M_PI).ToQuaternion()),
      1e-16);
}
