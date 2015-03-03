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
 *  @date   Dec 9, 2014
 *  @author Josh Bialkowski (jbialk@mit.edu)
 */

#include <gtest/gtest.h>
#include <fiber/fiber.h>

#include "assert_equals.h"

using namespace fiber;

TEST(BlockTest,test4x4) {
  fiber::Matrix<double,4,4> A;
  A <<  1,  2,  3,  4,
        5,  6,  7,  8,
        9, 10, 11, 12,
       13, 14, 15, 16;

  fiber::Matrix<double,2,2> expected2x2;
  expected2x2 << 1, 2,
                 5, 6;
  EXPECT_PRED_FORMAT2(AssertMatrixEquality, (fiber::View<2,2>(A, 0, 0)),
                      expected2x2);
  expected2x2 << 2, 3,
                 6, 7;
  EXPECT_PRED_FORMAT2(AssertMatrixEquality, (fiber::View<2,2>(A, 0, 1)),
                      expected2x2);
  expected2x2 << 11, 12,
                 15, 16;
  EXPECT_PRED_FORMAT2(AssertMatrixEquality, (fiber::View<2,2>(A, 2, 2)),
                      expected2x2);

  fiber::Matrix<double,4,1> expected4x1;
  expected4x1 << 1, 5, 9, 13;
  EXPECT_PRED_FORMAT2(AssertMatrixEquality, fiber::GetColumn(A, 0), expected4x1);
  expected4x1 << 2, 6, 10, 14;
  EXPECT_PRED_FORMAT2(AssertMatrixEquality, fiber::GetColumn(A, 1), expected4x1);
  expected4x1 << 3, 7, 11, 15;
  EXPECT_PRED_FORMAT2(AssertMatrixEquality, fiber::GetColumn(A, 2), expected4x1);
  expected4x1 << 4, 8, 12, 16;
  EXPECT_PRED_FORMAT2(AssertMatrixEquality, fiber::GetColumn(A, 3), expected4x1);

  fiber::Matrix<double,1,4> expected1x4;
  expected1x4 << 1, 2, 3, 4;
  EXPECT_PRED_FORMAT2(AssertMatrixEquality, fiber::GetRow(A, 0), expected1x4);
  expected1x4 << 5, 6, 7, 8;
  EXPECT_PRED_FORMAT2(AssertMatrixEquality, fiber::GetRow(A, 1), expected1x4);
  expected1x4 << 9, 10, 11, 12;
  EXPECT_PRED_FORMAT2(AssertMatrixEquality, fiber::GetRow(A, 2), expected1x4);
  expected1x4 << 13, 14, 15, 16;
  EXPECT_PRED_FORMAT2(AssertMatrixEquality, fiber::GetRow(A, 3), expected1x4);

  fiber::Block<2,2>(A,0,0) << -1, -2, -5, -6;
  fiber::Matrix<double,4,4> B;
  B << -1, -2,  3,  4,
       -5, -6,  7,  8,
        9, 10, 11, 12,
       13, 14, 15, 16;
  EXPECT_PRED_FORMAT2(AssertMatrixEquality, A, B);

}
