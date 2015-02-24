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
 *  @date   Dec 9, 2014
 *  @author Josh Bialkowski (jbialk@mit.edu)
 */

#include <gtest/gtest.h>
#include <mpblocks/linalg.h>

#include "assert_equals.h"

using namespace mpblocks;

TEST(BlockTest,test4x4) {
  linalg::Matrix<double,4,4> A;
  A <<  1,  2,  3,  4,
        5,  6,  7,  8,
        9, 10, 11, 12,
       13, 14, 15, 16;

  linalg::Matrix<double,2,2> expected2x2;
  expected2x2 << 1, 2,
              5, 6;
  EXPECT_PRED_FORMAT2(AssertMatrixEquality, linalg::View(A, 0, 0, 2, 2),
                      expected2x2);
  expected2x2 << 2, 3,
              6, 7;
  EXPECT_PRED_FORMAT2(AssertMatrixEquality, linalg::View(A, 0, 1, 2, 2),
                      expected2x2);
  expected2x2 << 11, 12,
              15, 16;
  EXPECT_PRED_FORMAT2(AssertMatrixEquality, linalg::View(A, 2, 2, 2, 2),
                      expected2x2);

  linalg::Matrix<double,4,1> expected4x1;
  expected4x1 << 1, 5, 9, 13;
  EXPECT_PRED_FORMAT2(AssertMatrixEquality, linalg::Column(A, 0), expected4x1);
  expected4x1 << 2, 6, 10, 14;
  EXPECT_PRED_FORMAT2(AssertMatrixEquality, linalg::Column(A, 1), expected4x1);
  expected4x1 << 3, 7, 11, 15;
  EXPECT_PRED_FORMAT2(AssertMatrixEquality, linalg::Column(A, 2), expected4x1);
  expected4x1 << 4, 8, 12, 16;
  EXPECT_PRED_FORMAT2(AssertMatrixEquality, linalg::Column(A, 3), expected4x1);

  linalg::Matrix<double,1,4> expected1x4;
  expected1x4 << 1, 2, 3, 4;
  EXPECT_PRED_FORMAT2(AssertMatrixEquality, linalg::Row(A, 0), expected1x4);
  expected1x4 << 5, 6, 7, 8;
  EXPECT_PRED_FORMAT2(AssertMatrixEquality, linalg::Row(A, 1), expected1x4);
  expected1x4 << 9, 10, 11, 12;
  EXPECT_PRED_FORMAT2(AssertMatrixEquality, linalg::Row(A, 2), expected1x4);
  expected1x4 << 13, 14, 15, 16;
  EXPECT_PRED_FORMAT2(AssertMatrixEquality, linalg::Row(A, 3), expected1x4);

  linalg::Block(A,0,0,2,2) << -1, -2, -5, -6;
  linalg::Matrix<double,4,4> B;
  B << -1, -2,  3,  4,
       -5, -6,  7,  8,
        9, 10, 11, 12,
       13, 14, 15, 16;
  EXPECT_PRED_FORMAT2(AssertMatrixEquality, A, B);

}
