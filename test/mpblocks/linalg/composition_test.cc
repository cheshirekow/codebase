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

TEST(CompositionTest,test4x4) {
  linalg::Matrix<double,4,4> A;
  A <<  1,  2,  3,  4,
        5,  6,  7,  8,
        9, 10, 11, 12,
       13, 14, 15, 16;

  linalg::Matrix<double,4,4> B;
  B << -1, -2,  0,  0,
       -5, -6,  0,  0,
        0,  0,  0,  0,
        0,  0,  0,  0;

  linalg::Matrix<double,4,4> C;
  C <<  0,  0,  3,  4,
        0,  0,  7,  8,
        9, 10, 11, 12,
       13, 14, 15, 16;

  EXPECT_PRED_FORMAT2(AssertMatrixEquality,C,(A+B));

  C <<  0,  0,  9, 13,
        0,  0, 10, 14,
        3,  7, 11, 15,
        4,  8, 12, 16;
  EXPECT_PRED_FORMAT2(AssertMatrixEquality,C,linalg::Transpose(A+B));

  C <<  0/2.,  0/2.,  9/2., 13/2.,
        0/2.,  0/2., 10/2., 14/2.,
        3/2.,  7/2., 11/2., 15/2.,
        4/2.,  8/2., 12/2., 16/2.;
  EXPECT_PRED_FORMAT2(AssertMatrixEquality,C,0.5*linalg::Transpose(A+B));
}
