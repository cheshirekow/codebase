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


TEST(ProductTest,test3x3) {
  linalg::Matrix<double,3,3> A;
  A <<  1,  2,  3,
        4,  5,  6,
        7,  8,  9;

  linalg::Matrix<double,3,3> C;
  C <<  30,  36,  42,
        66,  81,  96,
        102, 126, 150;

  EXPECT_PRED_FORMAT2(AssertMatrixEquality,C,(A*A));
}

TEST(ProductTest,test4x4) {
  linalg::Matrix<double,4,4> A;
  A <<  1,  2,  3,  4,
        5,  6,  7,  8,
        9, 10, 11, 12,
       13, 14, 15, 16;

  linalg::Matrix<double,4,4> C;
  C <<  90,    100,   110,   120,
        202,   228,   254,   280,
        314,   356,   398,   440,
        426,   484,   542,   600;

  EXPECT_PRED_FORMAT2(AssertMatrixEquality,C,(A*A));
}

TEST(ProductTest,test3x5) {
  linalg::Matrix<double,3,5> A;
  A <<   1,  2,  3,  4,  5,
         6,  7,  8,  9, 10,
        11, 12, 13, 14, 15;

  linalg::Matrix<double,5,3> B;
  B <<   1,  2,  3,
         4,  5,  6,
         7,  8,  9,
        10, 11, 12,
        13, 14, 15;

  linalg::Matrix<double,3,3> C;
  C <<  135, 150, 165,
        310, 350, 390,
        485, 550, 615;

  EXPECT_PRED_FORMAT2(AssertMatrixEquality,C,(A*B));
}

TEST(ProductTest,test5x5) {
  linalg::Matrix<double,5,5> A;
  A <<   1,  2,  3,  4,  5,
         6,  7,  8,  9, 10,
        11, 12, 13, 14, 15,
        16, 17, 18, 19, 20,
        21, 22, 23, 24, 25;

  linalg::Matrix<double,5,5> C;
  C <<   215,    230,    245,    260,    275,
         490,    530,    570,    610,    650,
         765,    830,    895,    960,   1025,
        1040,   1130,   1220,   1310,   1400,
        1315,   1430,   1545,   1660,   1775;

  EXPECT_PRED_FORMAT2(AssertMatrixEquality,C,(A*A));
}
