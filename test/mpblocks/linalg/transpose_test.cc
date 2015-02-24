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

using namespace mpblocks;

template <int Rows, int Cols>
class ScaleTest : public testing::Test {
 public:
  testing::AssertionResult AssertATransposedMatchesExpected() {
    m_computedT = linalg::Transpose(m_A);
    for(int i=0; i < Rows; i++) {
      for(int j = 0; j < Cols; j++) {
        if(m_A(i,j) != m_computedT(j,i)) {
          return testing::AssertionFailure() << "A^T != expected"
              << "\n where A:\n"
              << m_A
              << "\n computed transpose:\n"
              << m_computedT;
        }
      }
    }
    return testing::AssertionSuccess();
  }

 protected:
  linalg::Matrix<double, Rows, Cols> m_A;
  linalg::Matrix<double, Cols, Rows> m_computedT;
};

typedef ScaleTest<3,3> TransposeTest3x3;
typedef ScaleTest<3,4> TransposeTest3x4;
typedef ScaleTest<3,6> TransposeTest3x6;

TEST_F(TransposeTest3x3, ManualTests) {
  m_A << 0, 1, 2,
         3, 4, 5,
         6, 7, 8;
  EXPECT_TRUE(AssertATransposedMatchesExpected());
}

TEST_F(TransposeTest3x4, ManualTests) {
  m_A << 0, 1,  2,  3,
         4, 5,  6,  7,
         8, 9, 10, 11;
  EXPECT_TRUE(AssertATransposedMatchesExpected());
}

TEST_F(TransposeTest3x6, ManualTests) {
  m_A <<   0,  1,  2,  3,  4,  5,
           6,  7,  8,  9, 10, 11,
          12, 13, 14, 15, 16, 17;
  EXPECT_TRUE(AssertATransposedMatchesExpected());
}
