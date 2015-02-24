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
  testing::AssertionResult AssertScaledAMatchesExpected(
      const std::string& scale_str, double scale) {
    linalg::Matrix<double, Rows, Cols> scaledA = scale * m_A;
    for (int i = 0; i < Rows; i++) {
      for (int j = 0; j < Cols; j++) {
        if (scaledA(i, j) != scale * m_A(i, j)) {
          return testing::AssertionFailure()
                 << "s*A != expected"
                 << "\n where s: " << scale << "\n A:\n" << m_A
                 << "\n computed s*A :\n" << scaledA;
        }
      }
    }
    return testing::AssertionSuccess();
  }

 protected:
  linalg::Matrix<double, Rows, Cols> m_A;
};

typedef ScaleTest<3,3> ScaleTest3x3;
typedef ScaleTest<3,4> ScaleTest3x4;
typedef ScaleTest<3,6> ScaleTest3x6;

TEST_F(ScaleTest3x3, ManualTests) {
  m_A << 0, 1, 2,
         3, 4, 5,
         6, 7, 8;
  EXPECT_PRED_FORMAT1(AssertScaledAMatchesExpected, 2.0);
  EXPECT_PRED_FORMAT1(AssertScaledAMatchesExpected, 4.0);
  EXPECT_PRED_FORMAT1(AssertScaledAMatchesExpected, 10.0);
}

TEST_F(ScaleTest3x4, ManualTests) {
  m_A << 0, 1,  2,  3,
         4, 5,  6,  7,
         8, 9, 10, 11;
  EXPECT_PRED_FORMAT1(AssertScaledAMatchesExpected, 2.0);
  EXPECT_PRED_FORMAT1(AssertScaledAMatchesExpected, 4.0);
  EXPECT_PRED_FORMAT1(AssertScaledAMatchesExpected, 10.0);
}

TEST_F(ScaleTest3x6, ManualTests) {
  m_A <<   0,  1,  2,  3,  4,  5,
           6,  7,  8,  9, 10, 11,
          12, 13, 14, 15, 16, 17;
  EXPECT_PRED_FORMAT1(AssertScaledAMatchesExpected, 2.0);
  EXPECT_PRED_FORMAT1(AssertScaledAMatchesExpected, 4.0);
  EXPECT_PRED_FORMAT1(AssertScaledAMatchesExpected, 10.0);
}
