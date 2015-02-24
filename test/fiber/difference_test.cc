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

using namespace fiber;

template <int Rows, int Cols>
class TransposeTest : public testing::Test {
 public:
  testing::AssertionResult AssertAMinusBEqualsC() {
    m_computedC = m_A - m_B;
    for(int i=0; i < Rows; i++) {
      for(int j = 0; j < Cols; j++) {
        if(m_expectedC(i,j) != m_computedC(i,j)) {
          return testing::AssertionFailure() << "A - B != C"
              << "\n where A:\n"
              << m_A
              << "\n B:\n"
              << m_B
              << "\n expected C:\n"
              << m_expectedC
              << "\n computed C:\n"
              << m_computedC;
        }
      }
    }
    return testing::AssertionSuccess();
  }

 protected:
  fiber::Matrix<double, Rows, Cols> m_A;
  fiber::Matrix<double, Rows, Cols> m_B;
  fiber::Matrix<double, Rows, Cols> m_expectedC;
  fiber::Matrix<double, Rows, Cols> m_computedC;
};

typedef TransposeTest<3,3> DifferenceTest3x3;
typedef TransposeTest<3,4> DifferenceTest3x4;
typedef TransposeTest<3,6> DifferenceTest3x6;

TEST_F(DifferenceTest3x3,ManualTests) {
  m_A << 1, 2, 3,
         4, 5, 6,
         7, 8, 9;
  m_B << 0, 1, 2,
         3, 4, 5,
         6, 7, 8;
  m_expectedC << 1, 1, 1,
                 1, 1, 1,
                 1, 1, 1;
  EXPECT_TRUE(AssertAMinusBEqualsC());
}

TEST_F(DifferenceTest3x4,ManualTests) {
  m_A << 1,  2,  3,  4,
         5,  6,  7,  8,
         9, 10, 11, 12;
  m_B << 0,  1,  2,  3,
         4,  5,  6,  7,
         8,  9, 10, 11;
  m_expectedC << 1, 1, 1, 1,
                 1, 1, 1, 1,
                 1, 1, 1, 1;
  EXPECT_TRUE(AssertAMinusBEqualsC());
}

TEST_F(DifferenceTest3x6,ManualTests) {
  m_A <<  1,  2,  3,  4,  5,  6,
          7,  8,  9, 10, 11, 12,
         13, 14, 15, 16, 17, 18;
  m_B <<  0,  1,  2,  3,  4,  5,
          6,  7,  8,  9, 10, 11,
         12, 13, 14, 15, 16, 17;
  m_expectedC << 1, 1, 1, 1, 1, 1,
                 1, 1, 1, 1, 1, 1,
                 1, 1, 1, 1, 1, 1;
  EXPECT_TRUE(AssertAMinusBEqualsC());
}
