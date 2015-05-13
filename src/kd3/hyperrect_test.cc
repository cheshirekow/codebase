/*
 *  Copyright (C) 2012 Josh Bialkowski (josh.bialkowski@gmail.com)
 *
 *  This file is part of kd3.
 *
 *  kd3 is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  kd3 is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with kd3.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <gtest/gtest.h>
#include <kd3/hyperrect.h>

TEST(HyperRectTest, SplitLesserTest) {
  kd3::HyperRect<double, 2> hrect{{0, 0}, {1, 1}};
  hrect.SplitLesser(0, 0.5);
  EXPECT_EQ(0.0, hrect.min_ext[0]);
  EXPECT_EQ(0.0, hrect.min_ext[1]);
  EXPECT_EQ(0.5, hrect.max_ext[0]);
  EXPECT_EQ(1.0, hrect.max_ext[1]);
  hrect.SplitLesser(1, 0.5);
  EXPECT_EQ(0.0, hrect.min_ext[0]);
  EXPECT_EQ(0.0, hrect.min_ext[1]);
  EXPECT_EQ(0.5, hrect.max_ext[0]);
  EXPECT_EQ(0.5, hrect.max_ext[1]);
}

TEST(HyperRectTest, SplitGreaterTest) {
  kd3::HyperRect<double, 2> hrect{{0, 0}, {1, 1}};
  hrect.SplitGreater(0, 0.5);
  EXPECT_EQ(0.5, hrect.min_ext[0]);
  EXPECT_EQ(0.0, hrect.min_ext[1]);
  EXPECT_EQ(1.0, hrect.max_ext[0]);
  EXPECT_EQ(1.0, hrect.max_ext[1]);
  hrect.SplitGreater(1, 0.5);
  EXPECT_EQ(0.5, hrect.min_ext[0]);
  EXPECT_EQ(0.5, hrect.min_ext[1]);
  EXPECT_EQ(1.0, hrect.max_ext[0]);
  EXPECT_EQ(1.0, hrect.max_ext[1]);
}
