/*
 *  Copyright (C) 2014 Josh Bialkowski (josh.bialkowski@gmail.com)
 *
 *  This file is part of cpp-nix.
 *
 *  cpp-nix is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  cpp-nix is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with cpp-nix.  If not, see <http://www.gnu.org/licenses/>.
 */
/**
 *  @file
 *  @date   June 7th, 2014
 *  @author Josh Bialkowski (josh.bialkowski@gmail.com)
 *  @brief  
 */


#include <gtest/gtest.h>

#include <cpp_nix/timespec.h>

TEST(TimespecTest,ComparisonOperators){
  timespec a{1,2};
  timespec b{3,4};
  EXPECT_LT(a,b);
  EXPECT_LE(a,b);
  EXPECT_GT(b,a);
  EXPECT_GE(b,a);
}


TEST(TimespecTest,EqualityOfConvertedType){
  timespec a{1,2};
  nix::Timespec cpp = a;
  EXPECT_EQ(a,cpp);
  EXPECT_LE(a,cpp);
  EXPECT_GE(a,cpp);
}

TEST(TimespecTest,Arithmetic){
  timespec zero{0,0};
  timespec one{1,1};
  timespec two{2,2};

  EXPECT_EQ(zero + zero, zero);
  EXPECT_EQ(zero + one, one);
  EXPECT_EQ(one + one, two);

  EXPECT_EQ(zero - zero, zero);
  EXPECT_EQ(one - zero, one);
  EXPECT_EQ(one - one, zero);
}

TEST(TimespecTest,AdditionOverflowHandled){
  EXPECT_EQ((timespec{0,999999999}) + (timespec{0,1}), (timespec{1,0}));
}

TEST(TimespecTest,SubtractionUndeflowHandled){
  EXPECT_EQ((timespec{1,0}) - (timespec{0,999999999}), (timespec{0,1}));
}
