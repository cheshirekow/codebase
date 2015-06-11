/*
 *  Copyright (C) 2015 Josh Bialkowski (josh.bialkowski@gmail.com)
 *
 *  This file is part of clarkson93.
 *
 *  clarkson93 is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  clarkson93 is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with clarkson93.  If not, see <http://www.gnu.org/licenses/>.
 */
#include <gtest/gtest.h>
#include "clarkson93/bit_member.h"

enum TestSets { SET_A, SET_B, SET_C, SET_D, NUM_SETS };

TEST(BitMemberTest, SetAndTestExpectedValues) {
  clarkson93::BitMember<TestSets, NUM_SETS> set_member;
  EXPECT_FALSE(set_member.IsMemberOf(SET_A));
  EXPECT_FALSE(set_member.IsMemberOf(SET_B));
  EXPECT_FALSE(set_member.IsMemberOf(SET_C));
  EXPECT_FALSE(set_member.IsMemberOf(SET_D));
  set_member.AddTo(SET_B);
  set_member.AddTo(SET_D);
  EXPECT_FALSE(set_member.IsMemberOf(SET_A));
  EXPECT_TRUE(set_member.IsMemberOf(SET_B));
  EXPECT_FALSE(set_member.IsMemberOf(SET_C));
  EXPECT_TRUE(set_member.IsMemberOf(SET_D));
}
