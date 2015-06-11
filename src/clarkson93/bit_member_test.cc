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
