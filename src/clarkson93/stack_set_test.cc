#include <gtest/gtest.h>

#include "clarkson93/stack_set.h"

enum TestSet {
  IN_STACK_A = 0,
  IN_STACK_B,
  NUM_SETS
};

typedef clarkson93::BitMember<TestSet, NUM_SETS> TestItem;

TEST(StackSetTest,ItemInStackHasCorrectMembership) {
  TestItem item_a;
  TestItem item_b;
  TestItem item_c;

  clarkson93::StackSet<TestItem> stack_a(IN_STACK_A);
  clarkson93::StackSet<TestItem> stack_b(IN_STACK_B);

  EXPECT_FALSE(item_a.IsMemberOf(IN_STACK_A));
  EXPECT_FALSE(item_a.IsMemberOf(IN_STACK_B));
  EXPECT_FALSE(item_b.IsMemberOf(IN_STACK_A));
  EXPECT_FALSE(item_b.IsMemberOf(IN_STACK_B));
  EXPECT_FALSE(item_c.IsMemberOf(IN_STACK_A));
  EXPECT_FALSE(item_c.IsMemberOf(IN_STACK_B));

  stack_a.Push(&item_a);
  stack_a.Push(&item_c);
  stack_b.Push(&item_b);
  stack_b.Push(&item_c);

  EXPECT_TRUE(item_a.IsMemberOf(IN_STACK_A));
  EXPECT_FALSE(item_a.IsMemberOf(IN_STACK_B));
  EXPECT_FALSE(item_b.IsMemberOf(IN_STACK_A));
  EXPECT_TRUE(item_b.IsMemberOf(IN_STACK_B));
  EXPECT_TRUE(item_c.IsMemberOf(IN_STACK_A));
  EXPECT_TRUE(item_c.IsMemberOf(IN_STACK_B));

  stack_a.Pop();
  stack_b.Pop();

  EXPECT_TRUE(item_a.IsMemberOf(IN_STACK_A));
  EXPECT_FALSE(item_a.IsMemberOf(IN_STACK_B));
  EXPECT_FALSE(item_b.IsMemberOf(IN_STACK_A));
  EXPECT_TRUE(item_b.IsMemberOf(IN_STACK_B));
  EXPECT_FALSE(item_c.IsMemberOf(IN_STACK_A));
  EXPECT_FALSE(item_c.IsMemberOf(IN_STACK_B));

  stack_a.Clear();
  stack_b.Clear();

  EXPECT_FALSE(item_a.IsMemberOf(IN_STACK_A));
  EXPECT_FALSE(item_a.IsMemberOf(IN_STACK_B));
  EXPECT_FALSE(item_b.IsMemberOf(IN_STACK_A));
  EXPECT_FALSE(item_b.IsMemberOf(IN_STACK_B));
  EXPECT_FALSE(item_c.IsMemberOf(IN_STACK_A));
  EXPECT_FALSE(item_c.IsMemberOf(IN_STACK_B));
}
