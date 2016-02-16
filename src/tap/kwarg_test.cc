#include <gtest/gtest.h>
#include <tap/tap.h>

TEST(TapKwargsTest, TestHash) {
  enum { kHashCT = tap::HashStringCT("HelloWorld") };

  EXPECT_EQ(kHashCT, tap::HashString("HelloWorld"));
}
