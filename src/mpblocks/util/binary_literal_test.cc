#include <cstdint>
#include <gtest/gtest.h>
#include <mpblocks/util/binary_literal.h>

TEST(BinaryLiteralTest,StringLengthTest) {
  using namespace mpblocks::utility;

  static_assert(12 == StringLength("hello world!"),
                "Length of 'hello world!' is not 12!");
  EXPECT_EQ(12,StringLength("hello world!"));
}

TEST(BinaryLiteralTest,CharIsBinaryTest) {
  using namespace mpblocks::utility;

  static_assert(IsBinary('0'), "Zero is not binary?!?");
  EXPECT_TRUE(IsBinary('0'));

  static_assert(IsBinary('1'), "One is not binary?!?");
  EXPECT_TRUE(IsBinary('1'));

  static_assert(!IsBinary('2'), "Two is binary?!?");
  EXPECT_FALSE(IsBinary('2'));

  static_assert(!IsBinary('a'), "'a' is binary?!?");
  EXPECT_FALSE(IsBinary('a'));
}

TEST(BinaryLiteralTest, TestSomeValues) {
  using namespace mpblocks::utility;

  static_assert(0 == BinaryLiteral("00"),
                "'00' does not evaluate to correct binary value");
  EXPECT_EQ(0, BinaryLiteral("00"));

  static_assert(1 == BinaryLiteral("01"),
                "'01' does not evaluate to correct binary value");
  EXPECT_EQ(1, BinaryLiteral("01"));

  static_assert(2 == BinaryLiteral("10"),
                "'10' does not evaluate to correct binary value");
  EXPECT_EQ(2, BinaryLiteral("10"));

  static_assert(3 == BinaryLiteral("11"),
                "'11' does not evaluate to correct binary value");
  EXPECT_EQ(3, BinaryLiteral("11"));

  static_assert(4 == BinaryLiteral("100"),
                "'100' does not evaluate to correct binary value");
  EXPECT_EQ(4, BinaryLiteral("100"));

  //  static_assert(0 == BinaryLiteral("abc"),
  //                "This line should fail to compile");
  EXPECT_THROW(BinaryLiteral("abc"), std::logic_error);
}

TEST(BinaryLiteralTest, TestMacros) {
  static_assert(0 == BINARY_LITERAL(00),
                "'00' does not evaluate to correct binary value");
  EXPECT_EQ(0, BINARY_LITERAL(00));

  static_assert(1 == BINARY_LITERAL(01),
                "'01' does not evaluate to correct binary value");
  EXPECT_EQ(1, BINARY_LITERAL(01));

  static_assert(2 == BINARY_LITERAL(10),
                "'10' does not evaluate to correct binary value");
  EXPECT_EQ(2, BINARY_LITERAL(10));

  static_assert(3 == BINARY_LITERAL(11),
                "'11' does not evaluate to correct binary value");
  EXPECT_EQ(3, BINARY_LITERAL(11));

  static_assert(4 == BINARY_LITERAL(100),
                "'100' does not evaluate to correct binary value");
  EXPECT_EQ(4, BINARY_LITERAL(100));

  //  static_assert(0 == BINARY_LITERAL(abc),
  //                "This line should fail to compile");
  EXPECT_THROW(BINARY_LITERAL("abc"), std::logic_error);
}


TEST(BinaryLiteralTest, TestNonDefaultType) {
  using namespace mpblocks::utility;

  static_assert(0 == BinaryLiteral<uint8_t>("00"),
                "'00' does not evaluate to correct binary value");
  EXPECT_EQ(0, BinaryLiteral<uint8_t>("00"));

  static_assert(1 == BinaryLiteral<uint8_t>("01"),
                "'01' does not evaluate to correct binary value");
  EXPECT_EQ(1, BinaryLiteral<uint8_t>("01"));

  static_assert(2 == BinaryLiteral<uint8_t>("10"),
                "'10' does not evaluate to correct binary value");
  EXPECT_EQ(2, BinaryLiteral<uint8_t>("10"));

  static_assert(3 == BinaryLiteral<uint8_t>("11"),
                "'11' does not evaluate to correct binary value");
  EXPECT_EQ(3, BinaryLiteral<uint8_t>("11"));

  static_assert(4 == BinaryLiteral<uint8_t>("100"),
                "'100' does not evaluate to correct binary value");
  EXPECT_EQ(4, BinaryLiteral<uint8_t>("100"));

  //static_assert(0 == BinaryLiteral<uint8_t>("1000000000"),
  //              "This line should fail to compile");
  EXPECT_THROW(BinaryLiteral<uint8_t>("10000000000"), std::logic_error);
}

#if (__cplusplus >= 201103L)
TEST(BinaryLiteralTest, TestUserDefinedLiteral) {
  using namespace mpblocks::utility;

  static_assert(0 == 00_b,
                "'00' does not evaluate to correct binary value");
  EXPECT_EQ(0, 00_b);

  static_assert(1 == 01_b,
                "'01' does not evaluate to correct binary value");
  EXPECT_EQ(1, 01_b);

  static_assert(2 == 10_b,
                "'10' does not evaluate to correct binary value");
  EXPECT_EQ(2, 10_b);

  static_assert(3 == 11_b,
                "'11' does not evaluate to correct binary value");
  EXPECT_EQ(3, 11_b);

  static_assert(4 == 100_b,
                "'100' does not evaluate to correct binary value");
  EXPECT_EQ(4, 100_b);
}
#endif


