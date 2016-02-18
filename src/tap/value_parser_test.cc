#include <gtest/gtest.h>
#include "value_parsers.h"

TEST(TapValueParserTest, TestStringParser) {
  char kStringValue[] = "Hello World";
  std::string outvar;
  EXPECT_EQ(0, tap::ParseValue(kStringValue, &outvar));
  EXPECT_EQ(outvar, kStringValue);
}

TEST(TapValueParserTest, TestDoubleParser) {
  char kGoodValue[] = "-12.345";
  char kBadValue[] = "omg12.345";
  double outvar;
  EXPECT_EQ(0, tap::ParseValue(kGoodValue, &outvar));
  EXPECT_EQ(outvar, -12.345);

  EXPECT_NE(0, tap::ParseValue(kBadValue, &outvar));
}

TEST(TapValueParserTest, TestFloatParser) {
  char kGoodValue[] = "-12.345";
  char kBadValue[] = "omg12.345";
  float outvar;
  EXPECT_EQ(0, tap::ParseValue(kGoodValue, &outvar));
  EXPECT_EQ(outvar, -12.345f);

  EXPECT_NE(0, tap::ParseValue(kBadValue, &outvar));
}

TEST(TapValueParserTest, TestUnsignedParser8) {
  uint8_t outvar;

  // test a good conversion
  {
    char kStrValue[] = "123";
    EXPECT_EQ(0, tap::ParseValue(kStrValue, &outvar));
    EXPECT_EQ(123, outvar);
  }

  // test correct handling of negative / unexpected chars
  {
    char kStrValue[] = "-123";
    EXPECT_NE(0, tap::ParseValue(kStrValue, &outvar));
  }

  // test correct handling of initial zeros
  {
    char kStrValue[] = "00000000000123";
    EXPECT_EQ(0, tap::ParseValue(kStrValue, &outvar));
    EXPECT_EQ(123, outvar);
  }

  // test correct handling of overflow
  {
    char kStrValue[] = "257";
    EXPECT_NE(0, tap::ParseValue(kStrValue, &outvar));
  }

  // test max value doesn't overflow
  {
    char kStrValue[] = "255";
    EXPECT_EQ(0, tap::ParseValue(kStrValue, &outvar));
    EXPECT_EQ(255, outvar);
  }
}

TEST(TapValueParserTest, TestSignedParser8) {
  int8_t outvar = 'x';

  // test a good conversion
  {
    char kStrValue[] = "123";
    EXPECT_EQ(0, tap::ParseValue(kStrValue, &outvar));
    EXPECT_EQ(123, outvar);
  }

  // test correct handling of negative / unexpected chars
  {
    char kStrValue[] = "-123";
    EXPECT_EQ(0, tap::ParseValue(kStrValue, &outvar));
    EXPECT_EQ(-123, outvar);
  }

  // test correct handling of initial zeros
  {
    char kStrValue[] = "-00000000000123";
    EXPECT_EQ(0, tap::ParseValue(kStrValue, &outvar));
    EXPECT_EQ(-123, outvar);
  }

  // test correct handling of overflow
  {
    char kStrValue[] = "129";
    EXPECT_NE(0, tap::ParseValue(kStrValue, &outvar));
  }

  // test max value doesn't overflow
  {
    char kStrValue[] = "127";
    EXPECT_EQ(0, tap::ParseValue(kStrValue, &outvar));
    EXPECT_EQ(127, outvar);
  }

  // test min value doesn't overflow
  {
    char kStrValue[] = "-128";
    EXPECT_EQ(0, tap::ParseValue(kStrValue, &outvar));
    EXPECT_EQ(-128, outvar);
  }
}
