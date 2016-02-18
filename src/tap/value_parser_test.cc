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
