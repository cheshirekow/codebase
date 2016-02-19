#include <gtest/gtest.h>
#include <tap/tap.h>

TEST(TapTest, TestEmptyParser) {
  int argc = 0;
  tap::ArgumentParser parser;
  parser.ParseArgs(&argc, nullptr);
}
