#include <gtest/gtest.h>
#include <tap/tap.h>

TEST(TapTest, TestEmptyParser) {
  std::list<char*> args;
  tap::ArgumentParser parser;
  parser.ParseArgs(&args);
}

TEST(TapTest, TestSimpleParser) {
  int foo = 0;
  double bar = 0;
  tap::ArgumentParser parser;

  using namespace tap::kw;
  parser.AddArgument("--foo", dest = &foo);
  parser.AddArgument("--bar", dest = &bar);
  std::list<std::string> str_args = {"--foo", "2", "--bar", "2.0"};
  std::list<char*> args;
  for (std::string& str : str_args) {
    args.push_back(&(str[0]));
  }

  parser.ParseArgs(&args);
  EXPECT_EQ(0, args.size());
  EXPECT_EQ(2, foo);
  EXPECT_EQ(2.0, bar);
}

TEST(TapTest, TestSimpleCommandLine) {
  int foo = 0;
  double bar = 0;
  tap::ArgumentParser parser;

  using namespace tap::kw;
  parser.AddArgument("--foo", dest = &foo);
  parser.AddArgument("--bar", dest = &bar);

  char kData[] = "program\0 --foo\0 2\0 --baz\0";
  int argc = 3;
  char* argv[] = {kData, kData + 9, kData + 16, kData + 19};
  parser.ParseArgs(&argc, argv);
  EXPECT_EQ(1, argc);
  EXPECT_STREQ("program", argv[0]);
  EXPECT_EQ(2, foo);
  EXPECT_EQ(0, bar);
}
