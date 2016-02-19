#include <gtest/gtest.h>
#include <tap/tap.h>

/// Construct with a list of strings. Will generate argc, argv suitable for
/// passing into
/// ArgumentParser::ParseArgs.
struct ArgStorage {
  int argc;
  char** argv;

  ArgStorage(std::initializer_list<const char*> args) {
    size_t storage_size = 0;
    for (const std::string& arg : args) {
      storage_size += arg.size() + 1;
    }
    storage_.reserve(storage_size);
    for (const std::string& arg : args) {
      argvec_.push_back(&storage_.back() + 1);
      std::copy(arg.begin(), arg.end(), std::back_inserter(storage_));
      storage_.push_back('\0');
      nullvec_.push_back(&storage_.back());
    }

    for (char* nullval : nullvec_) {
      // *nullval = '\0';
    }

    argc = args.size();
    argv = &argvec_[0];
  }

 private:
  std::string storage_;
  std::vector<char*> argvec_;
  std::vector<char*> nullvec_;
};

TEST(TapTest, TestHash) {
  enum { kHashCT = tap::HashStringCT("HelloWorld") };

  EXPECT_EQ(kHashCT, tap::HashString("HelloWorld"));
}

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

  ArgStorage args({"program", "--foo", "2"});
  parser.ParseArgs(&args.argc, args.argv);
  EXPECT_EQ(1, args.argc);
  EXPECT_STREQ("program", args.argv[0]);
  EXPECT_EQ(2, foo);
  EXPECT_EQ(0, bar);
}

TEST(TapTest, TestFullParser) {
  int32_t foo = 0;
  uint32_t bar = 0;
  uint32_t baz[3] = {0, 0, 0};

  tap::ArgumentParser parser;

  using namespace tap;
  using namespace tap::kw;
  parser.AddArgument("-f", "--foo", action = store, type = uint16_t(),
                     dest = &foo);
  parser.AddArgument("-b", "--bar", dest = &bar);
  parser.AddArgument("--baz", action = store, required = false,
                     type = uint16_t(), nargs = 3, choices = {1, 2, 3},
                     dest = baz);
  ArgStorage args(
      {"program", "--foo", "1", "--bar", "2", "--baz", "1", "2", "3"});
  parser.ParseArgs(&args.argc, args.argv);

  EXPECT_EQ(1, args.argc);
  EXPECT_STREQ("program", args.argv[0]);
  EXPECT_EQ(1, foo);
  EXPECT_EQ(2, bar);
  EXPECT_EQ(1, baz[0]);
  EXPECT_EQ(2, baz[1]);
  EXPECT_EQ(3, baz[2]);
}
