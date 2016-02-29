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

TEST(TapTest, TestStoreAction) {
  int foo = 0;
  tap::ArgumentParser parser;

  using namespace tap::kw;
  parser.AddArgument("--foo", action = tap::store, dest = &foo);
  ArgStorage args({"program", "--foo", "2"});

  parser.ParseArgs(&args.argc, args.argv);
  EXPECT_EQ(1, args.argc);
  EXPECT_EQ(2, foo);
}

TEST(TapTest, TestStoreConstAction) {
  int foo = 0;
  tap::ArgumentParser parser;

  using namespace tap::kw;
  parser.AddArgument("--foo", action = tap::store_const, constv = 2,
                     dest = &foo);
  ArgStorage args({"program", "--foo"});

  parser.ParseArgs(&args.argc, args.argv);
  EXPECT_EQ(1, args.argc);
  EXPECT_EQ(2, foo);
}

TEST(TapTest, TestStoreTrueAction) {
  bool foo = false;
  tap::ArgumentParser parser;

  using namespace tap::kw;
  parser.AddArgument("--foo", action = tap::store_true, dest = &foo);
  ArgStorage args({"program", "--foo"});

  parser.ParseArgs(&args.argc, args.argv);
  EXPECT_EQ(1, args.argc);
  EXPECT_TRUE(foo);
}

TEST(TapTest, TestStoreFalseAction) {
  bool foo = true;
  tap::ArgumentParser parser;

  using namespace tap::kw;
  parser.AddArgument("--foo", action = tap::store_false, dest = &foo);
  ArgStorage args({"program", "--foo"});

  parser.ParseArgs(&args.argc, args.argv);
  EXPECT_EQ(1, args.argc);
  EXPECT_FALSE(foo);
}

TEST(TapTest, TestAppendAction) {
  std::list<int> foo;
  tap::ArgumentParser parser;

  using namespace tap::kw;
  parser.AddArgument("--foo", action = tap::store, type = int(), dest = &foo);
  parser.AddArgument("--bar", action = tap::store, type = int(), dest = &foo);
  parser.AddArgument("--baz", action = tap::store, type = int(), dest = &foo);
  ArgStorage args({"program", "--foo", "1", "--bar", "2", "--baz", "3"});

  parser.ParseArgs(&args.argc, args.argv);
  EXPECT_EQ(1, args.argc);
  EXPECT_EQ(std::list<int>({1, 2, 3}), foo);
}

TEST(TapTest, TestAppendConstAction) {
  std::list<int> foo;
  tap::ArgumentParser parser;

  using namespace tap::kw;
  parser.AddArgument("--foo", action = tap::store_const, constv = 1,
                     type = int(), dest = &foo);
  parser.AddArgument("--bar", action = tap::store_const, constv = 2,
                     type = int(), dest = &foo);
  parser.AddArgument("--baz", action = tap::store_const, constv = 3,
                     type = int(), dest = &foo);
  ArgStorage args({"program", "--foo", "--bar", "--baz"});

  parser.ParseArgs(&args.argc, args.argv);
  EXPECT_EQ(1, args.argc);
  EXPECT_EQ(std::list<int>({1, 2, 3}), foo);
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
  std::array<uint32_t, 3> baz = {0, 0, 0};

  tap::ArgumentParser parser;

  using namespace tap;
  using namespace tap::kw;
  parser.AddArgument("-f", "--foo", action = store, type = uint16_t(),
                     dest = &foo);
  parser.AddArgument("-b", "--bar", dest = &bar);
  parser.AddArgument("--baz", action = store, required = false,
                     type = uint16_t(), nargs = 3, choices = {1, 2, 3},
                     dest = &baz);
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
