#include <iostream>
#include <glog/logging.h>
#include <tap/tap.h>

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  tap::ArgumentParser parser;
  using namespace tap::kw;
  double foo = 1.0;
  int bar = 2;
  int baz[3] = {1, 2, 3};

  parser.AddArgument("-f", "--foo", action = tap::store_true, type = double(),
                     dest = &foo);
  parser.AddArgument("-b", "--bar", type = int(), dest = &bar);
  parser.AddArgument(
      "-z", "--baz", action = tap::store, required = false, type = int(),
      choices = {1, 2, 3}, constv = 2, nargs = 3, dest = baz,
      help =
          "It's a baz... duh. This is some aburdly long help text, by the way. "
          "This is some aburdly long help text, by the way. This is some "
          "aburdly long help text, by the way. This is some aburdly long help "
          "text, by the way. This is some aburdly long help text, by the way. "
          "This is some aburdly long help text, by the way. This is some "
          "aburdly long help text, by the way. This is some aburdly long help "
          "text, by the way. This is some aburdly long help text, by the "
          "way. ");
  parser.AddArgument("-h", "--help", action = tap::help);
  parser.ParseArgs(&argc, &argv);

  std::cout << "foo : " << foo << "\n"
            << "bar : " << bar << "\n"
            << "baz : " << baz[0] << ", " << baz[1] << ", " << baz[2] << "\n";
}
