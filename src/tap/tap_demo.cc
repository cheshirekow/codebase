#include <iostream>
#include <glog/logging.h>
#include <tap/tap.h>

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  tap::ArgumentParser parser;
  using namespace tap::kw;
  double foo;
  int bar;
  int baz[3];

  parser.AddArgument("-f", "--foo", action = tap::store_true, type = double(),
                     dest = &foo);
  parser.AddArgument("-b", "--bar", type = int(), dest = &bar);
  parser.AddArgument("-z", "--baz", action = tap::store_true, required = false,
                     type = int(), choices = {1, 2, 3}, constv = 2, nargs = 3,
                     dest = baz, help = "It's a baz... duh");
}
