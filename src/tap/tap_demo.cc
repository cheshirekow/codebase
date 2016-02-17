#include <iostream>
#include <glog/logging.h>
#include <tap/tap.h>

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  tap::ArgumentParser parser;
  using namespace tap::kw;
  double foo;
  int bar;
  parser.AddArgument("-f", "--foo", action = tap::store_true, type = double(),
                     dest = &foo);
  parser.AddArgument("-b", "--bar", type = int(), dest = &bar);

  int values[3];
  tap::actions::StoreValue<int, int*> temp_action;
  temp_action.Init1("-f", "--foo", required = false, choices = {1.0, 2.0, 3.0},
                    constv = 0.2f, dest = values, metavar = "hello",
                    help = "hello", nargs = 2);
}
