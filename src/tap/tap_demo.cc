#include <iostream>
#include <glog/logging.h>
#include <tap/tap.h>

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  tap::ArgumentParser parser;
  using namespace tap::kw;
  parser.AddArgument("-f");
  parser.AddArgument("-b", "--bar");

  int values[3];
  tap::actions::StoreValue<int, int*> action;
  action.Init("-f", "--foo", required=false,
              tap::ChoicesSentinel<double>{{1, 2, 3}},
              tap::ConstSentinel<float>{0.2f}, tap::DestSentinel<int*>{values},
              metavar="hello",
              help="hello",
              nargs=2);
}
