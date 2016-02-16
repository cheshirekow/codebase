#include <iostream>
#include <glog/logging.h>
#include <tap/tap.h>

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  tap::ArgumentParser parser;
  using namespace tap::kw;

  parser.AddArgument("-f", action = "store_true");
  parser.AddArgument("-b", "--bar");
}
