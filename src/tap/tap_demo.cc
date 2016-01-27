#include <iostream>
#include <glog/logging.h>
#include <tap/tap.h>

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  tap::ArgumentParser parser;

  std::string foo;
  std::string bar;
  parser.AddArgument("-f", "--foo", &foo, "foo help");
  parser.AddArgument("-b", "--bar", &bar, "bar help");
  parser.ParseArgs(&argc, &argv);

  std::cout << "Foo: " << foo << "\n"
            << "Bar: " << bar << "\n";
}
