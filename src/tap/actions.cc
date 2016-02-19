#include "actions.h"
#include "tap.h"

namespace tap {
namespace actions {

void Help::ConsumeArgs(ArgumentParser* parser, std::list<char*>* args) {
  parser->GetHelp(&std::cout);
  std::exit(0);
}

}  // namespace actions
}  // namespace tap
