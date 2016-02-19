#include "tap_common.h"

namespace tap {

bool StringStartsWith(const std::string& query, const std::string& start) {
  if (query.size() < start.size()) {
    return false;
  }
  auto query_iter = query.begin();
  auto start_iter = start.begin();
  while (query_iter != query.end() && start_iter != start.end()) {
    if (*query_iter != *start_iter) {
      return false;
    }
    ++query_iter;
    ++start_iter;
  }
  return true;
}

std::string ToUpper(const std::string& val_in) {
  std::string val_out;
  val_out.reserve(val_in.size());
  for (char c : val_in) {
    val_out.push_back(toupper(c));
  }

  return val_out;
}

}  // namespace tap
