#pragma once

#include <functional>
#include <list>
#include <set>
#include <string>

#include "actions.h"
#include "hash.h"

namespace tap {




namespace internal {

template <uint64_t Key>
struct Tag {};

template <uint64_t ACTION_ID>
struct ActionTypeDispatch {};

struct ActionKW {
  template <std::size_t SIZE>
  constexpr uint64_t operator=(const char(&str)[SIZE]) {
    return tap::HashStringCT(str);
  }
};

}  // namespace internal

namespace kw {
extern const internal::ActionKW action;
}  // namespace kw

class ArgumentParser {
 public:
  template <typename... Args>
  void AddArgument(const std::string& name, Args&&... args) {}

  template <typename... Args>
  void AddArgument(const std::string& short_name, const std::string& long_name,
                   Args&&... args) {}

 private:
  std::list<Action*> all_actions_;
  std::list<Action*> allocated_actions_;
};

}  // namespace tap
