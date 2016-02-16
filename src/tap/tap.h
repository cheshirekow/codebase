#pragma once

#include <functional>
#include <list>
#include <set>
#include <string>

namespace tap {

namespace internal {

template <std::size_t SIZE>
inline constexpr uint64_t HashStringCT(const char(&str)[SIZE], std::size_t i,
                                       uint64_t hash) {
  return i == SIZE ? hash
                   : HashStringCT<SIZE>(str, i + 1,
                                        ((hash << 5) ^ (hash >> 27)) ^ str[i]);
}

}  // namespace internal

template <std::size_t SIZE>
inline constexpr uint64_t HashStringCT(const char(&str)[SIZE]) {
  return internal::HashStringCT<SIZE>(str, 0, 0);
}

inline uint64_t HashString(const std::string& str) {
  uint64_t hash = 0;
  // NOTE(josh): include the null terminating character in the hash
  // so that it matches HashStringCT.
  for (std::size_t i = 0; i <= str.size(); i++) {
    hash = ((hash << 5) ^ (hash >> 27)) ^ str[i];
  }
  return hash;
}

class Action {
 public:
  virtual ~Action();
  virtual void Consume(int* argc, char*** argv) = 0;
};

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

enum NArgs {
  NARGS_N = 0,
  NARGS_ZERO_OR_ONE = '?',
  NARGS_ZERO_OR_MORE = '*',
  NARGS_ONE_OR_MORE = '+',
  NARGS_REMAINDER,
};

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
