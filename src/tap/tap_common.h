#pragma once

#include <cstdint>
#include <string>

#ifdef __GNUC__
#include <cxxabi.h>
#endif

namespace tap {

enum NamedActions {
  store,
  store_const,
  store_true,
  store_false,
  append,
  append_const,
  count,
  help,
  version,
  ACTION_INVALID,
  ACTION_NONE,
};

enum NArgs {
  NARGS_ZERO_OR_ONE = -1,   // '?'
  NARGS_ZERO_OR_MORE = -2,  // '*'
  NARGS_ONE_OR_MORE = -3,   // '+'
  NARGS_REMAINDER = -4,
};

#ifdef __GNUC__
// Return a human-readable string for the type parameter T
template <typename T>
std::string GetTypeName() {
  std::string result;
  char* name = 0;
  int status;
  name = abi::__cxa_demangle(typeid(T).name(), 0, 0, &status);
  if (name != nullptr) {
    result = name;
  } else {
    result = "[UNKNOWN]";
  }
  free(name);
  return result;
};
#else
// Return a human-readable string for the type parameter T
template <typename _Get_TypeName>
const std::string& GetTypeName() {
  static std::string name;

  if (name.empty()) {
    const char* beginStr = "_Get_TypeName =";
    const size_t beginStrLen = 15;
    size_t begin, length;
    name = __PRETTY_FUNCTION__;

    begin = name.find(beginStr) + beginStrLen + 1;
    length = name.find("]", begin) - begin;
    name = name.substr(begin, length);
  }

  return name;
}
#endif

template <std::size_t SIZE>
inline constexpr uint64_t HashStringCT(const char(&str)[SIZE], std::size_t i,
                                       uint64_t hash) {
  return i == SIZE ? hash
                   : HashStringCT<SIZE>(str, i + 1,
                                        ((hash << 5) ^ (hash >> 27)) ^ str[i]);
}

/// Hash a string at compile time or runtime. Uses Knuth's hash function.
template <std::size_t SIZE>
inline constexpr uint64_t HashStringCT(const char(&str)[SIZE]) {
  return HashStringCT<SIZE>(str, 0, 0);
}

/// Hash a string at compile time. Uses Knuth's hash function.
template <std::size_t SIZE>
inline constexpr uint64_t _H(const char(&str)[SIZE]) {
  return HashStringCT<SIZE>(str, 0, 0);
}

/// Hash a string at runtime, mostly used for testing HashStringCT.
inline uint64_t HashString(const std::string& str) {
  uint64_t hash = 0;
  // NOTE(josh): include the null terminating character in the hash
  // so that it matches HashStringCT.
  for (std::size_t i = 0; i <= str.size(); i++) {
    hash = ((hash << 5) ^ (hash >> 27)) ^ str[i];
  }
  return hash;
}

}  // namespace tap
