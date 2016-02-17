#pragma once

#include <cstdint>
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

}  // namespace tap
