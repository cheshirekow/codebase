#pragma once

#include <initializer_list>
#include <list>
#include <string>

#include "tap_common.h"

namespace tap {

template <uint64_t Key, typename T>
struct Sentinel {
  T value;
};

template <uint64_t Key>
struct KeyWord {
  template <typename T>
  Sentinel<Key, T> operator=(T&& value) const {
    return Sentinel<Key, T>{value};
  }
};

template <uint64_t Key, typename T>
struct TypedKeyWord {
  Sentinel<Key, T> operator=(T&& value) const {
    return Sentinel<Key, T>{value};
  }
};

template <typename T>
struct DestSentinel {
  T value;
};

template <typename T>
struct ChoicesSentinel {
  std::list<T> value;
};

typedef KeyWord<_H("nargs")> NArgsKW;
typedef KeyWord<_H("help")> HelpKW;
typedef KeyWord<_H("metavar")> MetavarKW;
typedef KeyWord<_H("required")> RequiredKW;
typedef KeyWord<_H("type")> TypeKW;
typedef KeyWord<_H("const")> ConstKW;

struct ChoicesKW {
  template <typename T>
  ChoicesSentinel<T> operator=(const std::initializer_list<T>& value) const {
    return ChoicesSentinel<T>{value};
  }
};

struct DestKW {
  template <typename OutputIterator>
  DestSentinel<OutputIterator> operator=(OutputIterator value) const {
    return DestSentinel<OutputIterator>{value};
  }
};

struct ActionKW {
  NamedActions operator=(NamedActions action) const {
    return action;
  }
};

namespace kw {

constexpr ChoicesKW choices;
constexpr DestKW dest;
constexpr NArgsKW nargs;
constexpr HelpKW help;
constexpr MetavarKW metavar;
constexpr RequiredKW required;
constexpr ConstKW constv;
constexpr TypeKW type;
constexpr ActionKW action;

}  // namespace kw`
}  // namespace tap
