#pragma once

#include <initializer_list>
#include <list>
#include <string>
#include "hash.h"

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

typedef Sentinel<_H("nargs"), int> NArgsSentinel;
typedef Sentinel<_H("help"), std::string> HelpSentinel;

struct MetavarSentinel {
  std::string value;
};

template <typename T>
struct ConstSentinel {
  T value;
};

template <typename T>
struct DestSentinel {
  T value;
};

template <typename T>
struct ChoicesSentinel {
  std::list<T> value;
};

struct RequiredSentinel {
  bool value;
};

template <typename T>
struct TypeSentinel {};

typedef KeyWord<_H("nargs")> NArgsKW;
typedef KeyWord<_H("help")> HelpKW;

struct MetavarKW {
  MetavarSentinel operator=(const std::string& value) const {
    return MetavarSentinel{value};
  }
};

struct RequiredKW {
  RequiredSentinel operator=(bool value) const {
    return RequiredSentinel{value};
  }
};

struct ChoicesKW {
  template <typename T>
  ChoicesSentinel<T> operator=(const std::initializer_list<T>& value) const {
    return ChoicesSentinel<T>{value};
  }
};

struct ConstKW {
  template <typename T>
  ConstSentinel<T> operator=(const T& value) const {
    return ConstSentinel<T>{value};
  }
};

struct DestKW {
  template <typename OutputIterator>
  DestSentinel<OutputIterator> operator=(OutputIterator value) const {
    return DestSentinel<OutputIterator>{value};
  }
};

struct TypeKW {
  template <typename T>
  TypeSentinel<T> operator=(const T& value) const {
    return TypeSentinel<T>{};
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

}  // namespace kw`
}  // namespace tap
