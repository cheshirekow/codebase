#pragma once

#include <initializer_list>
#include <list>
#include <string>

namespace tap {

struct NArgsSentinel {
  int value;
};

struct HelpSentinel {
  std::string value;
};

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

struct NArgsKW {
  NArgsSentinel operator=(int value) const {
    return NArgsSentinel{value};
  }
};

struct HelpKW {
  HelpSentinel operator=(const std::string& value) const {
    return HelpSentinel{value};
  }
};

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
