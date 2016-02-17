#pragma once

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

namespace kw {

NArgsKW nargs;
HelpKW help;
MetavarKW metavar;
RequiredKW required;

}  // namespace kw`
}  // namespace tap
