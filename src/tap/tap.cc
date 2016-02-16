#include "tap/tap.h"

#include <iostream>
#include <map>
#include <set>
#include <cppformat/format.h>
#include <glog/logging.h>

namespace tap {

namespace kw {
const internal::ActionKW action;
} // namespace kw

Action::~Action() {}

template <typename ValueType>
class Choices : public std::set<ValueType> {
 public:
  Choices() {}

  template <typename Container>
  Choices(const Container& container) {
    for (auto& value : container) {
      this->insert(value);
    }
  }
};

template <typename T>
struct Optional {
  bool has_value;
  T value;

  Optional() : has_value(false) {}

  Optional<T>& operator=(const T& value_in) {
    has_value = true;
    value = value_in;
  }
};

template <typename ValueType>
class StoreValueAction : public Action {
 public:
  virtual ~StoreValueAction() {}
  virtual void Consume(int* argc, char*** argv) {}

 private:
  std::list<std::string> option_strings_;
  Optional<ValueType> const_;    /// store this value if the flag is encountered
  Optional<ValueType> default_;  /// store this value if the flag is encountered
                                 /// but no value is set
  std::set<ValueType> choices_;  /// used if a limited set of choices are valid
  bool required_;                /// true if required arg
  std::string help_;             /// help text
  std::string metavar_;          /// used in help text
  ValueType* dest_;              /// where argument is parsed into
};

template <typename ValueType, typename Container>
class StoreMultiValueAction : public Action {
 public:
  virtual ~StoreMultiValueAction() {}
  virtual void Consume(int* argc, char*** argv) {}

 private:
  std::list<std::string> option_strings_;
  Optional<ValueType> const_;    /// store this value if the flag is encountered
  Optional<ValueType> default_;  /// store this value if the flag is encountered
                                 /// but no value is set
  std::set<ValueType> choices_;  /// used if a limited set of choices are valid
  bool required_;                /// true if required arg
  std::string help_;             /// help text
  std::string metavar_;          /// used in help text
  Container* dest_;              /// where argument is parsed into
};

}  // namespace tap
