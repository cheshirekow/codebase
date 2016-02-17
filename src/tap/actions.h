#pragma once

#include <cctype>
#include <list>
#include <set>
#include <string>

#ifdef __GNUC__
#include <cxxabi.h>
#endif

#include "kwargs.h"

namespace tap {

#ifdef __GNUC__
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

struct ActionKW {
  NamedActions operator=(NamedActions action) const {
    return action;
  }
};

namespace kw {
constexpr ActionKW action;
}  // namespace kw

enum NArgs {
  NARGS_ZERO_OR_ONE = -1,   // '?'
  NARGS_ZERO_OR_MORE = -2,  // '*'
  NARGS_ONE_OR_MORE = -3,   // '+'
  NARGS_REMAINDER = -4,
};

template <typename ValueType>
struct Optional {
  ValueType value;
  bool is_set;

  Optional() : is_set(false) {}
  Optional<ValueType>& operator=(const ValueType& value_in) {
    value = value_in;
    is_set = true;
    return *this;
  }
};

bool StringStartsWith(const std::string& query, const std::string& start);
std::string ToUpper(const std::string& val_in);

// Interface for parse actions.
class Action {
 public:
  virtual ~Action() {}

  const Optional<std::string>& GetShortFlag() const {
    return short_flag_;
  }

  const Optional<std::string>& GetLongFlag() const {
    return long_flag_;
  }

  const Optional<std::string>& GetType() const {
    return value_type_name_;
  }

  int GetNargs() const {
    return nargs_;
  }

  const Optional<std::string>& GetHelp() const {
    return help_;
  }

  const Optional<std::string>& GetMetavar() const {
    return metavar_;
  }

  bool IsPositional() const {
    return !(short_flag_.is_set || long_flag_.is_set);
  }

 protected:
  Action() : nargs_(1) {}

  // virtual void Consume(int* argc, char*** argv) = 0;

  void ConsumeNameOrFlag(const std::string& name_or_flag) {
    if (StringStartsWith(name_or_flag, "--")) {
      long_flag_ = name_or_flag;
      if (!metavar_.is_set) {
        metavar_ = ToUpper(name_or_flag.substr(2));
      }
    } else if (StringStartsWith(name_or_flag, "-")) {
      short_flag_ = name_or_flag;
    } else {
      metavar_ = name_or_flag;
    }
  }

  // Action was already consumed in order to determine which derived class was
  // constructed, but we'll store it anyway.
  void ConsumeArgSentinal(NamedActions action) {
    action_ = action;
  }

  // type was already consumed to construct this object. It's still within the
  // parameter pack but we don't need to do anything with it during
  // construction.
  template <typename T>
  void ConsumeArgSentinal(TypeSentinel<T> type) {
    value_type_name_ = GetTypeName<T>();
  }

  void ConsumeArgSentinal(NArgsSentinel nargs) {
    nargs_ = nargs.value;
  }

  void ConsumeArgSentinal(HelpSentinel help) {
    help_ = help.value;
  }

  void ConsumeArgSentinal(MetavarSentinel metavar) {
    metavar_ = metavar.value;
  }

  void SetFlags(const std::string& short_flag, const std::string& long_flag) {
    short_flag_ = short_flag;
    long_flag_ = long_flag;
  }

  /// not actually used, but is passed in during  construction so we may as well
  /// store it as it might be handy to determine derived type later on.
  Optional<NamedActions> action_;
  /// Used mostly for debugging
  Optional<std::string> value_type_name_;
  Optional<std::string> short_flag_;
  Optional<std::string> long_flag_;
  int nargs_;
  Optional<std::string> help_;
  Optional<std::string> metavar_;
};

namespace actions {

// This is an empty function which just allows us to use a parameter pack
// expansion of function calls without a recursive template.
template <typename... Args>
void NoOp(Args&&... args) {}

template <typename Derived>
class ActionInterface : public Action {
 public:
  virtual ~ActionInterface() {}
  using Action::ConsumeArgSentinal;

  template <typename... Tail>
  void Construct(const std::string& name_or_flag, Tail&&... tail) {
    this->ConsumeNameOrFlag(name_or_flag);
    this->Construct(tail...);
  }

  template <typename... Tail>
  void Construct(const char* name_or_flag, Tail&&... tail) {
    this->ConsumeNameOrFlag(name_or_flag);
    this->Construct(tail...);
  }

  template <typename... Tail>
  void Construct(Tail&&... tail) {
    this->InitRest(tail...);
  }

 protected:
  template <typename... Args>
  void InitRest(Args&&... args) {
    Derived* self = static_cast<Derived*>(this);
    NoOp((self->ConsumeArgSentinal(args), 0)...);
  }

  void InitRest() {}
};

// NOTE(josh): CTRP is required to gain access to derived classes' argument
// consumers.
// TODO(josh): add another interface in between this one and Action to provide
// the init function templates
template <typename Derived, typename ValueType, typename OutputIterator>
class ActionBase : public ActionInterface<Derived> {
 public:
  virtual ~ActionBase() {}
  using Action::ConsumeArgSentinal;

  template <typename T>
  void ConsumeArgSentinal(ConstSentinel<T> const_in) {
    const_ = const_in.value;
  }

  void ConsumeArgSentinal(DestSentinel<OutputIterator> dest) {
    dest_ = dest.value;
  }

 protected:
  ActionBase() {}

  Optional<ValueType> const_;
  OutputIterator dest_;
};

// Most common action, stores a single value in a variable
template <typename ValueType, typename OutputIterator>
class StoreValue : public ActionBase<StoreValue<ValueType, OutputIterator>,
                                     ValueType, OutputIterator> {
 public:
  template <typename... Args>
  StoreValue(Args&&... args) {
    this->Construct(args...);
  }

  virtual ~StoreValue() {}

  template <typename T>
  void ConsumeArgSentinal(ChoicesSentinel<T> choices) {
    for (const auto& choice : choices.value) {
      choices_.insert(choice);
    }
  }

  void ConsumeArgSentinal(RequiredSentinel required) {
    required_ = required.value;
  }

  using ActionBase<StoreValue<ValueType, OutputIterator>, ValueType,
                   OutputIterator>::ConsumeArgSentinal;

 protected:
  std::set<ValueType> choices_;
  bool required_;
};

template <typename ValueType, typename OutputIterator>
class StoreConst : public ActionBase<StoreConst<ValueType, OutputIterator>,
                                     ValueType, OutputIterator> {
 public:
  virtual ~StoreConst();
};

class StoreTrue : public StoreConst<bool, bool*> {
 public:
  StoreTrue() {
    nargs_ = 0;
    const_ = true;
  }
  virtual ~StoreTrue();
};

class StoreFalse : public StoreConst<bool, bool*> {
 public:
  StoreFalse() {
    nargs_ = 0;
    const_ = false;
  }
  virtual ~StoreFalse();
};

template <typename ValueType, typename OutputIterator>
class AppendValue : public ActionBase<AppendValue<ValueType, OutputIterator>,
                                      ValueType, OutputIterator> {
 public:
  virtual ~AppendValue() {}

 protected:
  std::set<ValueType> choices_;
  bool required_;
};

template <typename ValueType, typename OutputIterator>
class AppendConst : public ActionBase<AppendConst<ValueType, OutputIterator>,
                                      ValueType, OutputIterator> {
 public:
  virtual ~AppendConst() {}
};

}  // namespace actions

}  // namespace tap
