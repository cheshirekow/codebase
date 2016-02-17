#pragma once

#include <list>
#include <set>
#include <string>

#include "kwargs.h"

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
  INVALID_ACTION,
};

struct ActionKW {
  NamedActions operator=(NamedActions action) const {
    return action;
  }
};

namespace kw {
ActionKW action;
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

bool StringStartsWith(const std::string& query, const std::string& start) {
  if (query.size() < start.size()) {
    return false;
  }
  auto query_iter = query.begin();
  auto start_iter = start.begin();
  while (query_iter != query.end() && start_iter != start.end()) {
    if (*query_iter != *start_iter) {
      return false;
    }
    ++query_iter;
    ++start_iter;
  }
  return true;
}

// Interface for parse actions
class Action {
 public:
  virtual ~Action() {}

 protected:
  Action() : nargs_(1) {}

  // virtual void Consume(int* argc, char*** argv) = 0;

  void ConsumeNameOrFlag(const std::string& name_or_flag) {
    if (StringStartsWith(name_or_flag, "--")) {
      long_flag_ = name_or_flag;
    } else if (StringStartsWith(name_or_flag, "-")) {
      short_flag_ = name_or_flag;
    } else {
      metavar_ = name_or_flag;
    }
  }

  // action was already consumed to construct this object
  void ConsumeInit(NamedActions action) {}

  // type was already consumed to construct this object
  template <typename T>
  void ConsumeInit(TypeSentinel<T> type) {}

  void ConsumeInit(NArgsSentinel nargs) {
    nargs_ = nargs.value;
  }

  void ConsumeInit(HelpSentinel help) {
    help_ = help.value;
  }

  void ConsumeInit(MetavarSentinel metavar) {
    metavar_ = metavar.value;
  }

  void SetFlags(const std::string& short_flag, const std::string& long_flag) {
    short_flag_ = short_flag;
    long_flag_ = long_flag;
  }

  Optional<std::string> short_flag_;
  Optional<std::string> long_flag_;
  int nargs_;
  std::string help_;
  std::string metavar_;
};

namespace actions {

template <typename ValueType, typename OutputIterator>
class ActionBase : public Action {
 public:
  virtual ~ActionBase() {}
  using Action::ConsumeInit;

 protected:
  ActionBase() {}

  template <typename T>
  void ConsumeInit(ConstSentinel<T> const_in) {
    const_ = const_in.value;
  }

  void ConsumeInit(DestSentinel<OutputIterator> dest) {
    dest_ = dest.value;
  }

  Optional<ValueType> const_;
  OutputIterator dest_;
};

template <typename... Args>
void NoOp(Args&&... args) {}

// Most common action, stores a single value in a variable
template <typename ValueType, typename OutputIterator>
class StoreValue : public ActionBase<ValueType, OutputIterator> {
 public:
  template <typename... Args>
  StoreValue(Args&&... args) {
    this->Init1(args...);
  }

  virtual ~StoreValue() {}

  template <typename... Tail>
  void Init1(const std::string& name_or_flag, Tail&&... tail) {
    this->ConsumeNameOrFlag(name_or_flag);
    this->Init1(tail...);
  }

  template <typename... Tail>
  void Init1(const char* name_or_flag, Tail&&... tail) {
    this->ConsumeNameOrFlag(name_or_flag);
    this->Init1(tail...);
  }

  template <typename... Tail>
  void Init1(Tail&&... tail) {
    this->InitRest(tail...);
  }

 protected:
  using ActionBase<ValueType, OutputIterator>::ConsumeInit;

  template <typename T>
  void ConsumeInit(ChoicesSentinel<T> choices) {
    for (const auto& choice : choices.value) {
      choices_.insert(choice);
    }
  }

  void ConsumeInit(RequiredSentinel required) {
    required_ = required.value;
  }

  template <typename... Args>
  void InitRest(Args&&... args) {
    NoOp((this->ConsumeInit(args), 0)...);
  }

  void InitRest() {}

 protected:
  std::set<ValueType> choices_;
  bool required_;
};

template <typename ValueType, typename OutputIterator>
class StoreConst : public ActionBase<ValueType, OutputIterator> {
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
class AppendValue : public StoreValue<ValueType, OutputIterator> {
 public:
  virtual ~AppendValue() {}

 protected:
  std::set<ValueType> choices_;
  bool required_;
};

template <typename ValueType, typename OutputIterator>
class AppendConst : public ActionBase<ValueType, OutputIterator> {
 public:
  virtual ~AppendConst() {}
};

}  // namespace actions

}  // namespace tap
