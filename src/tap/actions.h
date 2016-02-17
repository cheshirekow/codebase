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
  version
};

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

// Interface for parse actions
class Action {
 public:
  virtual ~Action() {}

 protected:
  Action() : nargs_(1) {}

  // virtual void Consume(int* argc, char*** argv) = 0;
  void ConsumeInit(NArgsSentinel nargs) {
    nargs_ = nargs.value;
  }

  void ConsumeInit(HelpSentinel help) {
    help_ = help.value;
  }

  void ConsumeInit(MetavarSentinel metavar) {
    metavar_ = metavar.value;
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

// Most common action, stores a single value in a variable
template <typename ValueType, typename OutputIterator>
class StoreValue : public ActionBase<ValueType, OutputIterator> {
 public:
  virtual ~StoreValue() {}

  template <typename... Tail>
  void Init(const std::string&& short_flag, const std::string&& long_flag,
            Tail&&... tail) {
    this->short_flag_ = short_flag;
    this->long_flag_ = long_flag;
    InitRest(tail...);
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

  template <typename Head, typename... Tail>
  void InitRest(Head&& head, Tail&&... tail) {
    this->ConsumeInit(head);
    InitRest(tail...);
  }

  //  template <typename... Tail>
  //  void Init(const std::string&& head, Tail&&... tail) {
  //    this->short_flag_ = head;
  //    InitRest(tail...);
  //  }

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
