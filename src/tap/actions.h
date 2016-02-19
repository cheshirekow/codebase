#pragma once

#include <cctype>
#include <iostream>
#include <list>
#include <set>
#include <string>

#ifdef __GNUC__
#include <cxxabi.h>
#endif

#include "common.h"
#include "kwargs.h"
#include "value_parsers.h"

namespace tap {

class ArgumentParser;

// Interface for parse actions.
class Action {
 public:
  virtual ~Action() {}
  virtual void ConsumeArgs(ArgumentParser* parser, std::list<char*>* args) = 0;

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

  /// Action was already consumed in order to determine which derived class was
  /// constructed, but we'll store it anyway.
  void ConsumeArgSentinal(NamedActions action) {
    action_ = action;
  }

  /// Value type was already consumed to construct this object. We store the
  /// string representation though for help text and debugging.
  template <typename T>
  void ConsumeArgSentinal(Sentinel<_H("type"), T> type) {
    value_type_name_ = GetTypeName<T>();
  }

  template <typename T>
  void ConsumeArgSentinal(Sentinel<_H("nargs"), T> nargs) {
    nargs_ = nargs.value;
  }

  template <typename T>
  void ConsumeArgSentinal(Sentinel<_H("help"), T> help) {
    help_ = help.value;
  }

  template <typename T>
  void ConsumeArgSentinal(Sentinel<_H("metavar"), T> metavar) {
    metavar_ = metavar.value;
  }

  // These are default implementations which ignore the parameter, and are
  // overridden by derived classes. They are overridden by CTRP resolution
  // of the functions.
  // TODO(josh): warn if these are ever called
  template <typename T>
  void ConsumeArgSentinal(Sentinel<_H("const"), T> const_in) {}

  template <typename OutputIterator>
  void ConsumeArgSentinal(DestSentinel<OutputIterator> dest) {}

  template <typename T>
  void ConsumeArgSentinal(ChoicesSentinel<T> choices) {}

  template <typename T>
  void ConsumeArgSentinal(Sentinel<_H("required"), T> required) {}

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

// First base class just provides static interface. CTRP is required to gain
// access to derived class's argument consumers.
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

namespace actions {

// Second base class provides some actual storage that is common among some
// actions
template <typename Derived, typename ValueType, typename OutputIterator>
class ActionBase : public ActionInterface<Derived> {
 public:
  virtual ~ActionBase() {}
  using Action::ConsumeArgSentinal;

  template <typename T>
  void ConsumeArgSentinal(Sentinel<_H("const"), T> const_in) {
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

  void ConsumeArgs(ArgumentParser* parser, std::list<char*>* args) override {
    ValueType value;
    // TODO(josh): assert nargs_ > -5
    switch (this->nargs_) {
      case NARGS_ONE_OR_MORE:
        // TODO(josh): assert !IsFlag(args.front());
        while (args->size() > 0 && !IsFlag(args->front())) {
          int result = ParseValue(args->front(), &value);
          // TODO(josh): handle result
          *(this->dest_++) = value;
          args->pop_front();
        }
        break;

      case NARGS_REMAINDER:
        while (args->size() > 0) {
          int result = ParseValue(args->front(), &value);
          // TODO(josh): handle result
          *(this->dest_++) = value;
          args->pop_front();
        }
        break;

      case NARGS_ZERO_OR_MORE:
        while (args->size() > 0 && !IsFlag(args->front())) {
          int result = ParseValue(args->front(), &value);
          // TODO(josh): handle result
          *(this->dest_++) = value;
          args->pop_front();
        }
        break;

      case NARGS_ZERO_OR_ONE:
        if (args->size() > 0 && !IsFlag(args->front())) {
          int result = ParseValue(args->front(), &value);
          // TODO(josh): handle result
          *(this->dest_++) = value;
          args->pop_front();
        }
        break;

      default:
        // TODO(josh): assert args->size() > nargs_
        for (int i = 0; i < this->nargs_ && args->size() > 0; i++) {
          int result = ParseValue(args->front(), &value);
          // TODO(josh): handle result
          *(this->dest_++) = value;
          args->pop_front();
        }
    }
  }

  template <typename T>
  void ConsumeArgSentinal(ChoicesSentinel<T> choices) {
    for (const auto& choice : choices.value) {
      choices_.insert(choice);
    }
  }

  template <typename T>
  void ConsumeArgSentinal(Sentinel<_H("required"), T> required) {
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
  using ActionBase<StoreConst<ValueType, OutputIterator>, ValueType,
                   OutputIterator>::ConsumeArgSentinal;

  template <typename... Args>
  StoreConst(Args&&... args) {
    this->nargs_ = 0;
    this->Construct(args...);
  }

  virtual ~StoreConst() {}

  void ConsumeArgs(ArgumentParser* parser, std::list<char*>* args) override {
    // TODO(josh): assert nargs_ == 1;
    // TODO(josh): assert args->size() > nargs_
    // TODO(josh): assert const_.is_set;
    if (this->const_.is_set) {
      *(this->dest_++) = this->const_.value;
    }
  }
};

class Help : public ActionInterface<Help> {
 public:
  using ActionInterface<Help>::ConsumeArgSentinal;

  template <typename... Args>
  Help(Args&&... args) {
    this->Construct(args...);
  }

  virtual ~Help() {}
  void ConsumeArgs(ArgumentParser* parser, std::list<char*>* args) override;
};

}  // namespace actions
}  // namespace tap
