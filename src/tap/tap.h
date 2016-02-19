#pragma once

#include <cstring>
#include <functional>
#include <iterator>
#include <list>
#include <set>
#include <sstream>
#include <string>
#include <type_traits>

#include "actions.h"
#include "hash.h"

namespace tap {

template <typename... List>
struct GetValueType;

template <>
struct GetValueType<> {
  typedef Nil Type;
  enum { kListExhausted = 1 };
};

template <typename Head, typename... Tail>
struct GetValueType<Head, Tail...> {
  typedef typename GetValueType<Tail...>::Type Type;
  enum { kListExhausted = GetValueType<Tail...>::kListExhausted };
};

template <typename T, typename... Tail>
struct GetValueType<TypeSentinel<T>, Tail...> {
  typedef T Type;
  enum { kListExhausted = 0 };

  // assert that if we keep loking for a type sentinel, we reach the end of the
  // list and end up with the default (i.e. there are no other type sentinels in
  // the list.
  static_assert(
      GetValueType<Tail...>::kListExhausted,
      "You have specified a command line argument with more than one type!");
};

template <typename... List>
struct GetIteratorType;

template <>
struct GetIteratorType<> {
  typedef Nil* Type;
  enum { kListExhausted = 1 };
};

template <typename Head, typename... Tail>
struct GetIteratorType<Head, Tail...> {
  typedef typename GetIteratorType<Tail...>::Type Type;
  enum { kListExhausted = GetIteratorType<Tail...>::kListExhausted };
};

template <typename T, typename... Tail>
struct GetIteratorType<DestSentinel<T>, Tail...> {
  typedef T Type;
  enum { kListExhausted = 0 };

  // assert that if we keep loking for a type sentinel, we reach the end of the
  // list and end up with the default (i.e. there are no other type sentinels in
  // the list.
  static_assert(GetIteratorType<Tail...>::kListExhausted,
                "You have specified a command line argument with more than one "
                "destination!");
};

// This template  resolves the value type of the output iterator if the user
// failed
// to specify a value type for the parser.
template <typename ValueType, typename OutputIterator>
struct ResolveValueType {
  typedef ValueType Type;
};

template <typename OutputIterator>
struct ResolveValueType<Nil, OutputIterator> {
  typedef typename std::iterator_traits<OutputIterator>::value_type Type;
};

template <>
struct ResolveValueType<Nil, Nil> {
  typedef Nil Type;
};

inline NamedActions GetAction() {
  return ACTION_NONE;
}

template <typename... Tail>
NamedActions GetAction(NamedActions head, Tail&&... tail) {
  return head;
}

template <typename Head, typename... Tail>
NamedActions GetAction(Head&& head, Tail&&... tail) {
  return GetAction(tail...);
}

class ArgumentParser {
 public:
  template <typename... Args>
  void AddArgument(Args&&... args) {
    typedef typename GetIteratorType<Args...>::Type OutputIterator;
    typedef typename GetValueType<Args...>::Type ValueTypeIn;
    typedef
        typename ResolveValueType<ValueTypeIn, OutputIterator>::Type ValueType;

    // now check the action to determine how to proceed
    NamedActions named_action = GetAction(args...);
    if (named_action == ACTION_NONE) {
      named_action = tap::store;
    }

    Action* action = nullptr;
    switch (named_action) {
      case store:
        action = new actions::StoreValue<ValueType, OutputIterator>(args...);
        break;

      case store_const:
        break;

      case store_true:
        break;

      case store_false:
        break;

      case append:
        break;

      case append_const:
        break;

      case count:
        break;

      case help:
        action = new actions::Help(args...);
        break;

      case version:
        break;

      default:
        break;
    }

    if (action) {
      all_actions_.push_back(action);
      allocated_actions_.push_back(action);
    }
  }

  void GetUsage(std::ostream* help_out);
  void GetHelp(std::ostream* help_out);
  void ParseArgs(std::list<char*>* args);
  void ParseArgs(int* argc, char** argv);

 private:
  Optional<std::string> description_;
  Optional<std::string> argv0_;
  std::list<Action*> all_actions_;
  std::list<Action*> allocated_actions_;
};

}  // namespace tap
