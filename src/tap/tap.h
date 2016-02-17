#pragma once

#include <functional>
#include <iterator>
#include <list>
#include <set>
#include <string>
#include <type_traits>

#include "actions.h"
#include "hash.h"

namespace tap {

struct Nil;

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
  typedef Nil Type;
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

template <typename... List>
struct HasNamedAction;

template <>
struct HasNamedAction<> {
  enum { value = 0 };
};

template <typename Head, typename... Tail>
struct HasNamedAction<Head, Tail...> {
  enum { value = HasNamedAction<Tail...>::value };
};

template <typename... Tail>
struct HasNamedAction<NamedActions, Tail...> {
  enum { value = 1 };
};

NamedActions GetAction() {
  return ACTION_NONE;
}

template <typename Head, typename... Tail>
NamedActions GetAction(Head&& head, Tail&&... tail) {
  return GetAction(tail...);
}

template <typename... Tail>
NamedActions GetAction(NamedActions&& head, Tail&&... tail) {
  return head;
}

class ArgumentParser {
 public:
  template <typename... Args>
  void AddArgument(Args&&... args) {
    typedef typename GetIteratorType<Args...>::Type OutputIterator;
    static_assert(
        !std::is_same<Nil, OutputIterator>::value,
        "You failed to specify a desintation for a command line argument.");

    typedef typename GetValueType<Args...>::Type ValueTypeIn;
    typedef
        typename ResolveValueType<ValueTypeIn, OutputIterator>::Type ValueType;

    // now check the action to determine how to proceed
    NamedActions named_action = tap::store;
    if (HasNamedAction<Args...>::value) {
      named_action = GetAction(args...);
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
        break;

      case version:
        break;

      default:
        break;
    }

    all_actions_.push_back(action);
    allocated_actions_.push_back(action);
  }

 private:
  std::list<Action*> all_actions_;
  std::list<Action*> allocated_actions_;
};

}  // namespace tap
