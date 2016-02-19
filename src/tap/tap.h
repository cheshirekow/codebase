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
#include "common.h"

namespace tap {

/// Given a typelist of argument types to the function
/// ArgumentParser::AddArgument, determine
/// if any of the arguments is a TypeSentinel. If so, extract the ValueType
/// stored in the sentinel.
/// Otherwise resolve to Nil.
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
struct GetValueType<Sentinel<_H("type"), T>, Tail...> {
  typedef T Type;
  enum { kListExhausted = 0 };

  // assert that if we keep loking for a type sentinel, we reach the end of the
  // list and end up with the default (i.e. there are no other type sentinels in
  // the list.
  static_assert(
      GetValueType<Tail...>::kListExhausted,
      "You have specified a command line argument with more than one type!");
};

/// Given a typelist of argument types to the function
/// ArgumentParser::AddArgument, determine if any of the arguments is a
/// DestSentinel. If so, extract the OutputIterator type stored in the sentinel.
/// Otherwise, resolve to Nil.
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
  // list and end up with the default (i.e. there are no other dest sentinels in
  // the list.
  static_assert(GetIteratorType<Tail...>::kListExhausted,
                "You have specified a command line argument with more than one "
                "destination!");
};

/// This template  resolves the value type of the output iterator if the user
/// failed to specify a value type for the parser.
template <typename ValueType, typename OutputIterator>
struct ResolveValueType {
  typedef ValueType Type;
};

template <typename OutputIterator>
struct ResolveValueType<Nil, OutputIterator> {
  typedef typename std::iterator_traits<OutputIterator>::value_type Type;
};

template <typename Container>
struct ResolveValueType<Nil, std::back_insert_iterator<Container>> {
  typedef typename Container::value_type Type;
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

/// Template to extract the action type from a parameter pack. Recursively calls
/// itself until either a NamedAction is encountered or the list is exhausted.
template <typename Head, typename... Tail>
NamedActions GetAction(Head&& head, Tail&&... tail) {
  return GetAction(tail...);
}

/// Primary interface into TAP. Stores a specification of command line flags and
/// arguments, and provides methods for parsing and argument list and generating
/// help text.
class ArgumentParser {
 public:
  /// KWargs api to add an argument.
  /**
   *  positional arguments:
   *    * **name or flags**  one or two strings (or objects implicitly
   *      convertable to a string). Can be one of the following:
   *      * **short flag** i.e. "-x"
   *      * **long flag** i.e. "--foo"
   *      * **argument name** indicates the argument is a positional argument,
   *        and provides the metavar used as a placeholder for this argument
   *        in help text.
   *  kwargs:
   *    * **action** - which action to perform when this argument is
   *      encountered.
   *    * **nargs** - How many arguments to parse. May be an integer or one
   *      of the following.
   *    * **constv** - a constant value used by some of the actions
   *    * **type** - the value type of the data used to parse the string
   *    * **choices** - enumeration of valid values for the argument
   *    * **required** - whether or not the argument is required
   *    * **help** - help text for this argument
   *    * **metavar** - placeholder name used in help output
   *    * **dest** - output iterator where to store the value(s) when
   *      encountered on the command line
   */
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
        action = new actions::StoreConst<ValueType, OutputIterator>(args...);
        break;

      case store_true:
        action = new actions::StoreConst<bool, OutputIterator>(
            args..., kw::constv = true);
        break;

      case store_false:
        action = new actions::StoreConst<bool, OutputIterator>(
            args..., kw::constv = false);
        break;

      case append:
        // TODO(assert output iterator has size)
        action = new actions::StoreValue<ValueType, OutputIterator>(args...);
        break;

      case append_const:
        // TODO(assert output iterator has size)
        action = new actions::StoreConst<ValueType, OutputIterator>(args...);
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
