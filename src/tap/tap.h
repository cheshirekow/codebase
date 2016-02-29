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
#include "container_id.h"
#include "common.h"

namespace tap {

/// Given a typelist of argument types to the function
/// ArgumentParser::AddArgument, resolve the input value type (as declared by
/// type=x()) if it was specified, as well as the output value type which is the
/// type of the destination object, or the value_type if that object is a
/// container.
template <typename... List>
struct ParseTypes;

/// There are no more arguments to parse, so "return" Nil as the types.
template <>
struct ParseTypes<> {
  typedef Nil InputValueType;
  typedef Nil OutputValueType;
};

/// The next type in the list is not one of the sentinel types we care about, so
/// just recurse
template <typename Head, typename... Tail>
struct ParseTypes<Head, Tail...> {
  typedef ParseTypes<Tail...> Next;
  typedef typename Next::InputValueType InputValueType;
  typedef typename Next::OutputValueType OutputValueType;
};

/// The next type in the typelist is a type sentinel, this tells us the
/// user-specified parse-as type, so record it.
template <typename T, typename... Tail>
struct ParseTypes<Sentinel<_H("type"), T>, Tail...> {
  typedef ParseTypes<Tail...> Next;

  /// do not allow type to show up twice in the arguments
  static_assert(
      std::is_same<Nil, typename Next::InputValueType>::value,
      "You have called AddArgument and specified type= more than once.");

  typedef T InputValueType;
  typedef typename Next::OutputValueType OutputValueType;
};

/// The next type in the typelist is a dest sentinel, this tells us the
/// output_type type, so record it.
template <typename T, typename... Tail>
struct ParseTypes<Sentinel<_H("dest"), T*>, Tail...> {
  typedef ParseTypes<Tail...> Next;

  /// do not allow type to show up twice in the arguments
  static_assert(
      std::is_same<Nil, typename Next::OutputValueType>::value,
      "You have called AddArgument and specified dest= more than once.");

  typedef typename Next::InputValueType InputValueType;

  // if T is a container, then this will resolve it's value type. Otherwise, it
  // resolves T.
  typedef typename get_value_type<T>::value_type OutputValueType;
};

/// Template meta function resolves the first type in this typelist which is not
/// Nil.
template <typename... List>
struct FirstNotNil {};

template <>
struct FirstNotNil<> {
  typedef Nil Type;
};

template <typename... Tail>
struct FirstNotNil<Nil, Tail...> {
  typedef typename FirstNotNil<Tail...>::Type Type;
};

template <typename Head, typename... Tail>
struct FirstNotNil<Head, Tail...> {
  typedef Head Type;
};

/// Given a typelist of argument types to the function
/// ArgumentParser::AddArgument, determine
/// if any of the arguments is a TypeSentinel. If so, extract the ValueType
/// stored in the sentinel.
/// Otherwise resolve to Nil.
template <typename... List>
struct GetValueType {
  typedef ParseTypes<List...> ParsedTypes;
  typedef
      typename FirstNotNil<typename ParsedTypes::InputValueType,
                           typename ParsedTypes::OutputValueType>::Type Type;
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
    typedef Nil OutputIterator;
    typedef typename GetValueType<Args...>::Type ValueType;

    // now check the action to determine how to proceed
    NamedActions named_action = GetAction(args...);
    if (named_action == ACTION_NONE) {
      named_action = tap::store;
    }

    Action* action = nullptr;
    switch (named_action) {
      case store:
        action = new actions::StoreValue<ValueType>(args...);
        break;

      case store_const:
        action = new actions::StoreConst<ValueType>(args...);
        break;

      case store_true:
        action = new actions::StoreConst<bool>(args..., kw::constv = true);
        break;

      case store_false:
        action = new actions::StoreConst<bool>(args..., kw::constv = false);
        break;

      case append:
        // TODO(assert output iterator has size)
        action = new actions::StoreValue<ValueType>(args...);
        break;

      case append_const:
        // TODO(assert output iterator has size)
        action = new actions::StoreConst<ValueType>(args...);
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
