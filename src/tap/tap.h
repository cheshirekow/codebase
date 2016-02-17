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

NamedActions GetAction() {
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

std::string GetNargsSuffix(int nargs) {
  if (nargs > 1) {
    std::stringstream stream;
    stream << "[" << nargs << "]";
    return stream.str();
  }

  switch (nargs) {
    case NARGS_ZERO_OR_ONE:
      return "?";

    case NARGS_ZERO_OR_MORE:
      return "*";

    case NARGS_ONE_OR_MORE:
      return "+";

    case NARGS_REMAINDER:
      return "^";

    default:
      return "";
  }
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

  void GetUsage(const std::string& argv0, std::ostream* help_out) {
    (*help_out) << argv0 << " ";
    std::list<const Action*> positional_actions;
    std::list<const Action*> flag_actions;
    for (const Action* action : all_actions_) {
      if (action->IsPositional()) {
        positional_actions.push_back(action);
      } else {
        flag_actions.push_back(action);
      }
    }

    for (const Action* action : flag_actions) {
      const Optional<std::string>& short_flag = action->GetShortFlag();
      const Optional<std::string>& long_flag = action->GetLongFlag();
      if (short_flag.is_set && long_flag.is_set) {
        (*help_out) << "[" << short_flag.value << "|" << long_flag.value
                    << "] ";
      } else if (short_flag.is_set) {
        (*help_out) << short_flag.value;
      } else {
        (*help_out) << long_flag.value;
      }
      (*help_out) << "<" << action->GetType().value
                  << GetNargsSuffix(action->GetNargs()) << "> ";
    }

    for (const Action* action : positional_actions) {
      const Optional<std::string>& metavar = action->GetMetavar();
      if (metavar.is_set) {
        (*help_out) << metavar.value;
      } else {
        (*help_out) << "??";
      }
      (*help_out) << "(" << action->GetType().value << ")";
    }
    (*help_out) << "\n";
  }

  void GetHelp(const std::string& argv0, std::ostream* help_out) {
    GetUsage(argv0, help_out);

    const int kNameColumnSize = 20;
    const int kTypeColumnSize = 20;
    const int kHelpColumnSize = 40;

    const std::string kNamePad(kNameColumnSize, ' ');
    const std::string kTypePad(kTypeColumnSize, ' ');
    const std::string kHelpPad(kHelpColumnSize, ' ');

    const int kBufLen = 100;
    char buffer[kBufLen];

    std::list<const Action*> positional_actions;
    std::list<const Action*> flag_actions;
    for (const Action* action : all_actions_) {
      if (action->IsPositional()) {
        positional_actions.push_back(action);
      } else {
        flag_actions.push_back(action);
      }
    }

    if (flag_actions.size() > 0) {
      (*help_out) << "\n\n";
      (*help_out) << "Flags:\n";
      (*help_out) << std::string(kNameColumnSize - 1, '+') << " "
                  << std::string(kTypeColumnSize - 1, '+') << " "
                  << std::string(kHelpColumnSize - 1, '+') << "\n";
      for (const Action* action : flag_actions) {
        const Optional<std::string>& short_flag = action->GetShortFlag();
        const Optional<std::string>& long_flag = action->GetLongFlag();
        int name_chars = 0;
        if (short_flag.is_set && long_flag.is_set) {
          name_chars = short_flag.value.size() + long_flag.value.size() + 2;
          (*help_out) << short_flag.value << ", " << long_flag.value;
        } else if (short_flag.is_set) {
          name_chars = short_flag.value.size();
          (*help_out) << short_flag.value;
        } else {
          name_chars = long_flag.value.size();
          (*help_out) << long_flag.value;
        }
        if (name_chars > kNameColumnSize) {
          (*help_out) << "\n";
          (*help_out) << kNamePad;
        } else {
          (*help_out) << kNamePad.substr(0, kNameColumnSize - name_chars);
        }

        int type_chars = action->GetType().value.size() +
                         GetNargsSuffix(action->GetNargs()).size();
        (*help_out) << action->GetType().value
                    << GetNargsSuffix(action->GetNargs());
        if (type_chars > kTypeColumnSize) {
          (*help_out) << "\n";
          (*help_out) << kNamePad << kTypePad;
        } else {
          (*help_out) << kTypePad.substr(0, kTypeColumnSize - type_chars);
        }

        std::string help_text = action->GetHelp().value;
        char* help_ptr = &help_text[0];
        int help_chars_written = 0;
        for (const char* tok = strtok(help_ptr, " \n"); tok != NULL;
             tok = strtok(NULL, " \n")) {
          std::string word = tok;
          if ((word.size() + help_chars_written) > kHelpColumnSize) {
            (*help_out) << "\n" << kNamePad << kTypePad;
            (*help_out) << word;
            help_chars_written = word.size();
          } else {
            if (help_chars_written > 0) {
              (*help_out) << " ";
              help_chars_written += 1;
            }
            (*help_out) << word;
            help_chars_written += word.size();
          }
        }
        (*help_out) << "\n";
      }
    }

    if (positional_actions.size() > 0) {
      (*help_out) << "\n\n";
      (*help_out) << "Positional Arguments:\n";
      for (const Action* action : flag_actions) {
      }
    }
  }

 private:
  Optional<std::string> description_;
  std::list<Action*> all_actions_;
  std::list<Action*> allocated_actions_;
};

}  // namespace tap
