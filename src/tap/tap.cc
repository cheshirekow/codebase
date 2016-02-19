#include <iostream>
#include <list>
#include <map>
#include <set>
#include <cppformat/format.h>
#include <glog/logging.h>

#include "tap.h"
#include "actions.h"

namespace tap {

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

void ArgumentParser::GetUsage(std::ostream* help_out) {
  if (argv0_.is_set) {
    (*help_out) << argv0_.value << " ";
  } else {
    (*help_out) << "[program] ";
  }
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
      (*help_out) << "[" << short_flag.value << "|" << long_flag.value << "] ";
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

void ArgumentParser::GetHelp(std::ostream* help_out) {
  GetUsage(help_out);

  const int kNameColumnSize = 16;
  const int kTypeColumnSize = 16;
  const int kHelpColumnSize = 48;

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

void ArgumentParser::ParseArgs(int* argc, char** argv) {
  std::list<char*> args;

  if (*argc > 0) {
    argv0_ = argv[0];
  }

  for (int i = 1; i < *argc; ++i) {
    args.push_back(argv[i]);
  }

  ParseArgs(&args);

  if (args.size() > 0) {
    std::cerr << "Not all arguments were consumed. remaining args: ";
    for (char* arg : args) {
      std::cerr << arg << " ";
    }
    std::cerr << "\n";
    // TODO(josh): exit depending on options for the parser
  }

  for (*argc = 1; args.size() > 0; (*argc)++) {
    argv[*argc] = args.front();
    args.pop_front();
  }
}

void ArgumentParser::ParseArgs(std::list<char*>* args) {
  if (args->size() < 1) {
    return;
  }

  // TODO(josh): validate the parser before the following, as it makes
  // certain assumptions:
  //    short_flags are '-x'
  //    long_flags are '--xxx'
  //    flags are not repeated

  // build a list of positional arguments and a map for short and long
  // flag names
  std::list<Action*> positional_actions;
  std::map<char, Action*> short_flags;
  std::map<std::string, Action*> long_flags;

  for (Action* action : all_actions_) {
    if (action->IsPositional()) {
      positional_actions.push_back(action);
    } else {
      if (action->GetShortFlag().is_set) {
        char short_flag = action->GetShortFlag().value[1];
        short_flags[short_flag] = action;
      }
      if (action->GetLongFlag().is_set) {
        std::string long_flag = action->GetLongFlag().value.substr(2);
        long_flags[long_flag] = action;
      }
    }
  }

  // now walk through the argument list and run our parsers.
  while (args->size() > 0) {
    if (IsShortFlag(args->front())) {
      std::string flag_chars = std::string(args->front()).substr(1);
      args->pop_front();
      for (char c : flag_chars) {
        auto iter = short_flags.find(c);
        if (iter != short_flags.end()) {
          iter->second->ConsumeArgs(this, args);
        } else {
          std::cerr << "Uknown short flag " << c << "\n";
          std::exit(1);
        }
      }
    } else if (IsLongFlag(args->front())) {
      std::string flag_name = std::string(args->front()).substr(2);
      args->pop_front();
      auto iter = long_flags.find(flag_name);
      if (iter != long_flags.end()) {
        iter->second->ConsumeArgs(this, args);
      } else {
        std::cerr << "Uknown long flag " << flag_name << "\n";
        std::exit(1);
      }
    } else {
      if (positional_actions.size() > 0) {
        Action* action = positional_actions.front();
        positional_actions.pop_front();
        action->ConsumeArgs(this, args);
      } else {
        std::cerr << "No positional actions available to parse argument: "
                  << args->front() << "\n";
        std::exit(1);
      }
    }
  }
}

}  // namespace tap
