#include "tap/tap.h"

#include <iostream>
#include <map>
#include <set>
#include <cppformat/format.h>
#include <glog/logging.h>

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

void ArgumentParser::GetUsage(const std::string& argv0,
                              std::ostream* help_out) {
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

void ArgumentParser::GetHelp(const std::string& argv0, std::ostream* help_out) {
  GetUsage(argv0, help_out);

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

}  // namespace tap
