#include "tap/tap.h"

#include <iostream>
#include <map>
#include <set>
#include <cppformat/format.h>
#include <glog/logging.h>

namespace tap {

void Advance(int* argc, char*** argv) {
  *argc -= 1;
  *argv += 1;
}

Parser::~Parser() {}

void Parser::SetCommon(const std::string& short_name,
                       const std::string& long_name, const std::string& help) {
  short_name_ = short_name;
  long_name_ = long_name;
  help_ = help;
}

class BoolParser : public Parser {
 public:
  BoolParser(bool* storage) : storage_(storage) {
    *storage_ = false;
  }

  virtual ~BoolParser() {}
  void Parse(int* argc, char*** argv);

 private:
  bool* storage_;
};

void BoolParser::Parse(int* argc, char*** argv) {
  *storage_ = true;
  return;
}

Parser* MakeParser(bool* storage) {
  return new BoolParser(storage);
}

class StringParser : public Parser {
 public:
  StringParser(std::string* storage) : storage_(storage) {
    *storage_ = "[unset]";
  }
  virtual ~StringParser() {}
  void Parse(int* argc, char*** argv);

 private:
  std::string* storage_;
};

void StringParser::Parse(int* argc, char*** argv) {
  if (*argc < 1) {
    LOG(FATAL)
        << "Ran out of arguments to parse wile reading in value for flag "
        << this->long_name_;
  } else {
    *storage_ = std::string(**argv);
    Advance(argc, argv);
  }
}

Parser* MakeParser(std::string* storage) {
  return new StringParser(storage);
}

ArgumentParser::ArgumentParser(const std::string& description)
    : description_(description) {
  this->AddArgument("-h", "--help", &show_help_, "print this help");
}

void ArgumentParser::ParseArgs(int* argc, char*** argv) {
  std::map<std::string, Parser*> short_name_map;
  std::map<std::string, Parser*> long_name_map;

  for (Parser* parser : named_args_) {
    short_name_map[parser->GetShortName()] = parser;
    long_name_map[parser->GetLongName()] = parser;
  }

  CHECK(argc > 0);
  program_name_ = (*argv)[0];
  Advance(argc, argv);

  while (*argc > 0) {
    if (show_help_) {
      PrintHelp();
      exit(0);
    }

    std::map<std::string, Parser*>::iterator map_iter;
    std::string arg(**argv);
    std::size_t eq_idx = arg.find('=');
    if (arg.substr(0, 2) == "--") {
      if (eq_idx != std::string::npos) {
        arg = arg.substr(eq_idx);
      }

      map_iter = long_name_map.find(arg);
      if (map_iter != long_name_map.end()) {
        Advance(argc, argv);
        map_iter->second->Parse(argc, argv);
        continue;
      } else {
        LOG(FATAL) << "Unrecognized argument " << arg;
      }
    }

    if (arg.substr(0, 1) == "-") {
      if (eq_idx != std::string::npos) {
        arg = arg.substr(eq_idx);
      }

      map_iter = short_name_map.find(arg);
      if (map_iter != short_name_map.end()) {
        Advance(argc, argv);
        map_iter->second->Parse(argc, argv);
        continue;
      } else {
        LOG(FATAL) << "Unrecognized argument " << arg;
      }
    }

    if (positional_args_.size() < 1) {
      LOG(FATAL) << fmt::format(
          "No remaining positional arguments to parse when encountered '{}', "
          "arg='{}'",
          **argv, arg);
    } else {
      Parser* positional_parser = positional_args_.front();
      positional_args_.pop_front();
      positional_parser->Parse(argc, argv);
      continue;
    }
  }
}

void ArgumentParser::PrintHelp() {
  fmt::print("Usage: {}\n", program_name_);
  if (!description_.empty()) {
    fmt::print("\n");
    fmt::print(description_);
    fmt::print("\n");
  }

  std::string fmt_str = "{:2s}  {:15s} {:60}\n";
  for (Parser* parser : named_args_) {
    fmt::print(fmt_str, parser->GetShortName(), parser->GetLongName(),
               parser->GetHelp());
  }
}

}  // namespace tap
