#pragma once

#include <list>
#include <string>

namespace tap {

class Parser {
 public:
  virtual ~Parser();
  virtual void Parse(int* argc, char*** argv) = 0;

  // TODO(verify that short names are one single dash, one char, long
  // names are double dash an don't include an equals sign
  bool ValidateNames();

  std::string short_name;
  std::string long_name;
  std::string help;

 protected:
  Parser& operator=(const Parser& other) = delete;
};

Parser* MakeParser(bool* storage);
Parser* MakeParser(std::string* storage);

class ArgumentParser {
 public:
  ArgumentParser(const std::string& description = "");

  template <typename ArgType>
  void AddArgument(const std::string& short_name, const std::string& long_name,
                   ArgType* storage, const std::string& help = "") {
    Parser* arg = MakeParser(storage);
    arg->short_name = short_name;
    arg->long_name = long_name;
    arg->help = help;
    named_parsers_.push_back(arg);
  }

  template <typename ArgType>
  void AddPositional(ArgType* storage, const std::string& help = "") {
    Parser* arg = MakeParser(storage);
    arg->help = help;
    positional_parsers_.push_back(arg);
  }

  void ParseArgs(int* argc, char*** argv);
  void PrintHelp();

 private:
  bool show_help_;
  std::string program_name_;
  std::string description_;
  std::list<Parser*> named_parsers_;
  std::list<Parser*> positional_parsers_;
};

}  // namespace tap
