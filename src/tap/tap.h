#pragma once

#include <list>
#include <string>

namespace tap {

class Parser {
 public:
  virtual ~Parser();
  virtual void Parse(int* argc, char*** argv) = 0;

  void SetCommon(const std::string& short_name, const std::string& long_name,
                 const std::string& help);

  const std::string& GetShortName() {
    return short_name_;
  }

  const std::string& GetLongName() {
    return long_name_;
  }

  const std::string& GetHelp() {
    return help_;
  }

  // TODO(verify that short names are one single dash, one char, long
  // names are double dash an don't include an equals sign
  bool ValidateNames();

 protected:
  Parser& operator=(const Parser& other) = delete;

  std::string short_name_;
  std::string long_name_;
  std::string help_;
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
    arg->SetCommon(short_name, long_name, help);
    named_args_.push_back(arg);
  }

  template <typename ArgType>
  void AddPositional(ArgType* storage, const std::string& help = "") {
    Parser* arg = MakeParser(storage);
    arg->SetCommon("", "", help);
    positional_args_.push_back(arg);
  }

  void ParseArgs(int* argc, char*** argv);
  void PrintHelp();

 private:
  bool show_help_;
  std::string program_name_;
  std::string description_;
  std::list<Parser*> named_args_;
  std::list<Parser*> positional_args_;
};

}  // namespace tap
