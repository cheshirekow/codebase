#pragma once

#include <sys/stat.h>
#include <sys/user.h>
#include <cstdint>
#include <string>

#include <json/json.h>

namespace sys {

// Get a string name for the syscall id (the xxx part of the SYS_xxx macro)
std::string GetName(long syscall_id);

// Return a json object encoding the syscall id, name, and arguments
json::JSON GetCallAsJSON(int child_pid, bool include_output = false);

struct Args {
  virtual ~Args();
  virtual void Decode(const int child_pid, const user_regs_struct& regs) = 0;
  virtual json::JSON GetJSON(bool include_output) = 0;
};

namespace args {

// see: http://man7.org/linux/man-pages/man2/open.2.html
// int open(const char *pathname, int flags);
struct Open : public Args {
  std::string pathname;
  int flags;

  ~Open();
  void Decode(const int child_pid, const user_regs_struct& regs) override;
  json::JSON GetJSON(bool include_output) override;
};

// see: http://man7.org/linux/man-pages/man2/close.2.html
// int close(int fd);
struct Close : public Args {
  int fd;

  ~Close();
  void Decode(const int child_pid, const user_regs_struct& regs) override;
  json::JSON GetJSON(bool include_output) override;
};

// see: http://man7.org/linux/man-pages/man2/stat.2.html
// int stat(const char *pathname, struct stat *buf);
struct Stat : public Args {
  std::string pathname;
  struct stat stat_buf;

  ~Stat();
  void Decode(const int child_pid, const user_regs_struct& regs) override;
  json::JSON GetJSON(bool include_output) override;
};

// see: http://man7.org/linux/man-pages/man2/stat.2.html
// int stat(const char *pathname, struct stat *buf);
struct FStat : public Args {
  int fd;
  struct stat stat_buf;

  ~FStat();
  void Decode(const int child_pid, const user_regs_struct& regs) override;
  json::JSON GetJSON(bool include_output) override;
};

}  // namespace args
}  // namespace syscall_util
