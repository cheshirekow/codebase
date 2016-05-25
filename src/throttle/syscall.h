#pragma once

#include <sys/stat.h>
#include <sys/user.h>
#include <cstdint>
#include <string>

namespace sys {

std::string GetName(long syscall_id);
std::string GetJSON(int child_pid);

struct Call {
  virtual ~Call();
  virtual void Decode(const int child_pid, const user_regs_struct& regs) = 0;
  virtual std::string GetArgsJSON() = 0;
};

// see: http://man7.org/linux/man-pages/man2/open.2.html
// int open(const char *pathname, int flags);
struct Open : public Call {
  std::string pathname;
  int flags;

  ~Open();
  void Decode(const int child_pid, const user_regs_struct& regs) override;
  std::string GetArgsJSON() override;
};

// see: http://man7.org/linux/man-pages/man2/close.2.html
// int close(int fd);
struct Close : public Call {
  int fd;

  ~Close();
  void Decode(const int child_pid, const user_regs_struct& regs) override;
  std::string GetArgsJSON() override;
};

// see: http://man7.org/linux/man-pages/man2/stat.2.html
// int stat(const char *pathname, struct stat *buf);
struct Stat : public Call {
  std::string pathname;
  struct stat stat_buf;

  ~Stat();
  void Decode(const int child_pid, const user_regs_struct& regs) override;
  std::string GetArgsJSON() override;
};

// see: http://man7.org/linux/man-pages/man2/stat.2.html
// int stat(const char *pathname, struct stat *buf);
struct FStat : public Call {
  int fd;
  struct stat stat_buf;

  ~FStat();
  void Decode(const int child_pid, const user_regs_struct& regs) override;
  std::string GetArgsJSON() override;
};

// see: http://man7.org/linux/man-pages/man2/stat.2.html
// int lstat(const char *pathname, struct stat *buf);
struct LStat : public Call {
  std::string pathname;
  struct stat stat_buf;

  ~LStat();
  void Decode(const int child_pid, const user_regs_struct& regs) override;
  std::string GetArgsJSON() override;
};

}  // namespace syscall_util
