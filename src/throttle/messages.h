#pragma once

#include <cstdint>

#include <malloc.h>
#include <linux/limits.h>
#include <sys/types.h>

namespace throttle {

enum MessageType {
  EXECVE_REQUEST,
  EXECVE_RESPONSE,
  FORK_REQUEST,
  FORK_RESPONSE,
  PROCESS_START,
  PROCESS_EXIT
};

struct Header {
  uint8_t message_type;
  uint32_t message_len;
};

struct ExecveRequest {
  // path to the file to execute
  char filename[MAX_PATH];

  uint32_t argc; //< number of elements in argv
  uint32_t envc; //< number of elements in env

  // Concatenation of argv and env. Each element of argv and env is a null terminated string, and
  // the number of elements in each is encoded in argc, and env.
  char argv_env[MAX_ARGV];
};

enum ExecveReponseID {
  EXECVE_BLOCK = 0,
  EXECVE_RELEASE
};

typedef uint8_t ExecveResponse;

enum ForkReponseID {
  FORK_BLOCK = 0,
  FORK_RELEASE
};

typedef uint8_t ForkResponse;

struct ProcessStart {
  pid_t this_pid;
  pid_t parent_pid;
  uint64_t start_time;
};

struct ProcessExit {
  pid_t this_pid;
  uint64_t end_time;

};

}  // namespace throttle
