#include <sys/ptrace.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <cppformat/format.h>

extern char** environ;

int main(int argc, char** argv) {
  pid_t child_pid = fork();
  if (child_pid == 0) {
    // this is the child
    long ptrace_result = ptrace(PTRACE_TRACEME, 0, 0, 0);
    int execve_result = execve(argv[1], argv + 1, environ);
    fmt::print(stderr, "Failed to execute {}: [{}] {}\n", argv[1], errno,
               strerror(errno));
    return -1;
  } else {
    // this is the parent
    int wstatus;
    pid_t wait_result = waitpid(child_pid, &wstatus, 0);

  }

  return 0;
}
