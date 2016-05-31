#include <sys/ptrace.h>
#include <sys/reg.h>
#include <sys/types.h>
#include <sys/user.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <iostream>
#include <list>
#include <map>
#include <string>
#include <cppformat/format.h>

#include "throttle/syscall.h"

// NOTE(josh): see http://www.linuxjournal.com/article/6100?page=0,1
// NOTE(josh): see http://www.hick.org/code/skape/papers/needle.txt
// for a tutorial on how to execute code in another process and restore
// state (i.e. to allocate a path string).

// Provided by glibc
extern char** environ;

#define MAPENTRY(X) \
  { X, #X }

std::map<int, std::string> kSignoMap = {
    MAPENTRY(SIGHUP),     //
    MAPENTRY(SIGINT),     //
    MAPENTRY(SIGQUIT),    //
    MAPENTRY(SIGILL),     //
    MAPENTRY(SIGABRT),    //
    MAPENTRY(SIGFPE),     //
    MAPENTRY(SIGKILL),    //
    MAPENTRY(SIGSEGV),    //
    MAPENTRY(SIGPIPE),    //
    MAPENTRY(SIGALRM),    //
    MAPENTRY(SIGTERM),    //
    MAPENTRY(SIGUSR1),    //
    MAPENTRY(SIGUSR2),    //
    MAPENTRY(SIGCHLD),    //
    MAPENTRY(SIGCONT),    //
    MAPENTRY(SIGSTOP),    //
    MAPENTRY(SIGTSTP),    //
    MAPENTRY(SIGTTIN),    //
    MAPENTRY(SIGTTOU),    //
    MAPENTRY(SIGBUS),     //
    MAPENTRY(SIGPOLL),    //
    MAPENTRY(SIGPROF),    //
    MAPENTRY(SIGSYS),     //
    MAPENTRY(SIGTRAP),    //
    MAPENTRY(SIGURG),     //
    MAPENTRY(SIGVTALRM),  //
    MAPENTRY(SIGXCPU),    //
    MAPENTRY(SIGXFSZ),
};

std::map<int, std::string> kEventMap = {
    MAPENTRY(PTRACE_EVENT_VFORK),       //
    MAPENTRY(PTRACE_EVENT_FORK),        //
    MAPENTRY(PTRACE_EVENT_CLONE),       //
    MAPENTRY(PTRACE_EVENT_VFORK_DONE),  //
    MAPENTRY(PTRACE_EVENT_EXEC),        //
    MAPENTRY(PTRACE_EVENT_EXIT),        //
    // MAPENTRY(PTRACE_EVENT_STOP),  //
    MAPENTRY(PTRACE_EVENT_SECCOMP),  //

};

std::string GetMapEntry(const std::map<int, std::string>& map, int query) {
  auto iter = kSignoMap.find(query);
  if (iter == kSignoMap.end()) {
    return fmt::format("UNKNOWN[{}]", query);
  } else {
    return iter->second;
  }
}

std::string SignoToString(int signo) {
  return GetMapEntry(kSignoMap, signo);
}

std::string PtraceEventToString(int ptrace_event) {
  return GetMapEntry(kEventMap, ptrace_event);
}

bool IsStoppedForSyscall(int wstatus) {
  if (WIFSTOPPED(wstatus) && WSTOPSIG(wstatus) == SIGTRAP) {
    int ptrace_event = ((wstatus >> 8) & (~SIGTRAP)) >> 8;
    if (ptrace_event) {
      return false;
      // TODO(josh): Shouldn't this be ==? Does PTRACE_O_TRACESYSGOOD not
      // work on this architecture? Best to use PTRACE_GETSIGINFO I guess.
    } else if (WSTOPSIG(wstatus) | (SIGTRAP | 0x80)) {
      return true;
    }
  }
  return false;
}

std::string WaitStatusToString(int wstatus) {
  if (WIFEXITED(wstatus)) {
    return fmt::format("Exited: {}", WEXITSTATUS(wstatus));
  } else if (WIFSIGNALED(wstatus)) {
    return fmt::format("Signaled: {}, coredumpted: {}", WTERMSIG(wstatus),
                       WCOREDUMP(wstatus));
  } else if (WIFSTOPPED(wstatus)) {
    if (WSTOPSIG(wstatus) == SIGTRAP) {
      int ptrace_event = ((wstatus >> 8) & (~SIGTRAP)) >> 8;
      if (ptrace_event) {
        return fmt::format("Stopped on event: {}",
                           PtraceEventToString(ptrace_event));
        // TODO(josh): Shouldn't this be ==? Does PTRACE_O_TRACESYSGOOD not
        // work on this architecture? Best to use PTRACE_GETSIGINFO I guess.
      } else if (WSTOPSIG(wstatus) | (SIGTRAP | 0x80)) {
        return "Stopped on syscall";
      } else {
        return "Stopped on SIGTRAP";
      }
    }
    return fmt::format("Stopped on signal: {}",
                       SignoToString(WSTOPSIG(wstatus)));
  } else if (WIFCONTINUED(wstatus)) {
    return "CONTINUED";
  }
  return "UNKNOWN";
}

std::list<std::string> GetPathComponents() {
  char* path = std::getenv("PATH");
  if (path) {
    std::string path_str(path);
    char* next = &path_str[0];

    std::list<std::string> result;
    for (char* part = strtok(next, ":"); part != nullptr;
         part = strtok(next, ":")) {
      result.push_back(part);
    }
    return result;
  }
  return std::list<std::string>();
}

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
    ptrace(PTRACE_SETOPTIONS, NULL, PTRACE_O_EXITKILL | PTRACE_O_TRACESYSGOOD);

    long last_syscall = -1;
    int wstatus = 0;
    pid_t wait_result = child_pid;
    while (wait_result == child_pid) {
      wait_result = waitpid(child_pid, &wstatus, 0);
      fmt::print("ptrace: {}\n", WaitStatusToString(wstatus));
      if (IsStoppedForSyscall(wstatus)) {
        long this_syscall = sys::GetCallId(child_pid);
        bool on_return = (this_syscall == last_syscall);
        if (on_return) {
          last_syscall = -1;
        } else {
          last_syscall = this_syscall;
        }
        std::cout << sys::GetCallAsJSON(child_pid, on_return).dump(2);
        std::cout << ",\n";
      }

      if (WIFEXITED(wstatus)) {
        break;
      }
      ptrace(PTRACE_SYSCALL, child_pid, NULL, NULL);
    }
    if (wait_result == -1) {
      fmt::print(stderr, "wait() error, errno: {}, {}\n", errno,
                 strerror(errno));
    }
  }

  return 0;
}
