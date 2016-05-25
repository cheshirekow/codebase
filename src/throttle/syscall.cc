#include "throttle/syscall.h"

#include <linux/limits.h>
#include <unistd.h>
#include <sys/ptrace.h>
#include <sys/reg.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/user.h>
#include <unistd.h>
#include <map>

#include <cppformat/format.h>

// NOTE(josh): see http://man7.org/linux/man-pages/man2/syscall.2.html for
// registers used for
// syscall ID and params.
// NOTE(josh): see
// https://github.com/torvalds/linux/blob/master/include/linux/syscalls.h
// for syscall definitions

namespace sys {

#define MAP_ENTRY(X) \
  { X, #X }

static std::map<long, std::string> kSyscallNameMap = {
    // clang-format off
  MAP_ENTRY(SYS_read),
  MAP_ENTRY(SYS_write),
  MAP_ENTRY(SYS_open),
  MAP_ENTRY(SYS_close),
  MAP_ENTRY(SYS_stat),
  MAP_ENTRY(SYS_fstat),
  MAP_ENTRY(SYS_lstat),
  MAP_ENTRY(SYS_poll),
  MAP_ENTRY(SYS_lseek),
  MAP_ENTRY(SYS_mmap),
  MAP_ENTRY(SYS_mprotect),
  MAP_ENTRY(SYS_munmap),
  MAP_ENTRY(SYS_brk),
  MAP_ENTRY(SYS_rt_sigaction),
  MAP_ENTRY(SYS_rt_sigprocmask),
  MAP_ENTRY(SYS_rt_sigreturn),
  MAP_ENTRY(SYS_ioctl),
  MAP_ENTRY(SYS_pread64),
  MAP_ENTRY(SYS_pwrite64),
  MAP_ENTRY(SYS_readv),
  MAP_ENTRY(SYS_writev),
  MAP_ENTRY(SYS_access),
  MAP_ENTRY(SYS_pipe),
  MAP_ENTRY(SYS_select),
  MAP_ENTRY(SYS_sched_yield),
  MAP_ENTRY(SYS_mremap),
  MAP_ENTRY(SYS_msync),
  MAP_ENTRY(SYS_mincore),
  MAP_ENTRY(SYS_madvise),
  MAP_ENTRY(SYS_shmget),
  MAP_ENTRY(SYS_shmat),
  MAP_ENTRY(SYS_shmctl),
  MAP_ENTRY(SYS_dup),
  MAP_ENTRY(SYS_dup2),
  MAP_ENTRY(SYS_pause),
  MAP_ENTRY(SYS_nanosleep),
  MAP_ENTRY(SYS_getitimer),
  MAP_ENTRY(SYS_alarm),
  MAP_ENTRY(SYS_setitimer),
  MAP_ENTRY(SYS_getpid),
  MAP_ENTRY(SYS_sendfile),
  MAP_ENTRY(SYS_socket),
  MAP_ENTRY(SYS_connect),
  MAP_ENTRY(SYS_accept),
  MAP_ENTRY(SYS_sendto),
  MAP_ENTRY(SYS_recvfrom),
  MAP_ENTRY(SYS_sendmsg),
  MAP_ENTRY(SYS_recvmsg),
  MAP_ENTRY(SYS_shutdown),
  MAP_ENTRY(SYS_bind),
  MAP_ENTRY(SYS_listen),
  MAP_ENTRY(SYS_getsockname),
  MAP_ENTRY(SYS_getpeername),
  MAP_ENTRY(SYS_socketpair),
  MAP_ENTRY(SYS_setsockopt),
  MAP_ENTRY(SYS_getsockopt),
  MAP_ENTRY(SYS_clone),
  MAP_ENTRY(SYS_fork),
  MAP_ENTRY(SYS_vfork),
  MAP_ENTRY(SYS_execve),
  MAP_ENTRY(SYS_exit),
  MAP_ENTRY(SYS_wait4),
  MAP_ENTRY(SYS_kill),
  MAP_ENTRY(SYS_uname),
  MAP_ENTRY(SYS_semget),
  MAP_ENTRY(SYS_semop),
  MAP_ENTRY(SYS_semctl),
  MAP_ENTRY(SYS_shmdt),
  MAP_ENTRY(SYS_msgget),
  MAP_ENTRY(SYS_msgsnd),
  MAP_ENTRY(SYS_msgrcv),
  MAP_ENTRY(SYS_msgctl),
  MAP_ENTRY(SYS_fcntl),
  MAP_ENTRY(SYS_flock),
  MAP_ENTRY(SYS_fsync),
  MAP_ENTRY(SYS_fdatasync),
  MAP_ENTRY(SYS_truncate),
  MAP_ENTRY(SYS_ftruncate),
  MAP_ENTRY(SYS_getdents),
  MAP_ENTRY(SYS_getcwd),
  MAP_ENTRY(SYS_chdir),
  MAP_ENTRY(SYS_fchdir),
  MAP_ENTRY(SYS_rename),
  MAP_ENTRY(SYS_mkdir),
  MAP_ENTRY(SYS_rmdir),
  MAP_ENTRY(SYS_creat),
  MAP_ENTRY(SYS_link),
  MAP_ENTRY(SYS_unlink),
  MAP_ENTRY(SYS_symlink),
  MAP_ENTRY(SYS_readlink),
  MAP_ENTRY(SYS_chmod),
  MAP_ENTRY(SYS_fchmod),
  MAP_ENTRY(SYS_chown),
  MAP_ENTRY(SYS_fchown),
  MAP_ENTRY(SYS_lchown),
  MAP_ENTRY(SYS_umask),
  MAP_ENTRY(SYS_gettimeofday),
  MAP_ENTRY(SYS_getrlimit),
  MAP_ENTRY(SYS_getrusage),
  MAP_ENTRY(SYS_sysinfo),
  MAP_ENTRY(SYS_times),
  MAP_ENTRY(SYS_ptrace),
  MAP_ENTRY(SYS_getuid),
  MAP_ENTRY(SYS_syslog),
  MAP_ENTRY(SYS_getgid),
  MAP_ENTRY(SYS_setuid),
  MAP_ENTRY(SYS_setgid),
  MAP_ENTRY(SYS_geteuid),
  MAP_ENTRY(SYS_getegid),
  MAP_ENTRY(SYS_setpgid),
  MAP_ENTRY(SYS_getppid),
  MAP_ENTRY(SYS_getpgrp),
  MAP_ENTRY(SYS_setsid),
  MAP_ENTRY(SYS_setreuid),
  MAP_ENTRY(SYS_setregid),
  MAP_ENTRY(SYS_getgroups),
  MAP_ENTRY(SYS_setgroups),
  MAP_ENTRY(SYS_setresuid),
  MAP_ENTRY(SYS_getresuid),
  MAP_ENTRY(SYS_setresgid),
  MAP_ENTRY(SYS_getresgid),
  MAP_ENTRY(SYS_getpgid),
  MAP_ENTRY(SYS_setfsuid),
  MAP_ENTRY(SYS_setfsgid),
  MAP_ENTRY(SYS_getsid),
  MAP_ENTRY(SYS_capget),
  MAP_ENTRY(SYS_capset),
  MAP_ENTRY(SYS_rt_sigpending),
  MAP_ENTRY(SYS_rt_sigtimedwait),
  MAP_ENTRY(SYS_rt_sigqueueinfo),
  MAP_ENTRY(SYS_rt_sigsuspend),
  MAP_ENTRY(SYS_sigaltstack),
  MAP_ENTRY(SYS_utime),
  MAP_ENTRY(SYS_mknod),
  MAP_ENTRY(SYS_uselib),
  MAP_ENTRY(SYS_personality),
  MAP_ENTRY(SYS_ustat),
  MAP_ENTRY(SYS_statfs),
  MAP_ENTRY(SYS_fstatfs),
  MAP_ENTRY(SYS_sysfs),
  MAP_ENTRY(SYS_getpriority),
  MAP_ENTRY(SYS_setpriority),
  MAP_ENTRY(SYS_sched_setparam),
  MAP_ENTRY(SYS_sched_getparam),
  MAP_ENTRY(SYS_sched_setscheduler),
  MAP_ENTRY(SYS_sched_getscheduler),
  MAP_ENTRY(SYS_sched_get_priority_max),
  MAP_ENTRY(SYS_sched_get_priority_min),
  MAP_ENTRY(SYS_sched_rr_get_interval),
  MAP_ENTRY(SYS_mlock),
  MAP_ENTRY(SYS_munlock),
  MAP_ENTRY(SYS_mlockall),
  MAP_ENTRY(SYS_munlockall),
  MAP_ENTRY(SYS_vhangup),
  MAP_ENTRY(SYS_modify_ldt),
  MAP_ENTRY(SYS_pivot_root),
  MAP_ENTRY(SYS__sysctl),
  MAP_ENTRY(SYS_prctl),
  MAP_ENTRY(SYS_arch_prctl),
  MAP_ENTRY(SYS_adjtimex),
  MAP_ENTRY(SYS_setrlimit),
  MAP_ENTRY(SYS_chroot),
  MAP_ENTRY(SYS_sync),
  MAP_ENTRY(SYS_acct),
  MAP_ENTRY(SYS_settimeofday),
  MAP_ENTRY(SYS_mount),
  MAP_ENTRY(SYS_umount2),
  MAP_ENTRY(SYS_swapon),
  MAP_ENTRY(SYS_swapoff),
  MAP_ENTRY(SYS_reboot),
  MAP_ENTRY(SYS_sethostname),
  MAP_ENTRY(SYS_setdomainname),
  MAP_ENTRY(SYS_iopl),
  MAP_ENTRY(SYS_ioperm),
  MAP_ENTRY(SYS_create_module),
  MAP_ENTRY(SYS_init_module),
  MAP_ENTRY(SYS_delete_module),
  MAP_ENTRY(SYS_get_kernel_syms),
  MAP_ENTRY(SYS_query_module),
  MAP_ENTRY(SYS_quotactl),
  MAP_ENTRY(SYS_nfsservctl),
  MAP_ENTRY(SYS_getpmsg),
  MAP_ENTRY(SYS_putpmsg),
  MAP_ENTRY(SYS_afs_syscall),
  MAP_ENTRY(SYS_tuxcall),
  MAP_ENTRY(SYS_security),
  MAP_ENTRY(SYS_gettid),
  MAP_ENTRY(SYS_readahead),
  MAP_ENTRY(SYS_setxattr),
  MAP_ENTRY(SYS_lsetxattr),
  MAP_ENTRY(SYS_fsetxattr),
  MAP_ENTRY(SYS_getxattr),
  MAP_ENTRY(SYS_lgetxattr),
  MAP_ENTRY(SYS_fgetxattr),
  MAP_ENTRY(SYS_listxattr),
  MAP_ENTRY(SYS_llistxattr),
  MAP_ENTRY(SYS_flistxattr),
  MAP_ENTRY(SYS_removexattr),
  MAP_ENTRY(SYS_lremovexattr),
  MAP_ENTRY(SYS_fremovexattr),
  MAP_ENTRY(SYS_tkill),
  MAP_ENTRY(SYS_time),
  MAP_ENTRY(SYS_futex),
  MAP_ENTRY(SYS_sched_setaffinity),
  MAP_ENTRY(SYS_sched_getaffinity),
  MAP_ENTRY(SYS_set_thread_area),
  MAP_ENTRY(SYS_io_setup),
  MAP_ENTRY(SYS_io_destroy),
  MAP_ENTRY(SYS_io_getevents),
  MAP_ENTRY(SYS_io_submit),
  MAP_ENTRY(SYS_io_cancel),
  MAP_ENTRY(SYS_get_thread_area),
  MAP_ENTRY(SYS_lookup_dcookie),
  MAP_ENTRY(SYS_epoll_create),
  MAP_ENTRY(SYS_epoll_ctl_old),
  MAP_ENTRY(SYS_epoll_wait_old),
  MAP_ENTRY(SYS_remap_file_pages),
  MAP_ENTRY(SYS_getdents64),
  MAP_ENTRY(SYS_set_tid_address),
  MAP_ENTRY(SYS_restart_syscall),
  MAP_ENTRY(SYS_semtimedop),
  MAP_ENTRY(SYS_fadvise64),
  MAP_ENTRY(SYS_timer_create),
  MAP_ENTRY(SYS_timer_settime),
  MAP_ENTRY(SYS_timer_gettime),
  MAP_ENTRY(SYS_timer_getoverrun),
  MAP_ENTRY(SYS_timer_delete),
  MAP_ENTRY(SYS_clock_settime),
  MAP_ENTRY(SYS_clock_gettime),
  MAP_ENTRY(SYS_clock_getres),
  MAP_ENTRY(SYS_clock_nanosleep),
  MAP_ENTRY(SYS_exit_group),
  MAP_ENTRY(SYS_epoll_wait),
  MAP_ENTRY(SYS_epoll_ctl),
  MAP_ENTRY(SYS_tgkill),
  MAP_ENTRY(SYS_utimes),
  MAP_ENTRY(SYS_vserver),
  MAP_ENTRY(SYS_mbind),
  MAP_ENTRY(SYS_set_mempolicy),
  MAP_ENTRY(SYS_get_mempolicy),
  MAP_ENTRY(SYS_mq_open),
  MAP_ENTRY(SYS_mq_unlink),
  MAP_ENTRY(SYS_mq_timedsend),
  MAP_ENTRY(SYS_mq_timedreceive),
  MAP_ENTRY(SYS_mq_notify),
  MAP_ENTRY(SYS_mq_getsetattr),
  MAP_ENTRY(SYS_kexec_load),
  MAP_ENTRY(SYS_waitid),
  MAP_ENTRY(SYS_add_key),
  MAP_ENTRY(SYS_request_key),
  MAP_ENTRY(SYS_keyctl),
  MAP_ENTRY(SYS_ioprio_set),
  MAP_ENTRY(SYS_ioprio_get),
  MAP_ENTRY(SYS_inotify_init),
  MAP_ENTRY(SYS_inotify_add_watch),
  MAP_ENTRY(SYS_inotify_rm_watch),
  MAP_ENTRY(SYS_migrate_pages),
  MAP_ENTRY(SYS_openat),
  MAP_ENTRY(SYS_mkdirat),
  MAP_ENTRY(SYS_mknodat),
  MAP_ENTRY(SYS_fchownat),
  MAP_ENTRY(SYS_futimesat),
  MAP_ENTRY(SYS_newfstatat),
  MAP_ENTRY(SYS_unlinkat),
  MAP_ENTRY(SYS_renameat),
  MAP_ENTRY(SYS_linkat),
  MAP_ENTRY(SYS_symlinkat),
  MAP_ENTRY(SYS_readlinkat),
  MAP_ENTRY(SYS_fchmodat),
  MAP_ENTRY(SYS_faccessat),
  MAP_ENTRY(SYS_pselect6),
  MAP_ENTRY(SYS_ppoll),
  MAP_ENTRY(SYS_unshare),
  MAP_ENTRY(SYS_set_robust_list),
  MAP_ENTRY(SYS_get_robust_list),
  MAP_ENTRY(SYS_splice),
  MAP_ENTRY(SYS_tee),
  MAP_ENTRY(SYS_sync_file_range),
  MAP_ENTRY(SYS_vmsplice),
  MAP_ENTRY(SYS_move_pages),
  MAP_ENTRY(SYS_utimensat),
  MAP_ENTRY(SYS_epoll_pwait),
  MAP_ENTRY(SYS_signalfd),
  MAP_ENTRY(SYS_timerfd_create),
  MAP_ENTRY(SYS_eventfd),
  MAP_ENTRY(SYS_fallocate),
  MAP_ENTRY(SYS_timerfd_settime),
  MAP_ENTRY(SYS_timerfd_gettime),
  MAP_ENTRY(SYS_accept4),
  MAP_ENTRY(SYS_signalfd4),
  MAP_ENTRY(SYS_eventfd2),
  MAP_ENTRY(SYS_epoll_create1),
  MAP_ENTRY(SYS_dup3),
  MAP_ENTRY(SYS_pipe2),
  MAP_ENTRY(SYS_inotify_init1),
  MAP_ENTRY(SYS_preadv),
  MAP_ENTRY(SYS_pwritev),
  MAP_ENTRY(SYS_rt_tgsigqueueinfo),
  MAP_ENTRY(SYS_perf_event_open),
  MAP_ENTRY(SYS_recvmmsg),
  MAP_ENTRY(SYS_fanotify_init),
  MAP_ENTRY(SYS_fanotify_mark),
  MAP_ENTRY(SYS_prlimit64),
  MAP_ENTRY(SYS_name_to_handle_at),
  MAP_ENTRY(SYS_open_by_handle_at),
  MAP_ENTRY(SYS_clock_adjtime),
  MAP_ENTRY(SYS_syncfs),
  MAP_ENTRY(SYS_sendmmsg),
  MAP_ENTRY(SYS_setns),
  MAP_ENTRY(SYS_getcpu),
  MAP_ENTRY(SYS_process_vm_readv),
  MAP_ENTRY(SYS_process_vm_writev),
  MAP_ENTRY(SYS_kcmp),
  MAP_ENTRY(SYS_finit_module),
  MAP_ENTRY(SYS_sched_setattr),
  MAP_ENTRY(SYS_sched_getattr),
  MAP_ENTRY(SYS_renameat2),
  MAP_ENTRY(SYS_seccomp),
    // clang-format on
};

std::string GetName(long syscall_id) {
  auto map_iter = kSyscallNameMap.find(syscall_id);
  if (map_iter == kSyscallNameMap.end()) {
    return "Unknown";
  } else {
    return map_iter->second.substr(4);
  }
}

static json::JSON GetArgsAsJSON(long syscall_id, int child_pid,
                                bool include_output) {
  std::unique_ptr<Args> args;
  switch (syscall_id) {
    case SYS_open: {
      args.reset(new args::Open());
      break;
    }

    case SYS_close: {
      args.reset(new args::Close());
      break;
    }

    case SYS_stat:
    case SYS_lstat: {
      args.reset(new args::Stat());
      break;
    }

    case SYS_fstat: {
      args.reset(new args::FStat());
      break;
    }
  }

  user_regs_struct regs;
  ptrace(PTRACE_GETREGS, child_pid, NULL, &regs);
  if (args) {
    args->Decode(child_pid, regs);
    return args->GetJSON(include_output);
  } else {
    return json::JSON();
  }
}

json::JSON GetCallAsJSON(int child_pid, bool include_output) {
  // NOTE(josh): for 32bit use 4 * ORIG_EAX
  long syscall_id = ptrace(PTRACE_PEEKUSER, child_pid, 8 * ORIG_RAX, NULL);
  json::JSON json_out;
  json_out["syscall_id"] = syscall_id;
  json_out["syscall_name"] = GetName(syscall_id);
  json_out["args"] = GetArgsAsJSON(syscall_id, child_pid, include_output);
  if (include_output) {
    user_regs_struct regs;
    ptrace(PTRACE_GETREGS, child_pid, NULL, &regs);
    json_out["returncode"] = regs.rax;
  }
  return json_out;
}

static bool WordContainsTerminalChar(char* ptr) {
  for (int i = 0; i < sizeof(long); i++) {
    if (ptr[i] == '\0') {
      return true;
    }
  }
  return false;
}

static std::string GetStringAtAddress(const int child_pid,
                                      const char* string_addr) {
  char pathname_buf[PATH_MAX];
  char* write_ptr = pathname_buf;
  for (const char* read_ptr = string_addr; read_ptr < string_addr + PATH_MAX;
       read_ptr += sizeof(long)) {
    *reinterpret_cast<long*>(write_ptr) =
        ptrace(PTRACE_PEEKTEXT, child_pid, read_ptr, NULL);
    if (WordContainsTerminalChar(write_ptr)) {
      break;
    }
    write_ptr += sizeof(long);
  }
  return pathname_buf;
}

Args::~Args() {}

template <class FieldType>
struct StructToJSONImpl {};

template <class FieldType>
json::JSON StructToJSON(FieldType* field_buf) {
  return StructToJSONImpl<FieldType>::Decode(field_buf);
}

#define JSON_NATIVE(field) json_out[#field] = obj_buf->field
#define JSON_STRUCT(field) json_out[#field] = StructToJSON(&(obj_buf->field))

template <>
struct StructToJSONImpl<struct timespec> {
  static json::JSON Decode(struct timespec* obj_buf) {
    json::JSON json_out;

    JSON_NATIVE(tv_sec);
    JSON_NATIVE(tv_nsec);

    return json_out;
  }
};

template <>
struct StructToJSONImpl<struct stat> {
  static json::JSON Decode(struct stat* obj_buf) {
    json::JSON json_out;
    JSON_NATIVE(st_dev);
    JSON_NATIVE(st_ino);
    JSON_NATIVE(st_mode);
    JSON_NATIVE(st_nlink);
    JSON_NATIVE(st_uid);
    JSON_NATIVE(st_gid);
    JSON_NATIVE(st_rdev);
    JSON_NATIVE(st_size);
    JSON_NATIVE(st_blksize);
    JSON_NATIVE(st_blocks);
    JSON_STRUCT(st_atim);
    JSON_STRUCT(st_mtim);
    JSON_STRUCT(st_ctim);

    return json_out;
  }
};

namespace args {

Open::~Open() {}

void Open::Decode(const int child_pid, const user_regs_struct& regs) {
  const char* pathname_addr = reinterpret_cast<const char*>(regs.rdi);
  pathname = GetStringAtAddress(child_pid, pathname_addr);
  flags = static_cast<int>(regs.rsi);
}

json::JSON Open::GetJSON(bool include_output) {
  json::JSON json_out;
  json_out["pathname"] = pathname;
  json_out["flags"] = flags;
  return json_out;
}

Stat::~Stat() {}

void Stat::Decode(const int child_pid, const user_regs_struct& regs) {
  const char* pathname_addr = reinterpret_cast<const char*>(regs.rdi);
  pathname = GetStringAtAddress(child_pid, pathname_addr);
}

json::JSON Stat::GetJSON(bool include_output) {
  json::JSON json_out;
  json_out["pathname"] = pathname;
  if (include_output) {
    json_out["stat_buf"] = StructToJSON(&stat_buf);
  }
  return json_out;
}

Close::~Close() {}

void Close::Decode(const int child_pid, const user_regs_struct& regs) {
  fd = static_cast<int>(regs.rdi);
}

json::JSON Close::GetJSON(bool include_output) {
  json::JSON json_out;
  json_out["fd"] = fd;
  return json_out;
}

FStat::~FStat() {}

void FStat::Decode(const int child_pid, const user_regs_struct& regs) {
  fd = static_cast<int>(regs.rdi);
}

json::JSON FStat::GetJSON(bool include_output) {
  json::JSON json_out;
  json_out["fd"] = fd;
  if (include_output) {
    json_out["stat_buf"] = StructToJSON(&stat_buf);
  }
  return json_out;
}

}  // namespace args
}  // namespace syscall_util
