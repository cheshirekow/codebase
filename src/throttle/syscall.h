#pragma once

#include <sys/ptrace.h>
#include <sys/reg.h>
#include <sys/types.h>
#include <sys/user.h>
#include <unistd.h>

#include <cstdint>
#include <string>

namespace syscall_util {

std::string GetSyscallName(uint16_t syscall_id);

} // namespace syscall_util
