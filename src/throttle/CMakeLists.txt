set(pkgs glog gflags)

cc_include(${pkgs})
cc_require(${pkgs})

if(required_FOUND)
  cc_executable(ptrace_demo
                HEADERS syscall.h
                SOURCES ptrace_demo.cc
                        syscall.cc
                PKG_DEPENDS ${pkgs}
                TARGET_DEPENDS format_static)

  cc_executable(throttle_server
                SOURCES messages.h server.cc
                PKG_DEPENDS ${pkgs}
                TARGET_DEPENDS format_static)

  cc_executable(pthrottle
                HEADERS syscall.h
                SOURCES pthrottle.cc
                        syscall.cc
                PKG_DEPENDS ${pkgs}
                TARGET_DEPENDS format_static)
else()
  mark_excluded("throttle: missing ${required_MISSING}")
endif()
