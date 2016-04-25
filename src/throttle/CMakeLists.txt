set(pkgs glog gflags)

cc_include(${pkgs})
cc_require(${pkgs})

if(required_FOUND)
  cc_executable(ptrace_demo
                SOURCES ptrace_demo.cc
                PKG_DEPENDS ${pkgs}
                TARGET_DEPENDS format)
else()
  mark_excluded("throttle: missing ${required_MISSING}")
endif()
