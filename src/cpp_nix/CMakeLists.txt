cc_library(cpp_nix
           HEADERS clock.h
                   epoll.h
                   fd_set.h
                   notify_pipe.h
                   select_set.h
                   timespec.h
                   timeval.h
           SOURCES clock.cc
                    epoll.cc
                    fd_set.cc
                    notify_pipe.cc
                    select_set.cc
                    timespec.cc
                    timeval.cc
           PKG_DEPENDS glog)

# All thests for this library are a single source file and link the library
# itself
macro(nix_test test_name)
  cc_test(nix_${test_name} SOURCES ${test_name}.cc)
  target_link_libraries(nix_${test_name}_exe cpp_nix_shared)
endmacro()

nix_test(clock_test)
nix_test(epoll_test)
nix_test(notify_pipe_test)
nix_test(timespec_test)
nix_test(timeval_test)
