include_directories(${glog_INCLUDE_DIRS})

set(LIBRARY_HEADERS clock.h
                    epoll.h
                    fd_set.h
                    notify_pipe.h
                    select_set.h
                    timespec.h
                    timeval.h)

set(LIBRARY_SOURCES clock.cc
                    epoll.cc
                    fd_set.cc
                    notify_pipe.cc
                    select_set.cc
                    timespec.cc
                    timeval.cc)

add_library(cpp_nix_static STATIC ${LIBRARY_SOURCES})
add_library(cpp_nix_shared SHARED ${LIBRARY_SOURCES})

set_target_properties(
  cpp_nix_static 
  cpp_nix_shared
  PROPERTIES OUTPUT_NAME cpp_nix)

install(FILES ${LIBRARY_HEADERS}
        DESTINATION include/cpp_nix)
  
install(TARGETS cpp_nix_static cpp_nix_shared
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
        
macro(nix_test TEST_NAME)
  cc_test(${TEST_NAME} ${TEST_NAME}.cc ${ARGN})
  target_link_libraries(${TEST_NAME} cpp_nix_static ${glog_LDFLAGS})
endmacro()

nix_test(clock_test clock_test.cc)
nix_test(epoll_test epoll_test.cc)
nix_test(notify_pipe_test notify_pipe_test.cc)
nix_test(timespec_test timespec_test.cc)
nix_test(timeval_test timeval_test.cc)
        
