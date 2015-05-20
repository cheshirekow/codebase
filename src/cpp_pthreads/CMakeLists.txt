cc_library(cpp_pthreads
           HEADERS attr.h
                   barrier_attr.h
                   barrier.h
                   cpp_pthreads.h
                   condition_attr.h
                   condition.h
                   enums.h
                   key.h
                   mutex_attr.h
                   mutex.h
                   thread_attr.h
                   thread.h
           SOURCES enums.cc
                   barrier.cc
                   barrier_attr.cc
                   condition.cc
                   condition_attr.cc
                   key.cc
                   mutex.cc
                   mutex_attr.cc
                   thread.cc
                   thread_attr.cc)

# All thests for this library are a single source file and link the library
# itself
macro(pthreads_test test_name)
  cc_test(pthreads_${test_name} SOURCES ${test_name}.cc)
  target_link_libraries(pthreads_${test_name}_exe cpp_pthreads_static)
endmacro()

pthreads_test(launch_join_test) 
pthreads_test(attr_api_test)
