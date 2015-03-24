set(LIBRARY_HEADERS
    attr.h
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
    thread.h)

set(LIBRARY_SOURCES
    enums.cc
    barrier.cc
    barrier_attr.cc
    condition.cc
    condition_attr.cc
    key.cc
    mutex.cc
    mutex_attr.cc
    thread.cc
    thread_attr.cc)

add_library(cpp_pthreads_static STATIC ${LIBRARY_SOURCES})
add_library(cpp_pthreads_shared SHARED ${LIBRARY_SOURCES})

set_target_properties(cpp_pthreads_static cpp_pthreads_shared
                      PROPERTIES OUTPUT_NAME cpp_pthreads)

install(FILES ${LIBRARY_HEADERS}
        DESTINATION include/cpp_pthreads)

install(TARGETS cpp_pthreads_static
                cpp_pthreads_shared
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib )

macro(pthreads_test TEST_NAME)
  cc_test(${TEST_NAME} ${TEST_NAME}.cc ${ARGN})
  target_link_libraries(${TEST_NAME} cpp_pthreads_static)
endmacro()

pthreads_test(launch_join_test) 
pthreads_test(attr_api_test)
