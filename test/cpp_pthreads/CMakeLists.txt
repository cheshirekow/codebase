macro(pthreads_test TEST_NAME)
  cc_test(${TEST_NAME} ${TEST_NAME}.cc ${ARGN})
  target_link_libraries(${TEST_NAME} cpp_pthreads_static)
endmacro()

pthreads_test(launch_join_test) 
pthreads_test(attr_api_test)
