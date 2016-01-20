add_subdirectory(impl_tests)

if(eigen3_FOUND)
  include_directories(${eigen3_INCLUDE_DIRS})
  cc_test(dubins_funcs_test SOURCES funcs_test.cc)
else()
  mark_excluded("dubins funcs test: missing eigen3")
endif()
