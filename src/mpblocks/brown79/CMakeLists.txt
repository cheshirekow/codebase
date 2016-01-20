if(eigen3_FOUND)
  cc_include(eigen3)
  cc_test(brown79_test SOURCES brown79_test.cc)
else()
  mark_excluded("brown79 test: missing eigen3")
endif()

