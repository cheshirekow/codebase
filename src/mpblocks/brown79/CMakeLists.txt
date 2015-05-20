if(eigen3_FOUND)
  cc_include(eigen3)
  cc_test(brown79_test SOURCES brown79_test.cc)
else()
  message(WARNING "wont build brown79_test")
endif()

