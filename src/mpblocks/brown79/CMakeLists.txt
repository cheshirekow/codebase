if(eigen3_FOUND)
  include_directories(${eigen3_INCLUDE_DIRS})
  cc_test(brown79_test brown79_test.cc)
else()
  message(WARNING "wont build brown79_test")
endif()

