if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
  cc_test(redblack_test redblack_test.cc)
  target_link_libraries(redblack_test
                        mpblocks_util)
else()
  message(WARNING "wont build redblack_test" )
endif()