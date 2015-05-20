if(Boost_FOUND)
  cc_include(Boost)
  cc_test(redblack_test
           SOURCES redblack_test.cc
           TARGET_DEPENDS mpblocks_util_static)
else()
  message(WARNING "wont build redblack_test" )
endif()
