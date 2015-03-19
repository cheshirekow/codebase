if(FastCGI_FOUND)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}
                      ${FastCGI_INCLUDE_DIRS})
        
   set(LIBS ${FastCGI_LIBRARIES})

   cc_test(cpp_fcgi_test main.cpp)  
   target_link_libraries(cpp_fcgi_test
                         cpp_fcgi_shared
                         ${FastCGI_LIBRARIES} )
else()
  message(WARNING "Can't build cpp_fcgi")
endif()