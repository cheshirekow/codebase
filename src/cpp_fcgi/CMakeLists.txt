if(FastCGI_FOUND)
  include_directories(${FastCGI_INCLUDE_DIRS})
  
  set(LIBRARY_SOURCES cpp_fcgi.cpp
                      Request.cpp
                      Socket.cpp)
  
  add_library(cpp_fcgi_shared SHARED ${LIBRARY_SOURCES})
  add_library(cpp_fcgi_static STATIC ${LIBRARY_SOURCES})
  
  set_target_properties(cpp_fcgi_shared
                        cpp_fcgi_static
                        PROPERTIES OUTPUT_NAME cpp_fcgi)
  
  install(FILES cpp_fcgi.h Request.h Socket.h
          DESTINATION include/cpp_fcgi)
                    
  install(TARGETS cpp_fcgi_shared cpp_fcgi_static
          RUNTIME DESTINATION bin
          LIBRARY DESTINATION lib
          ARCHIVE DESTINATION lib )

  cc_test(fcgi_test fcgi_test.cpp)  
  target_link_libraries(fcgi_test
                        cpp_fcgi_static
                        ${FastCGI_LIBRARIES} )
else()
  message(WARNING "Can't build cpp_fcgi")
endif()