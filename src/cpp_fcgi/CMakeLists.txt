if(FastCGI_FOUND)
  cc_library(cpp_fcgi
             HEADERS cpp_fcgi.h
                     Request.h
                     Socket.h
             SOURCES cpp_fcgi.cpp
                     Request.cpp
                     Socket.cpp
             CMAKE_DEPENDS FastCGI)

  cc_test(fcgi_test SOURCES fcgi_test.cpp TARGET_DEPENDS cpp_fcgi_static)
else()
  message(WARNING "Can't build cpp_fcgi")
endif()
