if(libcrypto++_FOUND)
  include_directories(${libcrypto++_INCLUDE_DIRS})
    
  add_executable(diffie_hellman_test
                 diffie_hellman_test.cpp)
  target_link_libraries(diffie_hellman_test ${libcrypto++_LDFLAGS})
else() 
  message(WARNING "Can't build diffie_hellman_test")
endif()

