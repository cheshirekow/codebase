configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
                ${CMAKE_CURRENT_BINARY_DIR}/main.cpp )
add_executable(test01 ${CMAKE_CURRENT_BINARY_DIR}/main.cpp )
target_link_libraries( test01 ${LIBS} )
