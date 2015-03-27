configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/main.cc
                ${CMAKE_CURRENT_BINARY_DIR}/main.cc )
add_executable(gltk_test_01 ${CMAKE_CURRENT_BINARY_DIR}/main.cc )
target_link_libraries(gltk_test_01 ${LIBS} )
