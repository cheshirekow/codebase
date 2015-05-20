configure_file(${CMAKE_CURRENT_SOURCE_DIR}/main.cc
               ${CMAKE_CURRENT_BINARY_DIR}/main.cc )

cc_executable(gltk_test_01
              SOURCES ${CMAKE_CURRENT_BINARY_DIR}/main.cc
              PKG_DEPENDS glew glfw
              CMAKE_DEPENDS OPENGL
              TARGET_DEPENDS gltk_static)
