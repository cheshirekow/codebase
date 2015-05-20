if((OPENGL_FOUND)
    AND (glew_FOUND)
    AND (glfw_FOUND))

  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/main.cc
                 ${CMAKE_CURRENT_BINARY_DIR}/main.cc )

  cc_executable(gltk_test_01
                SOURCES ${CMAKE_CURRENT_BINARY_DIR}/main.cc
                PKG_DEPENDS glew glfw
                CMAKE_DEPENDS OPENGL
                TARGET_DEPENDS gltk_static)
else()
  message(WARNING "Wont build gltk test01")
endif()
