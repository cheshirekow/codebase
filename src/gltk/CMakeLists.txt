set(LIBRARY_HEADERS buffer.h
                    framebuffer.h
                    gluintref.h
                    gltk.h
                    program.h
                    refcounted.h
                    refptr.h
                    shader.h
                    texture.h
                    vertexarray.h)

set(LIBRARY_SOURCES buffer.cc
                    framebuffer.cc
                    program.cc
                    refcounted.cc
                    shader.cc
                    texture.cc
                    vertexarray.cc)

add_library(gltk_static STATIC ${LIBRARY_SOURCES})
add_library(gltk_shared SHARED ${LIBRARY_SOURCES})

set_target_properties(gltk_static gltk_shared
                      PROPERTIES OUTPUT_NAME gltk)

install(FILES ${LIBRARY_HEADERS}
        DESTINATION include/gltk)

install(TARGETS gltk_static
                gltk_shared
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib )

if((OPENGL_FOUND)
    AND (glew_FOUND)
    AND (glfw_FOUND))

  include_directories(${OPENGL_INCLUDE_DIR} 
                     ${glew_INCLUDE_DIR}
                     ${glfw_INCLUDE_DIR} )

  set(LIBS ${LIBS}
      ${OPENGL_LIBRARY}
      ${glew_LDFLAGS}
      ${glfw_LDFLAGS}
      gltk_shared )

    add_subdirectory(test01)
else()
  message(WARNING "Wont build gltk test01")
endif()
