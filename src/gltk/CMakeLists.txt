set(LIBRARY_HEADERS buffer.h
                    eventloop.h
                    events.h
                    framebuffer.h
                    gluintref.h
                    gltk.h
                    pipeline.h
                    program.h
                    refcounted.h
                    refptr.h
                    shader.h
                    texture.h
                    vertexarray.h
                    widget.h)

set(LIBRARY_SOURCES buffer.cc
                    eventloop.cc
                    events.cc
                    framebuffer.cc
                    pipeline.cc
                    program.cc
                    refcounted.cc
                    shader.cc
                    texture.cc
                    vertexarray.cc
                    widget.cc)

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

macro(gltk_test test_name)
  cc_test(${test_name} ${test_name}.cc)
  target_link_libraries(${test_name} gltk_static
                                     cpp_nix_static
                                     ${glog_LDFLAGS})
endmacro()

gltk_test(eventloop_test)

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
      gltk_shared
      cpp_nix_shared
      ${glog_LDFLAGS} )

    add_subdirectory(test01)
else()
  message(WARNING "Wont build gltk test01")
endif()

if((OPENGL_FOUND) AND (x11_FOUND))
  add_subdirectory(xlib)
endif()

