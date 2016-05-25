return()
cc_library(gltk
           HEADERS buffer.h
                   framebuffer.h
                   gluintref.h
                   gltk.h
                   program.h
                   refcounted.h
                   refptr.h
                   shader.h
                   texture.h
                   vertexarray.h
           SOURCES buffer.cc
                   framebuffer.cc
                   program.cc
                   refcounted.cc
                   shader.cc
                   texture.cc
                   vertexarray.cc)
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
      gltk_shared)

    add_subdirectory(test01)
else()
  message(WARNING "Wont build gltk test01")
endif()

if((OPENGL_FOUND) AND (x11_FOUND))
  add_subdirectory(xlib)
endif()
