cc_library(gltk
           HEADERS buffer.h
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
                   widget.h
           SOURCES buffer.cc
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

macro(gltk_test test_name)
  cc_test(gltk_${test_name} SOURCES ${test_name}
          PKG_DEPENDS glog
          TARGET_DEPENDS gltk_static cpp_nix_static)
endmacro()

gltk_test(eventloop_test)

add_subdirectory(test01)

if((OPENGL_FOUND) AND (x11_FOUND))
  add_subdirectory(xlib)
endif()
