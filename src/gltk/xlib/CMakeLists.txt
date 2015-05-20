cc_include(glog OPENGL x11)

cc_library(gltk_xlib
           HEADERS translate_event.h
                   window.h
           SOURCES translate_event.cc
                   window.cc
           PKG_DEPENDS glog x11
           TARGET_DEPENDS gltk cpp_nix format)

cc_executable(gltk_xlib_window_demo
              SOURCES window_demo.cc
              PKG_DEPENDS x11
              CMAKE_DEPENDS OPENGL
              TARGET_DEPENDS gltk_xlib_static)
