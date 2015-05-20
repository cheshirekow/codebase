include_directories(${OPENGL_INCLUDE_DIR}
                    ${x11_INCLUDE_DIRS})

cc_executable(gltk_xlib_window_demo
              SOURCES xlib_window.cc
              PKG_DEPENDS x11
              CMAKE_DEPENDS OPENGL
              TARGET_DEPENDS format_static)
