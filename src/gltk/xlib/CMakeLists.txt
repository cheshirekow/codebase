include_directories(${OPENGL_INCLUDE_DIR}
                    ${x11_INCLUDE_DIRS})

add_executable(gltk_xlib_window_demo xlib_window.cc)

target_link_libraries(gltk_xlib_window_demo 
                      format_static
                      ${OPENGL_LIBRARIES}
                      ${x11_LDFLAGS})