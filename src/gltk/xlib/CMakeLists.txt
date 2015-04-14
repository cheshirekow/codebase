include_directories(${glog_INCLUDE_DIRS}
                    ${OPENGL_INCLUDE_DIR}
                    ${x11_INCLUDE_DIRS})

set(LIBRARY_HEADERS xlib_window.h)
set(LIBRARY_SOURCES xlib_window.cc)


add_library(gltk_xlib_static STATIC ${LIBRARY_SOURCES})
add_library(gltk_xlib_shared SHARED ${LIBRARY_SOURCES})

set_target_properties(gltk_xlib_static gltk_xlib_shared
                      PROPERTIES OUTPUT_NAME gltk_xlib)

install(FILES ${LIBRARY_HEADERS}
        DESTINATION include/gltk/xlib)

install(TARGETS gltk_xlib_static
                gltk_xlib_shared
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib )

add_executable(gltk_xlib_window_demo xlib_window_demo.cc)

target_link_libraries(gltk_xlib_window_demo 
                      gltk_xlib_static
                      gltk_static
                      format_static
                      ${glog_LDFLAGS}
                      ${OPENGL_LIBRARIES}
                      ${x11_LDFLAGS})
