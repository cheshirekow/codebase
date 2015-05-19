add_executable(lcm-logger lcm_logger.c
                          glib_util.c
                          glib_util.h)
target_link_libraries(lcm-logger lcm_static ${glib_LDFLAGS} ${gthread_LDFLAGS}
                      ${CMAKE_THREAD_LIBS_INIT})

add_executable(lcm-logplayer lcm_logplayer.c)
target_link_libraries(lcm-logplayer lcm_static ${glib_LDFLAGS}
                      ${CMAKE_THREAD_LIBS_INIT})

set_target_properties(
  lcm-logger
  lcm-logplayer
  PROPERTIES COMPILE_FLAGS "-Wno-deprecated-declarations -std=gnu99")
