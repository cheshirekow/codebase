cc_executable(lcm-logger
              SOURCES lcm_logger.c
                      glib_util.c
                      glib_util.h
              TARGET_DEPENDS lcm_static)
target_link_libraries(lcm-logger_exe ${CMAKE_THREAD_LIBS_INIT})

cc_executable(lcm-logplayer
              SOURCES lcm_logplayer.c
              TARGET_DEPENDS lcm_static)
target_link_libraries(lcm-logplayer_exe ${CMAKE_THREAD_LIBS_INIT})

set_target_properties(
  lcm-logger_exe
  lcm-logplayer_exe
  PROPERTIES COMPILE_FLAGS "-Wno-deprecated-declarations -std=gnu99")
