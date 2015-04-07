set(LIBRARY_HEADERS time.h
                    eventloop.h)

set(LIBRARY_SOURCES clock.cc
                    eventloop.cc)

add_library(kevent_static STATIC ${LIBRARY_SOURCES})
add_library(kevent_shared SHARED ${LIBRARY_SOURCES})

set_target_properties(kevent_static kevent_shared
                      PROPERTIES OUTPUT_NAME kevent)

install(FILES ${LIBRARY_HEADERS}
        DESTINATION include/kevent)

install(TARGETS kevent_static
                kevent_shared
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib )

macro(kevent_test TEST_NAME)
  cc_test(${TEST_NAME} ${TEST_NAME}.cc ${ARGN})
  target_link_libraries(${TEST_NAME} format_static kevent_static cpp_nix_static)
endmacro()

kevent_test(eventloop_test)