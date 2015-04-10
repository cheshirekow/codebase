set(LIBRARY_SOURCES is_call_possible.h
                    v1.h)

install(FILES ${LIBRARY_SOURCES} DESTINATION include/kwargs)

add_executable(kwargs_example_1 example_1.cc)
add_executable(kwargs_example_2 example_2.cc)
add_executable(kwargs_example_3 example_3.cc)
add_executable(kwargs_example_4 example_4.cc)
add_executable(kwargs_example_5 example_5.cc)
add_executable(kwargs_example_6 example_6.cc)
