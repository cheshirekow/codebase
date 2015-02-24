configure_file(xephyr_test.cc ${CMAKE_CURRENT_BINARY_DIR}/xephyr_test.cc @ONLY)
add_executable(xephyr_test ${CMAKE_CURRENT_BINARY_DIR}/xephyr_test.cc)

include_directories(${Boost_INCLUDE_DIRS}
                    ${cairomm_INCLUDE_DIRS}
                    ${eigen3_INCLUDE_DIRS}
                    ${glog_INCLUDE_DIRS}
                    ${gflags_INCLUDE_DIRS}
                    ${pangomm_INCLUDE_DIRS}
                    ${x11_INCLUDE_DIRS}
                    ${xcomposite_INCLUDE_DIRS}
                    ${xdamage_INCLUDE_DIRS}
                    ${xext_INCLUDE_DIRS}
                    ${xfixes_INCLUDE_DIRS}
                    ${xrender_INDLUDE_DIRS})

set(LIBS
    ${cairomm_LDFLAGS}
    ${glog_LDFLAGS}
    ${gflags_LDFLAGS}
    ${pangomm_LDFLAGS}
    ${x11_LDFLAGS}
    ${xcomposite_LDFLAGS}
    ${xdamage_LDFLAGS}
    ${xext_LDFLAGS}
    ${xfixes_LDFLAGS}
    ${xrender_LDFLAGS})

cc_test(tree_test "tree_test.cc")
target_link_libraries(tree_test ${LIBS})