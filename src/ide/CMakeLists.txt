include_directories(${glog_INCLUDE_DIRS}
                    ${tclap_INCLUDE_DIRS})

add_executable(editor_poc editor_poc.cc)
target_link_libraries(editor_poc meerkat_static
                                 format_static
                                 ${glog_LDFLAGS}
                                 ${tclap_LDFLAGS})
