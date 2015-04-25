include_directories(${glog_INCLUDE_DIRS}
                    ${tclap_INCLUDE_DIRS}
                    /usr/lib/llvm-3.5/include) #TODO(josh): make findscript

add_executable(editor_poc editor_poc.cc)
add_executable(clang_render clang_render.cc
                            string_to_argv.cc)

target_link_libraries(editor_poc meerkat_static
                                 format_static
                                 json_spirit_static
                                 ${glog_LDFLAGS}
                                 ${tclap_LDFLAGS})

target_link_libraries(clang_render format_static
                                 json_spirit_static
                                 ${glog_LDFLAGS}
                                 ${tclap_LDFLAGS}
                                 -lclang-3.5)
