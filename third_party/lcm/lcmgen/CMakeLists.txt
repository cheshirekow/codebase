add_executable(lcmgen main.c
				              lcmgen.c
				              lcmgen.h
				              getopt.c
				              getopt.h
				              tokenize.c
				              tokenize.h
				              emit_c.c
				              emit_cpp.c
				              emit_java.c
				              emit_csharp.c
				              emit_python.c
				              emit_lua.c)

set(cflags_list -Wno-deprecated-declarations
                -Wno-format-zero-length
                -std=gnu99)
string (REGEX REPLACE ";" " " cflags_str "${cflags_list}")
set_target_properties(lcmgen PROPERTIES COMPILE_FLAGS "${cflags_str}")

target_link_libraries(lcmgen lcm_static ${glib_LDFLAGS})

install(TARGETS lcmgen
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
        

