cc_executable(lcmgen
              SOURCES main.c
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
                      emit_lua.c
              TARGET_DEPENDS lcm_static)

set(cflags_list -Wno-deprecated-declarations
                -Wno-format-zero-length
                -std=gnu99)
string (REGEX REPLACE ";" " " cflags_str "${cflags_list}")
set_target_properties(lcmgen_exe PROPERTIES COMPILE_FLAGS "${cflags_str}")
