set(LIBRARY_HEADERS eventlog.h
                    lcm.h
                    lcm-cpp.hpp
                    lcm-cpp-impl.hpp
                    lcmtypes/channel_to_port_t.h
                    lcmtypes/channel_port_map_udpate_t.h)

set(LIBRARY_SOURCES dbg.h
	                  lcm.c
	                  lcm.h
	                  lcm_udpm.c
	                  lcm_file.c
	                  lcm_memq.c
	                  lcm_mpudpm.c
	                  lcm_tcpq.c
	                  ringbuffer.c
	                  ringbuffer.h
	                  udpm_util.c
	                  udpm_util.h
	                  eventlog.c
	                  eventlog.h
	                  ioutils.h
	                  lcm_internal.h
	                  lcm-cpp.hpp
	                  lcm-cpp-impl.hpp
	                  lcmtypes/channel_to_port_t.c
	                  lcmtypes/channel_to_port_t.h
	                  lcmtypes/channel_port_map_update_t.c
	                  lcmtypes/channel_port_map_update_t.h)

add_library(lcm_static STATIC ${LIBRARY_SOURCES})
add_library(lcm_shared SHARED ${LIBRARY_SOURCES})

set_target_properties(
  lcm_static 
  lcm_shared
  PROPERTIES OUTPUT_NAME lcm
             COMPILE_FLAGS "-Wno-deprecated-declarations -std=gnu99")


install(FILES ${LIBRARY_HEADERS}
        DESTINATION include/lcm)
  
install(TARGETS lcm_static lcm_shared
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

