cc_library(lcm
           HEADERS eventlog.h
                   lcm.h
                   lcm-cpp.hpp
                   lcm-cpp-impl.hpp
                   lcmtypes/channel_to_port_t.h
                   lcmtypes/channel_port_map_update_t.h
           SOURCES dbg.h
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
                   lcmtypes/channel_port_map_update_t.c
            PKG_DEPENDS glib gthread)

set_target_properties(
  lcm_static 
  lcm_shared
  PROPERTIES OUTPUT_NAME lcm
             COMPILE_FLAGS "-Wno-deprecated-declarations -std=gnu99")
