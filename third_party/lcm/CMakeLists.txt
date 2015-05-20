# This listfiles was not provided as part of the upstream distribution
# Edits to upstream:
#  - add #include <unistd.h> to lcm-logger

# This directory needs to be on the include path so that targets can find
# lcm/lcm.h
include_directories(${CMAKE_CURRENT_SOURCE_DIR}
                    ${glib_INCLUDE_DIRS})

if(glib_FOUND)
  add_subdirectory(lcm)
  add_subdirectory(lcmgen)
  add_subdirectory(lcm-java)
  add_subdirectory(lcm-logger)
endif()
