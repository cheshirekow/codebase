# TODO(bialkowski): Change fiber to use static const instead of enum so we can
# allow this warning
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-enum-compare")

add_executable(fiber_demo demo.cc)