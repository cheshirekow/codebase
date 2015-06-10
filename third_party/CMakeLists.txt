add_subdirectory(cppformat)
add_subdirectory(gtest)
add_subdirectory(kdtree)
add_subdirectory(lcm)

if(NOT (re2_FOUND))
  add_subdirectory(re2)  
  set(re2_FOUND TRUE PARENT_SCOPE)
  set(re2_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/re2 PARENT_SCOPE)
  set(re2_LDFLAGS re2_static PARENT_SCOPE)
endif()

