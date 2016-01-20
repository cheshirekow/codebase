cc_require(eigen3 GTKmm TinyXML2 yamlcpp)
if(required_FOUND)
  include_directories(${eigen3_INCLUDE_DIRS}
                      ${GTKmm_INCLUDE_DIRS}
                      ${yamlcpp_INCLUDE_DIRS}
                      ${TinyXML2_INCLUDE_DIRS}
                      ${CMAKE_CURRENT_SOURCE_DIR})

  configure_file( config.cpp ${CMAKE_CURRENT_BINARY_DIR}/config.cpp )
  add_executable(mpblocks_demo_uniform_delaunay
                 main.cpp
                 ${CMAKE_CURRENT_BINARY_DIR}/config.cpp)
  target_link_libraries(mpblocks_demo_uniform_delaunay
      ${GTKmm_LDFLAGS}
      ${yamlcpp_LDFLAGS}
      ${TinyXML2_LDFLAGS}
      mpblocks_gtk)
else()
  mark_excluded("uniform delaunay triangulation demo: missing"
                " ${required_MISSING}")
endif()


