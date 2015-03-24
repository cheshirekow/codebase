if((eigen3_FOUND)
   AND (GTKmm_FOUND)
   AND (TinyXML2_FOUND)
   AND (yamlcpp_FOUND))

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
  message( WARNING "disabling build of demo_uniform_delaunay")
endif()


