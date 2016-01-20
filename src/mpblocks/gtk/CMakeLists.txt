cc_require(eigen3 GTKmm TinyXML2 yamlcpp)
if(required_FOUND)
  include_directories(${eigen3_INCLUDE_DIRS}
                      ${GTKmm_INCLUDE_DIRS}
                      ${TinyXML2_INCLUDE_DIRS}
                      ${yamlcpp_INCLUDE_DIRS})
  set(LIBS
      ${GTKmm_LDFLAGS}
      ${TinyXML2_LDFLAGS}
      ${yamlcpp_LDFLAGS})

  add_library(mpblocks_gtk SHARED
              LayoutMap.cpp
              EigenCairo.cpp
              SimpleView.cpp)
  target_link_libraries(mpblocks_gtk ${LIBS})

  add_library(mpblocks_gtk_static STATIC
              LayoutMap.cpp
              EigenCairo.cpp
              SimpleView.cpp)
  target_link_libraries(mpblocks_gtk_static ${LIBS})


  install(TARGETS
          mpblocks_gtk 
          mpblocks_gtk_static
          RUNTIME DESTINATION bin
          LIBRARY DESTINATION lib
          ARCHIVE DESTINATION lib)
else()
  mark_excluded("mpblocks gtk: missing ${required_MISSING}")
endif()
