cc_require(eigen3 sigc++ GTKmm cairomm TinyXML2 yamlcpp)
if(required_FOUND)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}
        ${eigen3_INCLUDE_DIRS}
        ${sigc++_INCLUDE_DIRS}
        ${GTKmm_INCLUDE_DIRS}
        ${cairomm_INCLUDE_DIRS}
        ${yamlcpp_INCLUDE_DIRS}
        ${TinyXML2_INCLUDE_DIRS}) 
        
    set(LIBS ${LIBS}
        ${GTKmm_LDFLAGS}
        ${SigC++_LDFLAGS}
        ${YAMLCPP_LDFLAGS}
        ${TinyXML2_LDFLAGS}
        ${MPBlocks_gtk_LDFLAGS})
         
    configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/config.cpp
                    ${CMAKE_CURRENT_BINARY_DIR}/config.cpp )
    
    add_executable( mpblocks_demo_uniform_simplex EXCLUDE_FROM_ALL
                        ${CMAKE_CURRENT_BINARY_DIR}/config.cpp
                        main.cpp 
                        )
    target_link_libraries(mpblocks_demo_uniform_simplex ${LIBS} )
else()
    mark_excluded("uniform simplex triangulation demo: missing" 
                  " ${required_MISSING}")
endif()

