cc_require(eigen3 GTKmm TinyXML2 yamlcpp)
if(required_FOUND)
    include_directories(
        ${eigen3_INCLUDE_DIRS}
        ${GTKmm_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}
        )
        
    set( LIBS ${LIBS} 
         ${GTKmm_LIBRARIES}
         ${yamlcpp_LIBRARY}
         ${TinyXML2_LIBRARY}
         mpblocks_gtk )
         
    configure_file( config.cpp ${CMAKE_CURRENT_BINARY_DIR}/config.cpp )
        
    add_executable( mpblocks_demo_voronoi2d EXCLUDE_FROM_ALL
                    main.cpp 
                    ${CMAKE_CURRENT_BINARY_DIR}/config.cpp
                    )
                    
    target_link_libraries( mpblocks_demo_voronoi2d ${LIBS} )
                    
else()
    mark_excluded("triangulation demo voronoi2d: missing ${required_MISSING}")
endif()


