if( (Eigen3_FOUND) 
    AND (GTKmm_FOUND)
    AND (YAMLCPP_FOUND)
    AND (TinyXML2_FOUND)
    AND (MPBlocks_FOUND) ) 

    include_directories(
        ${Eigen3_INCLUDE_DIR}
        ${GTKmm_INCLUDE_DIRS}
        ${MPBlocks_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}
        )
        
    set( LIBS ${LIBS} 
         ${GTKmm_LIBRARIES}
         ${MPBlocks_LIBRARIES}
         ${MPBlocks_gtk_LIBRARY}
         ${YAMLCPP_LIBRARY}
         ${TinyXML2_LIBRARY} )
         
    configure_file( config.cpp ${CMAKE_CURRENT_BINARY_DIR}/config.cpp )
        
    add_executable( mpblocks_demo_voronoi2d EXCLUDE_FROM_ALL
                    main.cpp 
                    ${CMAKE_CURRENT_BINARY_DIR}/config.cpp
                    )
                    
    target_link_libraries( mpblocks_demo_voronoi2d ${LIBS} )
                    
else()
    message( WARNING "Skipping target demo-voronoi2d" )

    
endif()


