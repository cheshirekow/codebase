if( (Eigen3_FOUND) 
        AND (GTKmm_FOUND)
        AND (Boost_FOUND)
        AND (MPBlocks_FOUND)
        AND (Loki_FOUND) ) 

    include_directories(
        ${Eigen3_INCLUDE_DIR}
        ${GTKmm_INCLUDE_DIRS}
        ${MPBlocks_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${Boost_INCLUDE_DIRS}
        )
        
    set( LIBS ${LIBS} 
         ${GTKmm_LIBRARIES}
         ${Boost_LIBRARIES}
         ${MPBlocks_LIBRARIES} )
    
    add_executable( mpblocks_2d_obstacle_set_creator
                    main.cpp 
                    ViewArea.cpp
                    ${CMAKE_CURRENT_BINARY_DIR}/config.cpp
                    )
                    
    target_link_libraries( mpblocks_2d_obstacle_set_creator ${LIBS} )
                    
    configure_file( config.cpp ${CMAKE_CURRENT_BINARY_DIR}/config.cpp )
 
else()
    message( WARNING "Skipping target 2d_obstacle_set_creator" )
endif()


