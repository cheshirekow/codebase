find_package( Eigen3 )
find_package( GTKmm )
find_package( MPBlocks )
find_package( Boost COMPONENTS random )
find_package( Loki )

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
    set( MISSING "" )
    
    if( NOT (Eigen3_FOUND ) )
        set( MISSING "${MISSING}, Eigen3" )
    endif()
    
    if( NOT (Boost_FOUND ) )
        set( MISSING "${MISSING}, boost->random" )
    endif()
    
    if( NOT (MPBlocks_FOUND ) )
        set( MISSING "${MISSING}, mpblocks" )
    endif()
    
    if( NOT (GTKmm_FOUND ) )
        set( MISSING "${MISSING}, GTKmm" )
    endif()
    
    if( NOT (Loki_FOUND ) )
        set( MISSING "${MISSING}, loki" )
    endif()

    message( WARNING "The following packages were not found, disabling "
                     "build of 2d obstacle sets: " ${MISSING} )

    
endif()


