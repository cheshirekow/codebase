find_package( Eigen3 )
find_package( GTKmm )
find_package( YamlCpp )
find_package( TinyXML2 )
find_package( MPBlocks COMPONENTS gtk )

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
         
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x" )
    
    configure_file( config.cpp ${CMAKE_CURRENT_BINARY_DIR}/config.cpp )
        
    add_executable( mpblocks_voronoi_diagram_demo
                    main.cpp 
                    ${CMAKE_CURRENT_BINARY_DIR}/config.cpp
                    )
                    
    target_link_libraries( mpblocks_voronoi_diagram_demo ${LIBS} )
                    
else()
    set( MISSING "" )
    
    if( NOT (Eigen3_FOUND ) )
        set( MISSING ${MISSING} "Eigen3" )
    endif()
    
    if( NOT (GTKmm_FOUND ) )
        set( MISSING ${MISSING} "GTKmm" )
    endif()
    
    if( NOT (MPBlocks_FOUND ) )
        set( MISSING ${MISSING} "MPBlocks" )
    endif()

    message( WARNING "The following packages were not found, disabling "
                     "build of demo: " ${MISSING} )

    
endif()


