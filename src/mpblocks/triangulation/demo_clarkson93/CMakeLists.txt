if( (eigen3_FOUND) 
    AND (GTKmm_FOUND) 
    AND (sigc++_FOUND)
    AND (yamlcpp_FOUND)
    AND (TinyXML2_FOUND) ) 

    include_directories(
        ${eigen3_INCLUDE_DIR}
        ${sigc++_INCLUDE_DIRS}
        ${GTKmm_INCLUDE_DIRS}
        ${yamlcpp_INCLUDE_DIR}
        ${TinyXML2_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}
        )
        
    set( LIBS ${LIBS} 
         ${sigc++_LIBRARIES}
         ${GTKmm_LIBRARIES}
         ${yamlcpp_LIBRARY}
         ${TinyXML2_LIBRARIES}
         mpblocks_gtk )
   
    add_executable( mpblocks_demo_clarkson93 EXCLUDE_FROM_ALL
                    main.cpp 
                    ${CMAKE_CURRENT_BINARY_DIR}/config.cpp
                    )
                    
    target_link_libraries( mpblocks_demo_clarkson93 ${LIBS} )
                    
    configure_file( config.cpp ${CMAKE_CURRENT_BINARY_DIR}/config.cpp )
 
else()
    set( MISSING "" )

    if( NOT (Eigen3_FOUND ) )
        set( MISSING ${MISSING} "Eigen3, ")
    endif()
    
    if( NOT (GTKmm_FOUND ) )
        set( MISSING ${MISSING} "GTKmm, ")
    endif()
    
    message( WARNING "Skipping target demo_clarkson93" )
endif()


