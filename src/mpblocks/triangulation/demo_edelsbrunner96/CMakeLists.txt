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
         ${sigc++_LDFLAGS}
         ${GTKmm_LDFLAGS}
         ${yamlcpp_LDFLAGS}
         ${TinyXML2_LDFLAGS}
         mpblocks_gtk )
   
    add_executable( mpblocks_demo_edelsbrunner96 EXCLUDE_FROM_ALL
                    main.cpp 
                    ${CMAKE_CURRENT_BINARY_DIR}/config.cpp
                    )
                    
    target_link_libraries( mpblocks_demo_edelsbrunner96 ${LIBS} )
                    
    configure_file( config.cpp ${CMAKE_CURRENT_BINARY_DIR}/config.cpp )
 
else()
    message( WARNING "Skipping target demo_edelsbrunner96" )
endif()


