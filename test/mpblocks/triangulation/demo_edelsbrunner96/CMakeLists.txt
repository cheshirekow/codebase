if( (Eigen3_FOUND) 
    AND (GTKmm_FOUND) 
    AND (SigC++_FOUND)
    AND (YAMLCPP_FOUND)
    AND (TinyXML2_FOUND) ) 

    include_directories(
        ${Eigen3_INCLUDE_DIR}
        ${SigC++_INCLUDE_DIRS}
        ${GTKmm_INCLUDE_DIRS}
        ${YAMLCPP_INCLUDE_DIR}
        ${TinyXML2_INCLUDE_DIRS}
        ${MPBlocks_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
        )
        
    set( LIBS ${LIBS} 
         ${SigC++_LIBRARIES}
         ${GTKmm_LIBRARIES}
         ${MPBlocks_gtk_LIBRARY}
         ${YAMLCPP_LIBRARY}
         ${TinyXML2_LIBRARIES} )
   
    add_executable( mpblocks_demo_edelsbrunner96 EXCLUDE_FROM_ALL
                    main.cpp 
                    ${CMAKE_CURRENT_BINARY_DIR}/config.cpp
                    )
                    
    target_link_libraries( mpblocks_demo_edelsbrunner96 ${LIBS} )
                    
    configure_file( config.cpp ${CMAKE_CURRENT_BINARY_DIR}/config.cpp )
 
else()
    message( WARNING "Skipping target demo_edelsbrunner96" )
endif()


