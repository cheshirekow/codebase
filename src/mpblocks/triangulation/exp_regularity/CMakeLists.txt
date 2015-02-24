set( OUTPUT exp_regularity )

if( (Eigen3_FOUND)
    AND (GTKmm_FOUND)
    AND (YAMLCPP_FOUND)
    AND (TinyXML2_FOUND)
    AND (Boost_FOUND)
    AND (MPBlocks_FOUND) )
    
    include_directories(
        ${Eigen3_INCLUDE_DIR}
        ${GTKmm_INCLUDE_DIRS}
        ${MPBlocks_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
        ${CPPThreads_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
        )
        
    set( LIBS ${LIBS} 
         ${GTKmm_LIBRARIES}
         ${MPBlocks_LIBRARIES}
         ${MPBlocks_gtk_LIBRARY}
         ${YAMLCPP_LIBRARY}
         ${Boost_LIBRARIES}
         ${CPPThreads_LIBRARIES}
         ${TinyXML2_LIBRARY} )
         
    configure_file( config.cpp ${CMAKE_CURRENT_BINARY_DIR}/config.cpp )
        
    add_executable( ${OUTPUT} #EXCLUDE_FROM_ALL
                    main.cpp 
                    ${CMAKE_CURRENT_BINARY_DIR}/config.cpp
                    )
                    
    target_link_libraries( ${OUTPUT} ${LIBS} )
                    
else()
    message( WARNING
        "disabling build of ${OUTPUT}, missing: ${FINDALL_${OUTPUT}_MISSING}")
endif()


