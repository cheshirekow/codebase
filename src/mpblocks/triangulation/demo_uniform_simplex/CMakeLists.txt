if( (Eigen3_FOUND)
    AND (SigC++_FOUND)
    AND (MPBlocks_FOUND )
    AND (MPBlocks_gtk_FOUND )
    AND (GTKmm_FOUND)
    AND (Cairomm_FOUND)
    AND (YAMLCPP_FOUND)
    AND (TinyXML2_FOUND)
    )
    
    include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${Eigen3_INCLUDE_DIR}
        ${SigC++_INCLUDE_DIRS}
        ${MPBlocks_INCLUDE_DIRS}
        ${GTKmm_INCLUDE_DIRS}
        ${Cairomm_INCLUDE_DIRS}
        ${YAMLCPP_INCLUDE_DIR}
        ${TinyXML2_INCLUDE_DIRS}
        ${MPBlocks_INCLUDE_DIR}
        ) 
        
    set( LIBS ${LIBS}
        ${GTKmm_LIBRARIES}
        ${SigC++_LIBRARIES}
        ${YAMLCPP_LIBRARY}
        ${TinyXML2_LIBRARIES}
        ${MPBlocks_gtk_LIBRARY}
         )
         
    configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/config.cpp
                    ${CMAKE_CURRENT_BINARY_DIR}/config.cpp )
    
    add_executable( mpblocks_demo_uniform_simplex EXCLUDE_FROM_ALL
                        ${CMAKE_CURRENT_BINARY_DIR}/config.cpp
                        main.cpp 
                        )
    target_link_libraries(mpblocks_demo_uniform_simplex ${LIBS} )
else()
    message( WARNING "Skipping target demo_uniform_simplex" )
endif()

