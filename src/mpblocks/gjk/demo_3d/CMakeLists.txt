


if( (Eigen3_FOUND)
    AND (MPBlocks_FOUND)
    AND (TinyXML2_FOUND)
    AND (YAMLCPP_FOUND)
    AND (GTKmm_FOUND)
    AND (assimp_FOUND) )
    
    set( THIS_TARGET "mpblocks-test-gjk-demo-3d" )
    
    include_directories( 
        ${Eigen3_INCLUDE_DIR} 
        ${TinyXML2_INCLUDE_DIR}
        ${YAMLCPP_INCLUDE_DIR}
        ${MPBlocks_INCLUDE_DIR}
        ${GTKmm_INCLUDE_DIRS}
        ${assimp_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}
        )
    
    configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/config.cpp
                    ${CMAKE_CURRENT_BINARY_DIR}/config.cpp )
    
    add_executable( ${THIS_TARGET}
                     main.cpp
                     Main.cpp
                     QuaternionWidget.cpp
                     ${CMAKE_CURRENT_BINARY_DIR}/config.cpp )

    target_link_libraries( ${THIS_TARGET}
                ${MPBlocks_gtk_LIBRARY}
                ${TinyXML2_LIBRARY}
                ${YAMLCPP_LIBRARY}
                ${assimp_LIBRARY}
                ${GTKmm_LIBRARIES} )
                     
endif()