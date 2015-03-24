find_package( Eigen3 )

if( (Eigen3_FOUND) )
    
    include_directories( 
        ${CMAKE_CURRENT_SOURCE_DIR} 
        ${Eigen3_INCLUDE_DIR} )
        
    set( LIBS ${LIBS} )
    
    add_executable( mpblocks_simplex_tree_compile_test
                        compile.cpp )

    target_link_libraries( mpblocks_simplex_tree_compile_test
                            ${LIBS} )                        
else()

    set(MISSING "")
    
    if( NOT (Eigen3_FOUND) )
        set(MISSING, "${MISSING}, eigen3")
    endif()
    
    message(WARNING "can't compile simplex_tree headers, missing: ${MISSING}")
endif()