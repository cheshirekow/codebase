find_package(SigC++)
find_package(Eigen3)

if( (SigC++_FOUND) 
    AND (Eigen3_FOUND)  )

    include_directories( 
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${SigC++_INCLUDE_DIRS}
        ${Eigen3_INCLUDE_DIR}  
        )
    
    # adds the libsigc++ libraries to the list
    set(LIBS ${LIBS} ${SigC++_LIBRARIES})
    
    # adds samplerTest to the list of executables to generate
    add_executable( mpblocks_kdtree_test 
            kdtree_test.cpp 
            kdtree.c )
            
    add_executable( mpblocks_kdtree_test2 
            kdtree_test2.cpp 
            kdtree.c )
            
    add_executable( mpblocks_kd2_test
            kd2test.cpp
            kdtree.c )
            
    # sets the list of libraries that samplerTest needs to link against
    target_link_libraries(mpblocks_kdtree_test ${LIBS})
    target_link_libraries(mpblocks_kdtree_test2 ${LIBS})

    add_executable( tplTest tpltest.cpp )    
    target_link_libraries(tplTest ${LIBS})
else()
 
    set( MISSING "" )
    
    if( NOT (SigC++_FOUND) )
        set( MISSING ${MISSING} "sigc++, " )
    endif()
    
    if( NOT (Eigen3_FOUND) )
        set( MISSING ${MISSING} "eigen3, " )
    endif()
    
    message( WARNING "missing required libraries, wont build the test, "
                     "missing: ${MISSING}")

endif()

