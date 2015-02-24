find_package( Eigen3 )
find_package( SigC++ )
find_package( MPBlocks  )

if( (Eigen3_FOUND)
    AND (SigC++_FOUND)
    AND (MPBlocks_FOUND )
    )
    
    include_directories(
        ${Eigen3_INCLUDE_DIR}
        ${SigC++_INCLUDE_DIRS}
        ${MPBlocks_INCLUDE_DIRS}
        ) 
        
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x" )
        
    set( LIBS ${LIBS}
        ${SigC++_LIBRARIES} )
    
    add_executable( mpblocks-delaunay-compile-test
                        justcompile.cpp )
                        
    target_link_libraries( mpblocks-delaunay-compile-test ${LIBS} )
else()
    set(MISSING "")
    
    if( NOT (Eigen3_FOUND) )
        set(MISSING "${MISSING}, eigen3")
    endif()
    
    if( NOT (MPBlocks_FOUND) )
        set(MISSING "${MISSING}, mpblocks::convex_hull::clarkson")
    endif()
    
    message( WARNING "Missing required packages for compile test. Packages: "
                     "${MISSING} " )
endif()

add_subdirectory( demo2 )
