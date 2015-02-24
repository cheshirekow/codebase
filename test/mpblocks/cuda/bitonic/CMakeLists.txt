find_package( CUDA )
find_package( MPBlocks COMPONENTS utility )
find_package( Boost )

if( CUDA_FOUND 
    AND (MPBlocks_FOUND)
    AND (Boost_FOUND ) )

    include_directories(
        ${CUDA_TOOLKIT_INCLUDE}
        ${MPBlocks_INCLUDE_DIR}
    )
    
    set( LIBS "${LIBS}"
         mpblocks_cuda
         ${MPBlocks_utility_LIBRARY}
         ${CUDA_LIBRARY}
         ${CUDA_RT_LIBRARY}
         -lrt
         ) 
         
    cuda_add_executable( mpblocks_cuda_bitonic_test 
                            test.cu
                            test.cpp )
    
    target_link_libraries( 
        mpblocks_cuda_bitonic_test ${LIBS} )
        
        

else()

    set( MISSING "" )
    
    if( NOT (CUDA_FOUND) )
        set( MISSING "cuda, ${MISSING}" )
    endif()
    
    if( NOT (MPBlocks_FOUND) )
        set( MISSING "mpblocks, ${MISSING}" )
    endif()
    
    if( NOT (Boost_FOUND) )
        set( MISSING "boost, ${MISSING}" )
    endif()
    
    message( WARNING "can't build mpblocks_cuda_bitonic_test,"
                     "missing ${MISSING}" )

endif()
