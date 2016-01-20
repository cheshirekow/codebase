find_package( CUDA )
find_package( Boost )

cc_require(CUDA Boost)
if(required_FOUND)

    include_directories(
        ${CUDA_TOOLKIT_INCLUDE}
    )
    
    set( LIBS "${LIBS}"
         mpblocks_cuda
         mpblocks_util_static
         ${CUDA_CUDART_LIBRARY}
         ${CUDA_CUDA_LIBRARY}
         -lrt
         ) 
         
    cuda_add_executable( mpblocks_cuda_bitonic_test 
                            test.cu
                            test.cpp )
    
    target_link_libraries( 
        mpblocks_cuda_bitonic_test ${LIBS} )
        
        

else()
    mark_excluded("cuda bitonic test: missing ${required_MISSING}")    
endif()
