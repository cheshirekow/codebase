
add_executable( mpblocks-cuda-polynomial-intlist-test
                    intlist.cpp )
                    
add_executable( mpblocks-cuda-polynomial-test
                    basic.cpp )
                    
find_package( CUDA )


set( TARGET mpblocks-cuda-polynomial-ktest )
if( CUDA_FOUND  )

    include_directories(
        ${CUDA_TOOLKIT_INCLUDE}
    )
    
    set( LIBS "${LIBS}"
         mpblocks_cuda
         ${CUDA_LIBRARY}
         ${CUDA_RT_LIBRARY}
         -lrt
         ) 
         
    cuda_add_executable( ${TARGET} 
                            ktest.cpp
                            ktest.cu )
    
    target_link_libraries( ${TARGET} ${LIBS} )
else()

    message( WARNING "can't build ${TARGET}, missing cuda" )

endif()