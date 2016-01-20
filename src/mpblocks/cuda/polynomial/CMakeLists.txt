
add_executable(mpblocks-cuda-polynomial-intlist-test intlist.cpp )
add_executable(mpblocks-cuda-polynomial-test basic.cpp )
                    
if( CUDA_FOUND  )
    include_directories(${CUDA_TOOLKIT_INCLUDE})
    
    set( LIBS "${LIBS}"
         mpblocks_cuda
         ${CUDA_CUDA_LIBRARY}
         ${CUDA_CUDART_LIBRARY}
         -lrt
         ) 
         
    cuda_add_executable(mpblocks-cuda-polynomial-ktest 
                        ktest.cpp
                        ktest.cu )
    
    target_link_libraries(mpblocks-cuda-polynomial-ktest ${LIBS})
else()
    mark_excluded("cuda polynomial ktest: missingcuda" )
endif()
