set( TARGET_NAME "mpblocks_cudaNN_test" )

if( (MPBlocks_FOUND)
    AND (CUDA_FOUND)
    AND (Eigen3_FOUND) )

    include_directories(
        ${MPBlocks_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
        ${Eigen3_INCLUDE_DIR}
    )
    
    cuda_add_executable( ${TARGET_NAME} main.cpp )
                    
    target_link_libraries( ${TARGET_NAME}
        mpblocks_cudaNN
        ${MPBlocks_cuda_LIBRARY}
        ${CUDA_CUDA_LIBRARY}
        ${CUDA_CUDART_LIBRARY}
    )
    
else()
    message( WARNING "Skipping target ${TARGET_NAME}" )
endif()

