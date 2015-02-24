set( TARGET_NAME "mpblocks_cudaCert" )

if( (MPBlocks_FOUND)
    AND (CUDA_FOUND)
    AND (Eigen3_FOUND) )

    include_directories(
        ${MPBlocks_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
        ${Eigen3_INCLUDE_DIR}
    )
    
    cuda_add_library( ${TARGET_NAME}
                    kernels.cu
                    kernels2.cu
                    CertSet.cu
                    CertSet.cpp
                    CertSet2.cu
                    CertSet2.cpp
                    fattr.cu
                     )
                    
    target_link_libraries( ${TARGET_NAME}
        ${MPBlocks_cuda_LIBRARY}
        ${CUDA_CUDA_LIBRARY}
        ${CUDA_CUDART_LIBRARY}
    )
    
    install( TARGETS ${TARGET_NAME}
             RUNTIME DESTINATION bin
             LIBRARY DESTINATION lib
             ARCHIVE DESTINATION lib )
else()
    message( WARNING "Skipping target ${TARGET_NAME}" )
endif()

