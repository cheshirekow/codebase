set( TARGET_NAME "mpblocks_cudaCert_kprops" )

if( (MPBlocks_FOUND)
    AND (CUDA_FOUND)
    AND (Eigen3_FOUND) )

    include_directories(
        ${MPBlocks_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
        ${Eigen3_INCLUDE_DIR}
    )

    set( TARGET_NAME "mpblocks_cudaCert_kprops" )    
    cuda_add_executable( ${TARGET_NAME}
                    kprops.cpp
                     )
                    
    target_link_libraries( ${TARGET_NAME}
        ${MPBlocks_cuda_LIBRARY}
        ${CUDA_CUDA_LIBRARY}
        ${CUDA_CUDART_LIBRARY}
        mpblocks_cudaCert
    )
    
    set( TARGET_NAME "mpblocks_cudaCert_CertSetTest" )    
    cuda_add_executable( ${TARGET_NAME}
                    CertSetTest.cpp
                     )
                    
    target_link_libraries( ${TARGET_NAME}
        ${MPBlocks_cuda_LIBRARY}
        ${CUDA_CUDA_LIBRARY}
        ${CUDA_CUDART_LIBRARY}
        mpblocks_cudaCert
    )
    
else()
    message( WARNING "Skipping target ${TARGET_NAME}" )
endif()

