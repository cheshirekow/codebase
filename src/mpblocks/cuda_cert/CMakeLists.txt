if((MPBlocks_FOUND)
    AND (CUDA_FOUND)
    AND (Eigen3_FOUND))

  include_directories(${MPBlocks_INCLUDE_DIRS}
                      ${CUDA_INCLUDE_DIRS}
                      ${Eigen3_INCLUDE_DIR})
    
  cuda_add_library(mpblocks_cudaCert
                   kernels.cu
                   kernels2.cu
                   CertSet.cu
                   CertSet.cpp
                   CertSet2.cu
                   CertSet2.cpp
                   fattr.cu)

  set(LIBS ${MPBlocks_cuda_LIBRARY}
           ${CUDA_CUDA_LIBRARY}
           ${CUDA_CUDART_LIBRARY})

  target_link_libraries(mpblocks_cudaCert ${LIBS})
    
  install(TARGETS mpblocks_cudaCert
          RUNTIME DESTINATION bin
          LIBRARY DESTINATION lib
          ARCHIVE DESTINATION lib )

  cuda_add_executable(mpblocks_cudaCert_kprops kprops.cpp)
  target_link_libraries(mpblocks_cudaCert_kprops mpblocks_cudaCert ${LIBS})
    
  cuda_add_executable(mpblocks_cudaCert_CertSetTest CertSetTest.cpp)
  target_link_libraries(mpblocks_cudaCert_CertSetTest 
                        mpblocks_cudaCert ${LIBS})                
else()
  message(WARNING "Skipping cuda cert")
endif()

