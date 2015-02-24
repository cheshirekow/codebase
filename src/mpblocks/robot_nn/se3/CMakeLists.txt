set( TARGET mpblocks-robot_nn-se3 )

if( (Eigen3_FOUND)
    AND (CUDA_FOUND)
    AND (MPBlocks_FOUND) )
    
    include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${Eigen3_INCLUDE_DIR}
        ${MPBlocks_INCLUDE_DIR}
        ${CUDA_INCLUDE_DIRS} )

    cuda_add_executable( ${TARGET}
                         se3.cpp
                         impl/gpu_bf.cpp
                         impl/cpu_bf.cpp
                         impl/cpu_rqt.cpp
                         impl/gpu_rqt.cpp
                         impl/cpu_sqt.cpp
                         impl/gpu_sqt.cpp
                         impl/build_schedule.cpp
                          )
                         
    target_link_libraries( ${TARGET}
        ${MPBlocks_utility_LIBRARY}
        ${MPBlocks_cudaNN_LIBRARY}
        ${MPBlocks_cuda_LIBRARY}
        ${CUDA_CUDART_LIBRARY}
        ${CUDA_CUDA_LIBRARY} )
    
else()
    message( WARNING "Wont build ${TARGET}" )
endif()


set( TARGET mpblocks-robot_nn-so3_test )

if( (Eigen3_FOUND)
    AND (CUDA_FOUND)
    AND (MPBlocks_FOUND) )
    
    include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${Eigen3_INCLUDE_DIR}
        ${MPBlocks_INCLUDE_DIR}
        ${CUDA_INCLUDE_DIRS} )

    cuda_add_executable( ${TARGET}
                         so3_test.cpp
                          )
                         
    target_link_libraries( ${TARGET}
        ${MPBlocks_utility_LIBRARY}
        ${MPBlocks_cudaNN_LIBRARY}
        ${MPBlocks_cuda_LIBRARY}
        ${CUDA_CUDART_LIBRARY}
        ${CUDA_CUDA_LIBRARY} )
    
else()
    message( WARNING "Wont build ${TARGET}" )
endif()