set( TARGET mpblocks-robot_nn-euclidean )

if( (Eigen3_FOUND)
    AND (CUDA_FOUND)
    AND (MPBlocks_FOUND) )
    
    include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${Eigen3_INCLUDE_DIR}
        ${MPBlocks_INCLUDE_DIR}
        ${CUDA_INCLUDE_DIRS} )

    cuda_add_executable( ${TARGET}
                         euclidean.cpp
                         impl/rqt.cpp )
                         
    target_link_libraries( ${TARGET}
        ${MPBlocks_utility_LIBRARY}
        ${MPBlocks_cudaNN_LIBRARY}
        ${MPBlocks_cuda_LIBRARY}
        ${CUDA_CUDART_LIBRARY}
        ${CUDA_CUDA_LIBRARY} )
    
else()
    message( WARNING "Wont build ${TARGET}" )
endif()