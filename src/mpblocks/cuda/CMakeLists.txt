if(CUDA_FOUND)
  include_directories(${CUDA_TOOLKIT_INCLUDE})
  cuda_add_library(mpblocks_cuda STATIC
                   wrap.cc)

  install(TARGETS mpblocks_cuda
          RUNTIME DESTINATION bin
          LIBRARY DESTINATION lib
          ARCHIVE DESTINATION lib)

  add_subdirectory(powers_of_two)
  add_subdirectory(linalg)
  add_subdirectory(linalg2)
  add_subdirectory(bitonic)
  add_subdirectory(polynomial)
else()
  message( WARNING "can't build mpblocks_cuda, missing ${MISSING}" )
endif()
