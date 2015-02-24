install(FILES layout.glade
        DESTINATION share/mpblocks/
        RENAME dubins_curves_demo.glade)

if((Cairomm_FOUND)
    AND (Eigen3_FOUND)
    AND (GTKmm_FOUND)
    AND (TinyXML2_FOUND)
    AND (YAMLCPP_FOUND))

  include_directories(${CMAKE_CURRENT_SOURCE_DIR}
                      ${Eigen3_INCLUDE_DIRS}
                      ${GTKmm_INCLUDE_DIRS}
                      ${Cairomm_INCLUDE_DIRS}
                      ${TinyXML2_INCLUDE_DIR}
                      ${YAMLCPP_INCLUDE_DIRS})

  set(LIBS ${LIBS}
      ${GTKmm_LIBRARIES}
      ${Cairomm_LIBRARIES}
      ${TinyXML2_LIBRARY}
      ${YAMLCPP_LDFLAGS}
      mpblocks_gtk
      mpblocks_path_util)

  set(SRCS
      main.cc
      solvers.cc)

  if(CUDA_FOUND)
    include_directories(${CUDA_INCLUDE_DIRS})
    cuda_add_executable(mpblocks_dubins_curves_cuda_demo ${SRCS}
                        cuda/with_cuda.cc
                        cuda/with_cuda.cu)

    set_source_files_properties(main.cc
                                solvers.cc
                                cuda/with_cuda.cc
                                PROPERTIES COMPILE_FLAGS -std=c++11)
    target_link_libraries(mpblocks_dubins_curves_cuda_demo ${LIBS}
                          ${CUDA_CUDART_LIBRARY}
                          ${CUDA_CUDA_LIBRARY}
                          mpblocks_cuda)
  else()
    message( WARNING "Can't find cuda, compiling dubins demo without" )
  endif()

  add_executable(mpblocks_dubins_curves_demo ${SRCS}
                 cuda/without_cuda.cc)
  target_link_libraries(mpblocks_dubins_curves_demo ${LIBS})
else()
  message( WARNING "Wont build dubins curves demo.")
endif()

