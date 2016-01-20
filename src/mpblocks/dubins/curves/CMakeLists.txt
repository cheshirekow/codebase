install(FILES layout.glade
        DESTINATION share/mpblocks/
        RENAME dubins_curves_demo.glade)


cc_require(cairomm eigen3 GTKmm TinyXML2 yamlcpp)
if(required_FOUND)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}
                      ${eigen3_INCLUDE_DIRS}
                      ${GTKmm_INCLUDE_DIRS}
                      ${cairomm_INCLUDE_DIRS}
                      ${TinyXML2_INCLUDE_DIR}
                      ${yamlcpp_INCLUDE_DIRS})

  set(LIBS ${LIBS}
      ${GTKmm_LDFLAGS}
      ${cairomm_LDFLAGS}
      ${TinyXML2_LDFLAGS}
      ${yamlcpp_LDFLAGS}
      mpblocks_gtk
      mpblocks_path_util_static)

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
  mark_excluded("dubins curves demo: missing ${required_MISSING}")  
endif()

