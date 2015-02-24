if((Cairomm_FOUND)
   AND (Eigen3_FOUND)
   AND (GTKmm_FOUND)
   AND (TinyXML2_FOUND)
   AND (YAMLCPP_FOUND))

  include_directories(${CMAKE_CURRENT_SOURCE_DIR}
                      ${Eigen3_INCLUDE_DIR}
                      ${Cairomm_INCLUDE_DIRS}
                      ${GTKmm_INCLUDE_DIRS}
                      ${TinyXML2_INCLUDE_DIRS}
                      ${YAMLCPP_INCLUDE_DIR}) 

  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config.cpp
                 ${CMAKE_CURRENT_BINARY_DIR}/config.cpp)

  add_executable(mpblocks_delaunay_brown79
                 ${CMAKE_CURRENT_BINARY_DIR}/config.cpp
                 brown_main.cpp)

  target_link_libraries(mpblocks_delaunay_brown79 
                        ${GTKmm_LIBRARIES}
                        ${SigC++_LIBRARIES}
                        ${YAMLCPP_LIBRARY}
                        ${TinyXML2_LIBRARIES}
                        ${MPBlocks_gtk_LIBRARY} )
else()
    message( WARNING "Missing required packages for compile test. Packages: "
                     "${MISSING} " )
endif()

