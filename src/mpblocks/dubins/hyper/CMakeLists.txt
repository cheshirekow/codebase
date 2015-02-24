install(FILES layout.glade
        DESTINATION ${CMAKE_INSTALL_PREFIX}/share/mpblocks/
        RENAME dubins_hyper_demo.glade)

if((Cairomm_FOUND)
    AND (Eigen3_FOUND)
    AND (GTKmm_FOUND)
    AND (TinyXML2_FOUND)
    AND (YAMLCPP_FOUND))

  include_directories(${CMAKE_CURRENT_SOURCE_DIR}
                      ${Eigen3_INCLUDE_DIR}
                      ${GTKmm_INCLUDE_DIRS}
                      ${Cairomm_INCLUDE_DIRS}
                      ${TinyXML2_INCLUDE_DIR}
                      ${YAMLCPP_INCLUDE_DIRS})

  set(LIBS ${LIBS}
      mpblocks_gtk
      mpblocks_path_util
      ${GTKmm_LIBRARIES}
      ${Cairomm_LIBRARIES}
      ${TinyXML2_LIBRARY}
      ${YAMLCPP_LDFLAGS})

  add_executable(mpblocks_dubins_hyper_demo 
                 main.cpp)
  target_link_libraries(mpblocks_dubins_hyper_demo ${LIBS})
else()
  message( WARNING "Missing packages, wont build dubbins hyper demo.")
endif()
