install(FILES layout.glade
        DESTINATION ${CMAKE_INSTALL_PREFIX}/share/mpblocks/
        RENAME dubins_hyper_demo.glade)

if((cairomm_FOUND)
    AND (eigen3_FOUND)
    AND (GTKmm_FOUND)
    AND (TinyXML2_FOUND)
    AND (yamlcpp_FOUND))

  include_directories(${CMAKE_CURRENT_SOURCE_DIR}
                      ${eigen3_INCLUDE_DIRS}
                      ${GTKmm_INCLUDE_DIRS}
                      ${cairomm_INCLUDE_DIRS}
                      ${TinyXML2_INCLUDE_DIRS}
                      ${yamlcpp_INCLUDE_DIRS})

  set(LIBS ${LIBS}
      mpblocks_gtk
      mpblocks_path_util_static
      ${GTKmm_LDFLAGS}
      ${cairomm_LDFLAGS}
      ${TinyXML2_LDFLAGS}
      ${yamlcpp_LDFLAGS})

  add_executable(mpblocks_dubins_hyper_demo 
                 main.cpp)
  target_link_libraries(mpblocks_dubins_hyper_demo ${LIBS})
else()
  message( WARNING "Missing packages, wont build dubbins hyper demo.")
endif()
