install(FILES layout.glade
        DESTINATION share/mpblocks/
        RENAME dubins_planner_demo.glade)

if((Boost_FILESYSTEM_FOUND)
    AND (Cairomm_FOUND)
    AND (Eigen3_FOUND)
    AND (GTKmm_FOUND)
    AND (TinyXML2_FOUND)
    AND (YAMLCPP_FOUND)
    AND (re2_FOUND))

  include_directories(${CMAKE_CURRENT_SOURCE_DIR}
                      ${BOOST_INCLUDE_DIRS}
                      ${Eigen3_INCLUDE_DIRS}
                      ${GTKmm_INCLUDE_DIRS}
                      ${Cairomm_INCLUDE_DIRS}
                      ${TinyXML2_INCLUDE_DIR}
                      ${YAMLCPP_INCLUDE_DIRS}
                      ${re2_INCLUDE_DIRS})

  add_executable(mpblocks_dubins_planner_demo
                 main.cc)
  target_link_libraries(mpblocks_dubins_planner_demo
                        ${Boost_LIBRARIES}
                        ${GTKmm_LIBRARIES}
                        ${Cairomm_LIBRARIES}
                        ${TinyXML2_LIBRARY}
                        ${YAMLCPP_LDFLAGS}
                        ${re2_LDFLAGS}
                        mpblocks_gtk
                        mpblocks_path_util)
else()
  message( WARNING "Wont build dubins planner demo.")
endif()

