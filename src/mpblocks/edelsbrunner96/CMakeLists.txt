install(FILES demo_2d.glade
        DESTINATION share/mpblocks/
        RENAME edelsbrunner96_demo_2d.glade)
install(FILES barycentric_demo_2d.glade
        DESTINATION share/mpblocks/
        RENAME edelsbrunner96_barycentric_demo_2d.glade)

if((Boost_FILESYSTEM_FOUND)
    AND (Cairomm_FOUND)
    AND (Eigen3_FOUND)
    AND (GTKmm_FOUND)
    AND (TinyXML2_FOUND)
    AND (YAMLCPP_FOUND)
    AND (re2_FOUND))

  include_directories(../../include/mpblocks)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}
                      ${BOOST_INCLUDE_DIRS}
                      ${Eigen3_INCLUDE_DIRS}
                      ${GTKmm_INCLUDE_DIRS}
                      ${Cairomm_INCLUDE_DIRS}
                      ${TinyXML2_INCLUDE_DIR}
                      ${YAMLCPP_INCLUDE_DIRS}
                      ${re2_INCLUDE_DIRS})

  add_executable(mpblocks_edelsbrunner96_demo_2d
                 demo_2d.cc)
  target_link_libraries(mpblocks_edelsbrunner96_demo_2d
                        ${Boost_LIBRARIES}
                        ${GTKmm_LIBRARIES}
                        ${Cairomm_LIBRARIES}
                        ${TinyXML2_LIBRARY}
                        ${YAMLCPP_LDFLAGS}
                        ${re2_LDFLAGS}
                        mpblocks_gtk
                        mpblocks_path_util)

  add_executable(mpblocks_edelsbrunner96_barycentric_demo_2d
                 barycentric_demo_2d.cc)
  target_link_libraries(mpblocks_edelsbrunner96_barycentric_demo_2d
                        ${Boost_LIBRARIES}
                        ${GTKmm_LIBRARIES}
                        ${Cairomm_LIBRARIES}
                        ${TinyXML2_LIBRARY}
                        ${YAMLCPP_LDFLAGS}
                        ${re2_LDFLAGS}
                        mpblocks_gtk
                        mpblocks_path_util)
else()
  message( WARNING "Wont build edelsbrunner96 2d demo.")
endif()

