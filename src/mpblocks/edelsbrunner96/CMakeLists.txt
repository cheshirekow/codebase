install(FILES demo_2d.glade
        DESTINATION share/mpblocks/
        RENAME edelsbrunner96_demo_2d.glade)
install(FILES barycentric_demo_2d.glade
        DESTINATION share/mpblocks/
        RENAME edelsbrunner96_barycentric_demo_2d.glade)

if((Boost_FILESYSTEM_FOUND)
    AND (cairomm_FOUND)
    AND (eigen3_FOUND)
    AND (GTKmm_FOUND)
    AND (TinyXML2_FOUND)
    AND (yamlcpp_FOUND)
    AND (re2_FOUND))

  include_directories(../../include/mpblocks)
  cc_include(Boost eigen3 GTKmm cairomm TinyXML2 yamlcpp re2)

  cc_executable(mpblocks_edelsbrunner96_demo_2d
                SOURCES demo_2d.cc
                CMAKE_DEPENDS Boost
                PKG_DEPENDS GTKmm cairomm TinyXML2 yamlcpp re2
                TARGET_DEPENDS mpblocks_gtk mpblocks_path_util_static)

  cc_executable(mpblocks_edelsbrunner96_barycentric_demo_2d
                SOURCES barycentric_demo_2d.cc
                CMAKE_DEPENDS Boost
                PKG_DEPENDS GTKmm cairomm TinyXML2 yamlcpp re2
                TARGET_DEPENDS mpblocks_gtk mpblocks_path_util_static)
else()
  message( WARNING "Wont build edelsbrunner96 2d demo.")
endif()

