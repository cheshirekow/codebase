install(FILES demo_2d.glade
        DESTINATION share/mpblocks/
        RENAME edelsbrunner96_demo_2d.glade)
install(FILES barycentric_demo_2d.glade
        DESTINATION share/mpblocks/
        RENAME edelsbrunner96_barycentric_demo_2d.glade)

cc_require(Boost_FILESYSTEM cairomm eigen3 GTKmm TinyXML2 yamlcpp re2)
if(required_FOUND)
  include_directories(../../include/mpblocks)
  cc_include(Boost eigen3 GTKmm cairomm TinyXML2 yamlcpp re2)

  cc_executable(edelsbrunner96_demo_2d
                SOURCES demo_2d.cc
                CMAKE_DEPENDS Boost
                PKG_DEPENDS GTKmm cairomm TinyXML2 yamlcpp re2
                TARGET_DEPENDS mpblocks_gtk mpblocks_path_util_static
                RAW_DEPENDS ${CMAKE_THREAD_LIBS_INIT})

  cc_executable(edelsbrunner96_barycentric_demo_2d
                SOURCES barycentric_demo_2d.cc
                CMAKE_DEPENDS Boost
                PKG_DEPENDS GTKmm cairomm TinyXML2 yamlcpp re2
                TARGET_DEPENDS mpblocks_gtk mpblocks_path_util_static
                RAW_DEPENDS ${CMAKE_THREAD_LIBS_INIT})
else()
  message(WARNING "Can't build edelsbrunner96 demos, missing "
                  ${required_MISSING})
endif()

