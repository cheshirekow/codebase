install(FILES layout.glade
        DESTINATION share/mpblocks/
        RENAME dubins_planner_demo.glade)

cc_require(Boost_FILESYSTEM cairomm eigen3 GTKmm TinyXML2 yamlcpp re2)
if(required_FOUND)
  cc_include(Boost
             eigen3
             GTKmm
             cairomm
             TinyXML2
             yamlcpp
             re2)

  cc_executable(mpblocks_dubins_planner_demo SOURCES main.cc
                CMAKE_DEPENDS Boost
                PKG_DEPENDS eigen3 GTKmm cairomm TinyXML2 yamlcpp re2
                TARGET_DEPENDS mpblocks_gtk mpblocks_path_util_static
                RAW_DEPENDS ${CMAKE_THREAD_LIBS_INIT})
else()
  mark_excluded("dubins planner demo: missing ${required_MISSING}")
endif()

