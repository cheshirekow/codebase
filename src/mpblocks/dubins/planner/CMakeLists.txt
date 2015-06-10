install(FILES layout.glade
        DESTINATION share/mpblocks/
        RENAME dubins_planner_demo.glade)

if((Boost_FILESYSTEM_FOUND)
    AND (cairomm_FOUND)
    AND (eigen3_FOUND)
    AND (GTKmm_FOUND)
    AND (TinyXML2_FOUND)
    AND (yamlcpp_FOUND)
    AND (re2_FOUND))

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
  message(WARNING "Cant build dubins planner demo.")
endif()

