if(eigen3_FOUND)
  cc_include(eigen3)

  cc_library(kd3
             HEADERS enumerator.h
                     hyperrect.h
                     kd3.cc
                     node.h
                     tree.h)

  macro(kd3_test test_name)
    cc_test(kd3_${test_name} SOURCES ${test_name}.cc)
  endmacro()

  kd3_test(hyperrect_test)
  kd3_test(node_test)
  kd3_test(tree_test)

  add_subdirectory(euclidean)
else()
  message(WARNING "Can't build kd3")
endif()

#    set(LIBS ${LIBS} ${SigC++_LIBRARIES})
    
#    # adds samplerTest to the list of executables to generate
#    add_executable( mpblocks_kdtree_test
#            kdtree_test.cpp
#            kdtree.c )
            
#    add_executable( mpblocks_kdtree_test2
#            kdtree_test2.cpp
#            kdtree.c )
            
#    add_executable( mpblocks_kd2_test
#            kd2test.cpp
#            kdtree.c )
            
#    # sets the list of libraries that samplerTest needs to link against
#    target_link_libraries(mpblocks_kdtree_test ${LIBS})
#    target_link_libraries(mpblocks_kdtree_test2 ${LIBS})

#    add_executable( tplTest tpltest.cpp )
#    target_link_libraries(tplTest ${LIBS})
