set(LIBRARY_HEADERS facet.h
                    facet.hpp
                    induced_subcomplex.h
                    induced_subcomplex.hpp
                    iterators.h
                    iterators.hpp
                    line_walker.h
                    line_walker.hpp
                    simplex.h
                    simplex.hpp
                    triangulation.h
                    triangulation.hpp)

install(FILES ${LIBRARY_HEADERS}
        DESTINATION include/edelsbrunner96)

if(eigen3_FOUND)
  include_directories(${eigen3_INCLUDE_DIRS})
  cc_test(edelsbrunner96_feature_walk_test feature_walk_test.cc)
  cc_test(edelsbrunner96_induced_subcomplex_test induced_subcomplex_test.cc)
  cc_test(edelsbrunner96_insert_test insert_test.cc)
  cc_test(edelsbrunner96_iterator_test iterator_test.cc)
  cc_test(edelsbrunner96_line_walk_test line_walk_test.cc)
  cc_test(edelsbrunner96_simplex_test simplex_test.cc)
  cc_test(edelsbrunner96_simplex_distance_test simplex_distance_test.cc)
  cc_test(edelsbrunner96_triangulation_test triangulation_test.cc)
  cc_test(edelsbrunner96_visible_hull_test visible_hull_test.cc)
else()
  message(WARNING "wont build edelsbrunner96 tests")
endif()

