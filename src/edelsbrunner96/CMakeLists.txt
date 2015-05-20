cc_library(edelsbrunner96
           HEADERS facet.h
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
# All thests for this library are a single source file and link the library
# itself
macro(edelsbrunner96_test test_name)
  cc_test(edelsbrunner96_${test_name} SOURCES ${test_name}.cc)
endmacro()

if(eigen3_FOUND)
  cc_include(eigen3)
  edelsbrunner96_test(feature_walk_test)
  edelsbrunner96_test(induced_subcomplex_test)
  edelsbrunner96_test(insert_test)
  edelsbrunner96_test(iterator_test)
  edelsbrunner96_test(line_walk_test)
  edelsbrunner96_test(simplex_test)
  edelsbrunner96_test(simplex_distance_test)
  edelsbrunner96_test(triangulation_test)
  edelsbrunner96_test(visible_hull_test)
else()
  message(WARNING "wont build edelsbrunner96 tests")
endif()
