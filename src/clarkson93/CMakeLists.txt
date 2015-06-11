cc_library(clarkson93
           HEADERS bit_member.h
                   dynamic/simplex.h
                   dynamic/simplex_impl.h
                   dynamic/triangulation.h
                   dynamic/triangulation_impl.h
                   example_traits2.h
                   example_traits.h
                   horizon_ridge.h
                   indexed.h
                   priority_queue.h
                   simplex2.h
                   simplex2_impl.h
                   simplex.h
                   simplex_impl.h
                   stack_set.h
                   triangulation.h
                   triangulation_impl.h
           SOURCES
           PKG_DEPENDS eigen3)

macro(clarkson93_test test_name)
  cc_test(clarkson93_${test_name}_test SOURCES ${test_name}_test.cc)
endmacro()

clarkson93_test(bit_member)
clarkson93_test(priority_queue)
clarkson93_test(stack_set)
