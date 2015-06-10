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
                   p_queue.h
                   simplex2.h
                   simplex2_impl.h
                   simplex.h
                   simplex_impl.h
                   static_stack.h
                   triangulation.h
                   triangulation_impl.h
           SOURCES
           PKG_DEPENDS eigen3)
