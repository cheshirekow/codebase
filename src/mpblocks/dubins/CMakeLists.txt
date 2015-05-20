add_subdirectory(curves)
add_subdirectory(hyper)
add_subdirectory(planner)

add_subdirectory(curves_eigen)
cc_test(mpblocks_dubins_packed_test SOURCES packed_test.cc)
