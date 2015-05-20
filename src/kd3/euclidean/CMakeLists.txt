
cc_library(kd3_euclidean
           HEADERS distance.h
                   euclidean.cc)

cc_test(kd3_euclidean_distance_test SOURCES distance_test.cc)
