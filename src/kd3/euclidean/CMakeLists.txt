
add_library(kd3_euclidean
            distance.h
            euclidean.cc)

cc_test(kd3_euclidean_distance_test distance_test.cc)
