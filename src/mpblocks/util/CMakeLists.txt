configure_file(path_util.cc.in
               ${CMAKE_CURRENT_BINARY_DIR}/path_util.cc
               @ONLY)

add_library(mpblocks_path_util
    ${CMAKE_CURRENT_BINARY_DIR}/path_util.cc)

add_library(mpblocks_util
    timespec.cc)

cc_test(mpblocks_binary_literal_test binary_literal_test.cc)
cc_test(mpblocks_set_operations_test set_operations_test.cc)