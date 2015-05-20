configure_file(path_util.cc.in
               ${CMAKE_CURRENT_BINARY_DIR}/path_util.cc
               @ONLY)

cc_library(mpblocks_path_util
           SOURCES ${CMAKE_CURRENT_BINARY_DIR}/path_util.cc)

cc_library(mpblocks_util
           SOURCES timespec.cc)

cc_test(mpblocks_binary_literal_test SOURCES binary_literal_test.cc)
cc_test(mpblocks_set_operations_test SOURCES set_operations_test.cc)
