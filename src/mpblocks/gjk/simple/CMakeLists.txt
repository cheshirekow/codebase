


find_package( Eigen3 )

include_directories( ${Eigen3_INCLUDE_DIR} )

add_executable( mpblocks-test-gjk-simple main.cpp )