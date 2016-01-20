cc_library(fiber
           HEADERS axis_angle.h
                   cross_matrix.h
                   difference.h
                   euler_angles.h
                   fiber.h
                   identity.h
                   lvalue.h
                   matrix.h
                   normalize.h
                   ostream.h
                   product.h
                   quaternion.h
                   rotation_conversions.h
                   rvalue.h
                   scale.h
                   stream_assignment.h
                   sum.h
                   transpose.h
                   view.h)

# TODO: Change fiber to use static const instead of enum so we can
# allow this warning
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-enum-compare")

# All thests for this library are a single source file and link the library
# itself
macro(fiber_test test_name)
  cc_test(fiber_${test_name} SOURCES ${test_name}.cc)
endmacro()

if(eigen3_FOUND)
  cc_include(eigen3)
  fiber_test(axis_angle_test)
  fiber_test(block_test)
  fiber_test(composition_test)
  fiber_test(difference_test)
  fiber_test(euler_angles_test)
  fiber_test(product_test)
  fiber_test(quaternion_test)
  fiber_test(rotation_conventions_test)
  fiber_test(scale_test)
  fiber_test(sum_test)
  fiber_test(transpose_test)
else()
  mark_excluded("fiber tests: missing eigen3")
endif()

cc_executable(fiber_demo SOURCES demo.cc TARGET_DEPENDS fiber_static)
