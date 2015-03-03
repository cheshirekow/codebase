# TODO(bialkowski): Change fiber to use static const instead of enum so we can
# allow this warning
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-enum-compare")

if(eigen3_FOUND)
  include_directories(${eigen3_INCLUDE_DIRS})
  cc_test(fiber_axis_angle_test axis_angle_test.cc)
  cc_test(fiber_block_test block_test.cc)
  cc_test(fiber_composition_test composition_test.cc)
  cc_test(fiber_difference_test difference_test.cc)
  cc_test(fiber_euler_angles_test euler_angles_test.cc)
  cc_test(fiber_product_test product_test.cc)
  cc_test(fiber_quaternion_test quaternion_test.cc)
  cc_test(fiber_rotation_conventions_test rotation_conventions_test.cc)
  cc_test(fiber_scale_test scale_test.cc)
  cc_test(fiber_sum_test sum_test.cc)
  cc_test(fiber_transpose_test transpose_test.cc)
else()
  message(WARNING "wont build fiber tests")
endif()



