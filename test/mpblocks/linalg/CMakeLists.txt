if(eigen3_FOUND)
  include_directories(${eigen3_INCLUDE_DIRS})
  cc_test(linalg_axis_angle_test axis_angle_test.cc)
  cc_test(linalg_block_test block_test.cc)
  cc_test(linalg_composition_test composition_test.cc)
  cc_test(linalg_difference_test difference_test.cc)
  cc_test(linalg_euler_angles_test euler_angles_test.cc)
  cc_test(linalg_product_test product_test.cc)
  cc_test(linalg_quaternion_test quaternion_test.cc)
  cc_test(linalg_rotation_conventions_test rotation_conventions_test.cc)
  cc_test(linalg_scale_test scale_test.cc)
  cc_test(linalg_sum_test sum_test.cc)
  cc_test(linalg_transpose_test transpose_test.cc)
else()
  message(WARNING "wont build linalg tests")
endif()



