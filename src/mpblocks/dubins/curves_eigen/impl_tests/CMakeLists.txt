macro(dubins_test test_name)
  cc_test(dubins_solver_${test_name} SOURCES solver_${test_name}.cc)
endmacro()

if(eigen3_FOUND)
  cc_include(eigen3)
  dubins_test(LRLa_test)
  dubins_test(LRLb_test)
  dubins_test(RLRa_test)
  dubins_test(RLRb_test)
  dubins_test(LSL_test)
  dubins_test(LSR_test)
  dubins_test(RSR_test)
  dubins_test(RSL_test)
else()
  message(WARNING "wont build dubins solver tests")
endif()


