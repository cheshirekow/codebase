
cc_require(glog)
if(required_FOUND)
  cc_library(tap SOURCES
             actions.h
             actions.cc
             hash.h
             kwargs.h
             tap.h
             tap.cc
             PKG_DEPENDS glog
             TARGET_DEPENDS format)

  cc_executable(tap_demo SOURCES tap_demo.cc
                TARGET_DEPENDS tap_static)
  cc_test(tap_kwarg_test SOURCES kwarg_test.cc)
else()
  mark_excluded("tap: missing glog")
endif()
