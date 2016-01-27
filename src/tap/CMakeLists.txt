
cc_require(glog)
if(required_FOUND)
  cc_library(tap SOURCES tap.h tap.cc
             PKG_DEPENDS glog
             TARGET_DEPENDS format)

  cc_executable(tap_demo SOURCES tap_demo.cc
                TARGET_DEPENDS tap_static)
else()
  mark_excluded("tap: missing glog")
endif()
