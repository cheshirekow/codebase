
cc_require(glog)
if(required_FOUND)
  cc_library(tap SOURCES
             actions.h
             actions.cc
             kwargs.h
             tap.h
             tap.cc
             tap_common.h
             value_parsers.h
             value_parsers.cc
             PKG_DEPENDS glog
             TARGET_DEPENDS format)

  cc_executable(tap_demo SOURCES tap_demo.cc
                TARGET_DEPENDS tap_static)

  cc_test(tap_value_parser_test SOURCES value_parser_test.cc
          TARGET_DEPENDS tap_static)
  cc_test(tap_test SOURCES tap_test.cc
          TARGET_DEPENDS tap_static)
else()
  mark_excluded("tap: missing glog")
endif()
