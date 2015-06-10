cc_require(eigen3 GTKmm TinyXML2 yamlcpp)
if(required_FOUND)
  cc_include(eigen3 GTKmm TinyXML2 yamlcpp)
  
  cc_library(ck_gtk
             HEADERS eigen_cairo.h
                     eigen_cairo_impl.h
                     layout_map.h
                     pan_zoom_view.h
                     simple_view.h
             SOURCES eigen_cairo.cc
                     layout_map.cc
                     pan_zoom_view.cc
                     simple_view.cc
             PKG_DEPENDS eigen3 GTKmm TinyXML2 yamlcpp) 
else()
  message(WARNING "Can't build ck_gtk, missing ${required_MISSING}")
endif()
