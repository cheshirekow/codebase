if((freetype2_FOUND) AND (sigc++_FOUND))
  
  include_directories(${freetype2_INCLUDE_DIRS}
                      ${sigc++_INCLUDE_DIRS})

  add_executable(freetype_tutorial main.cpp)
  target_link_libraries(freetype_tutorial
                        cpp_freetype_static
                        ${freetype2_LIBRARIES} 
                        ${freetype2_LIBRARIES})
else()
  message(WARNING "Can't build freetype2_tutorial") 
endif()