if(fontconfig_FOUND)
  include_directories(${fontconfig_INCLUDE_DIRS})
  add_executable(fontconfig_tutorial tutorial.cpp)
  target_link_libraries(fontconfig_tutorial 
                        cpp_fontconfig_static 
                        ${fontconfig_LIBRARIES})
else()
  message(WARNING "Wont build cpp_fontconfig")
endif()