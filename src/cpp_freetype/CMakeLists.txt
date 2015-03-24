if((freetype2_FOUND) AND (sigc++_FOUND))

  include_directories(${freetype2_INCLUDE_DIRS}
                      ${sigc++_INCLUDE_DIRS})

  set(LIBRARY_HEADERS
      AssignmentPair.h
      cpp_freetype.h
      CPtr.h
      Face.h
      GlyphSlot.h
      Library.h
      Memory.h
      Module.h
      ModuleClass.h
      OpenArgs.h
      Outline.h
      RefPtr.h
      types.h
      UnTag.h)

  set(LIBRARY_SOURCES
      cpp_freetype.cpp
      Face.cpp
      GlyphSlot.cpp
      Library.cpp
      Memory.cpp
      Module.cpp
      ModuleClass.cpp
      OpenArgs.cpp
      Outline.cpp
      Untag.cpp)

  add_library(cpp_freetype_shared SHARED ${LIBRARY_SOURCES}) 
  add_library(cpp_freetype_static STATIC ${LIBRARY_SOURCES}) 
    
  set_target_properties(cpp_freetype_shared
                        cpp_freetype_static
                        PROPERTIES OUTPUT_NAME cpp_freetype)

  install(FILES ${LIBRARY_HEADERS}
          DESTINATION include/cpp_freetype)

  install(TARGETS cpp_freetype_shared cpp_freetype_static
          RUNTIME DESTINATION bin
          LIBRARY DESTINATION lib
          ARCHIVE DESTINATION lib)

  add_executable(freetype_tutorial tutorial.cpp)
  target_link_libraries(freetype_tutorial
                        cpp_freetype_static
                        ${freetype2_LIBRARIES} 
                        ${freetype2_LIBRARIES})
else()
    message(WARNING "Can't build cpp_freetype")  
endif()