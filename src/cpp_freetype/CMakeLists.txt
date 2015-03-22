if((freetype2_FOUND) AND (sigc++_FOUND))

  include_directories(${freetype2_INCLUDE_DIRS}
                      ${sigc++_INCLUDE_DIRS})

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

  INSTALL(TARGETS cpp_freetype_shared cpp_freetype_static
          RUNTIME DESTINATION bin
          LIBRARY DESTINATION lib
          ARCHIVE DESTINATION lib)
else()
    message(WARNING "Can't build cpp_freetype")  
endif()