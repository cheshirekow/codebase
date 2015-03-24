if(fontconfig_FOUND)
  include_directories(${fontconfig_INCLUDE_DIRS})
    
  set(LIBRARY_HEADERS AssignmentPair.h
                      Atomic.h
                      Blanks.h
                      Cache.h
                      CharSet.h
                      common.h
                      Config.h
                      constant.h
                      cpp_fontconfig.h
                      CPtr.h
                      file.h
                      FontSet.h
                      freetype.h
                      init.h
                      LangSet.h
                      macros.h
                      Matrix.h
                      name.h
                      ObjectSet.h
                      ObjectType.h
                      Pattern.h
                      RefPtr.h
                      string.h
                      String.h
                      StrList.h
                      StrSet.h
                      TypeMap.h)

  set(LIBRARY_SOURCES Atomic.cpp
                      Blanks.cpp
                      CharSet.cpp
                      Cache.cpp
                      Config.cpp
                      Constant.cpp
                      file.cpp
                      FontSet.cpp
                      init.cpp
                      LangSet.cpp
                      Matrix.cpp
                      name.cpp
                      ObjectSet.cpp
                      ObjectType.cpp
                      Pattern.cpp
                      string.cpp
                      String.cpp
                      StrList.cpp
                      StrSet.cpp
                      TypeMap.cpp)
            
            
  if((freetype2_FOUND) AND (sigc++_FOUND))
    include_directories(${freetype2_INCLUDE_DIRS}
                        ${sigc++_INCLUDE_DIRS})
    set(LIBRARY_SOURCES ${LIBRARY_SOURCES} freetype.cpp)
    message( STATUS
            "-----\n"
            "building cpp_fontconfig with freetype support\n"
            "-----\n")
  endif()
            
  add_library(cpp_fontconfig_shared SHARED ${LIBRARY_SOURCES})
  add_library(cpp_fontconfig_static STATIC ${LIBRARY_SOURCES})
  
  install(FILES ${LIBRARY_HEADERS}
          DESTINATION include/cpp_fontconfig)
  
  install(TARGETS cpp_fontconfig_shared cpp_fontconfig_static
          RUNTIME DESTINATION bin
          LIBRARY DESTINATION lib
          ARCHIVE DESTINATION lib)
  
  add_executable(fontconfig_tutorial tutorial.cpp)
  target_link_libraries(fontconfig_tutorial 
                        cpp_fontconfig_static 
                        ${fontconfig_LIBRARIES})
else()
  message(WARNING "Not building cpp_fontconfig") 
endif()