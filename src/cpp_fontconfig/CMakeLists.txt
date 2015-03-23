if(fontconfig_FOUND)
  include_directories(${fontconfig_INCLUDE_DIRS})
    
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
  
  install(TARGETS cpp_fontconfig_shared cpp_fontconfig_static
          RUNTIME DESTINATION bin
          LIBRARY DESTINATION lib
          ARCHIVE DESTINATION lib)
else()
  message(WARNING "Not building cpp_fontconfig") 
endif()