if(fontconfig_FOUND)
  if((freetype2_FOUND) AND (sigc++_FOUND))
    set(extra_depends  freetype2 sigc++)
    set(extra_sources  freetype.cpp)
    message( STATUS
            "-----\n"
            "building cpp_fontconfig with freetype support\n"
            "-----\n")
  endif()


  cc_library(cpp_fontconfig
              HEADERS AssignmentPair.h
                      Atomic.h
                      Blanks.h
                      Cache.h
                      CharSet.h
                      common.h
                      Config.h
                      Constant.h
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
                      TypeMap.h
              SOURCES Atomic.cpp
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
                      TypeMap.cpp
                      ${extra_sources}
              PKG_DEPENDS fontconfig
                      ${extra_depends})

  cc_executable(fontconfig_tutorial
                SOURCES tutorial.cpp
                TARGET_DEPENDS cpp_fontconfig_static)
else()
  message(WARNING "Not building cpp_fontconfig") 
endif()
