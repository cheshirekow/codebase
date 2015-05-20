if((freetype2_FOUND) AND (sigc++_FOUND))
  cc_library(cpp_freetype
             HEADERS AssignmentPair.h
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
                     Untag.h
             SOURCES cpp_freetype.cpp
                     Face.cpp
                     GlyphSlot.cpp
                     Library.cpp
                     Memory.cpp
                     Module.cpp
                     ModuleClass.cpp
                     OpenArgs.cpp
                     Outline.cpp
                     Untag.cpp
            PKG_DEPENDS freetype2
                        sigc++)

  cc_executable(freetype_tutorial
                SOURCES tutorial.cpp
                TARGET_DEPENDS cpp_freetype_static)
else()
    message(WARNING "Can't build cpp_freetype")  
endif()
