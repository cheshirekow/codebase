set(LIBRARY_HEADERS Buffer.h
                    FrameBuffer.h
                    GLuintRef.h
                    gltk.h
                    Program.h
                    RefCounted.h
                    RefPtr.h
                    Shader.h
                    Texture.h
                    VertexArray.h)

set(LIBRARY_SOURCES Buffer.cpp
                    FrameBuffer.cpp
                    Program.cpp
                    RefCounted.cpp
                    Shader.cpp
                    Texture.cpp
                    VertexArray.cpp)

add_library(gltk_static STATIC ${LIBRARY_SOURCES})
add_library(gltk_shared SHARED ${LIBRARY_SOURCES})

set_target_properties(gltk_static gltk_shared
                      PROPERTIES OUTPUT_NAME gltk)

install(FILES ${LIBRARY_HEADERS}
        DESTINATION include/gltk)

install(TARGETS gltk_static
                gltk_shared
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib )

if((OPENGL_FOUND)
    AND (glew_FOUND)
    AND (glfw_FOUND))

  include_directories(${OPENGL_INCLUDE_DIR} 
                     ${glew_INCLUDE_DIR}
                     ${glfw_INCLUDE_DIR} )

  set(LIBS ${LIBS}
      ${OPENGL_LIBRARY}
      ${glew_LIBRARY}
      ${glfw_LIBRARY}
      gltk_shared )

    add_subdirectory(test01)
else()
  message(WARNING "Wont build gltk test01")
endif()
