find_package( OpenGL )
find_package( GLEW )
find_package( GLFW )

if( (OPENGL_FOUND)
    AND (GLEW_FOUND)
    AND (GLFW_FOUND) )
    
    include_directories(
        ${OPENGL_INCLUDE_DIR} 
        ${GLEW_INCLUDE_DIR}
        ${GLFW_INCLUDE_DIR} )
    
    set( LIBS ${LIBS}
        ${OPENGL_LIBRARY}
        ${GLEW_LIBRARY}
        ${GLFW_LIBRARY}
        gltk )

    add_subdirectory(test01)
else()

set( MISSING "" )

if( NOT(OPENGL_FOUND) )
    set( MISSING "${MISSING} opengl" )
endif()

if( NOT(GLEW_FOUND) )
    set( MISSING "${MISSING} glew" )
endif()

if( NOT(GLFW_FOUND) )
    set( MISSING "${MISSING} glfw" )
endif()

message( WARNING "Missing: ${MISSING}")

endif()
