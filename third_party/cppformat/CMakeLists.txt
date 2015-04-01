# Note(bialkowski): This listfile was not distributed with cppformat

set(LIBRARY_HEADERS format.h)
set(LIBRARY_SOURCES format.cc)

add_library(format_static STATIC ${LIBRARY_SOURCES})
add_library(format_shared SHARED ${LIBRARY_SOURCES})

set_target_properties(format_static format_shared
                      PROPERTIES OUTPUT_NAME format)

install(FILES ${LIBRARY_HEADERS}
        DESTINATION include/cppformat)

install(TARGETS format_static
                format_shared
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib )

set_target_properties(format_shared format_static PROPERTIES COMPILE_FLAGS
    "-Wall -Wextra -Wshadow -pedantic")

