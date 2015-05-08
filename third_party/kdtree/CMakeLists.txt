# Note(bialkowski): This listfile was not distributed with kdtree

set(LIBRARY_HEADERS kdtree.h)
set(LIBRARY_SOURCES kdtree.c)

add_library(kdtree_static STATIC ${LIBRARY_SOURCES})
add_library(kdtree_shared SHARED ${LIBRARY_SOURCES})

set_target_properties(kdtree_static kdtree_shared
                      PROPERTIES OUTPUT_NAME kdtree)

install(FILES ${LIBRARY_HEADERS}
        DESTINATION include)

install(TARGETS kdtree_static
                kdtree_shared
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib )


