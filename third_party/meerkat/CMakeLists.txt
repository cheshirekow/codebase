# Note(bialkowski): This listfile was not distributed with meerkat

set(LIBRARY_HEADERS meerkat/meerkat.h)
set(LIBRARY_SOURCES meerkat/meerkat.c)

add_library(meerkat_static STATIC ${LIBRARY_SOURCES})
add_library(meerkat_shared SHARED ${LIBRARY_SOURCES})

set_target_properties(meerkat_static meerkat_shared
                      PROPERTIES OUTPUT_NAME meerkat)

install(FILES ${LIBRARY_HEADERS}
        DESTINATION include/meerkat)

install(TARGETS meerkat_static
                meerkat_shared
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib )

