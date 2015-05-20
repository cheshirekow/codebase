# Note(bialkowski): This listfile was not distributed with cppformat
cc_library(format
           HEADERS cppformat/format.h
           SOURCES cppformat/format.cc)

set_target_properties(format_shared format_static PROPERTIES COMPILE_FLAGS
    "-Wall -Wextra -Wshadow -pedantic")

