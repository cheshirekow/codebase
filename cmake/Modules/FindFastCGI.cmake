# - Try to find FastCGI
# Once done, this will define
#
#  FastCGI_FOUND        - system has FastCGI
#  FastCGI_INCLUDE_DIRS - the FastCGI include directories
#  FastCGI_LIBRARIES    - link these to use FastCGI

# Main include dir
set(FastCGI_FOUND False)
set(FastCGI_INCLUDE_DIRS False)
set(FastCGI_LIBRARIES False)

find_path(FastCGI_STDIO_INCLUDE_DIR NAMES fcgi_stdio.h)
find_path(FastCGI_APP_INCLUDE_DIR NAMES fcgiapp.h)

find_library(FastCGI_LIBRARY fcgi)
find_library(FastCGI_CPP_LIBRARY fcgi++)

if((FastCGI_STDIO_INCLUDE_DIR)
   AND (FastCGI_APP_INCLUDE_DIR)
   AND (FastCGI_LIBRARY)
   AND (FastCGI_CPP_LIBRARY))
  set(FastCGI_FOUND TRUE)
  set(FastCGI_INCLUDE_DIRS ${FastCGI_STDIO_INCLUDE_DIR}
                           ${FastCGI_APP_INCLUDE_DIR})
  set(FastCGI_LIBRARIES    ${FastCGI_LIBRARY}
                           ${FastCGI_CPP_LIBRARY})
endif()

