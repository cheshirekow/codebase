# checks that xxx_FOUND is true for all required prefixes, and prints a
# warning message if they are not
function(cc_require)
  set(local_MISSING)
  set(local_FOUND TRUE)
  foreach(prefix ${ARGN})
    if(NOT ${prefix}_FOUND)
      list(APPEND local_MISSING ${prefix})
      set(local_FOUND FALSE)
    endif()
  endforeach()
  set(required_FOUND ${local_FOUND} PARENT_SCOPE)
  set(required_MISSING "${local_MISSING}" PARENT_SCOPE)
endfunction()

# include pkg_INCLUDE_DIRS for each pkg in the list
macro(cc_include)
  foreach(pkg ${ARGN})
    include_directories(${${pkg}_INCLUDE_DIRS})
  endforeach()
endmacro()

# Build a library target
#   Creates both a static library and a shared library target. The target names
#   are taken from ${target_name} suffixed with _static and _shared
#   respectively.
#
# example:
#   cc_library(foo
#              HEADERS foo.h bar.h
#              SOURCES foo.cc bar.cc
#              PKG_DEPENDS gtkmm
#              CMAKE_DEPENDS boost
#              TARGET_DEPENDS baz)
function(cc_library target_name)
  set(zero_value_args )
  set(one_value_args PACKAGE)
  set(multi_value_args SOURCES HEADERS PKG_DEPENDS CMAKE_DEPENDS TARGET_DEPENDS
                       RAW_DEPENDS)
  cmake_parse_arguments(cc "${zero_value_args}" "${one_value_args}"
                        "${multi_value_args}" ${ARGN} )

  # add INCLUDE_DIRS for all external dependencies
  foreach(dep ${cc_PKG_DEPENDS} ${cc_CMAKE_DEPENDS})
    include_directories(${${dep}_INCLUDE_DIRS})
  endforeach()

  # if there are no sources specified, then create a dummy source and build
  # an empty library just to get the link dependencies
  if(NOT cc_SOURCES)
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/${target_name}.cc
         "//Dummy library file")
    set(cc_SOURCES ${CMAKE_CURRENT_BINARY_DIR}/${target_name}.cc)
  endif()

  # create library targets
  add_library(${target_name}_static STATIC ${cc_HEADERS}
                                           ${cc_SOURCES})
  add_library(${target_name}_shared SHARED ${cc_HEADERS}
                                           ${cc_SOURCES})

  # Link dependencies into the library. This isn't actually necessary to link
  # the library, but it does associate dependencies so that they are
  # transitively linked into an executable.

  # link libraries from pkg-config imports
  foreach(dep ${cc_PKG_DEPENDS})
    target_link_libraries(${target_name}_static ${${dep}_LDFLAGS})
    target_link_libraries(${target_name}_shared ${${dep}_LDFLAGS})
  endforeach()

  # link libraries from cmake imports
  foreach(dep ${cc_CMAKE_DEPENDS})
    target_link_libraries(${target_name}_static ${${dep}_LIBRARY}
                                                ${${dep}_LIBRARIES})
    target_link_libraries(${target_name}_shared ${${dep}_LIBRARY}
                                                ${${dep}_LIBRARIES})
  endforeach()

  # link libraries that are build as part of this project
  foreach(dep ${cc_TARGET_DEPENDS})
    target_link_libraries(${target_name}_static ${dep}_static)
    target_link_libraries(${target_name}_shared ${dep}_shared)
  endforeach()

  if(${cc_RAW_DEPENDS})
    target_link_libraries(${target_name}_static ${cc_RAW_DEPENDS})
    target_link_libraries(${target_name}_shared ${cc_RAW_DEPENDS})
  endif()

  # set the output file basename the same for static and shared
  set_target_properties(${target_name}_static
                        ${target_name}_shared
                        PROPERTIES OUTPUT_NAME ${target_name})

  # install headers
  install(FILES ${LIBRARY_HEADERS}
          DESTINATION include/${target_name})

  # install targets
  install(TARGETS ${target_name}_static ${target_name}_shared
          RUNTIME DESTINATION bin
          LIBRARY DESTINATION lib
          ARCHIVE DESTINATION lib)
endfunction()


# Build an executable target
#
# example:
#   cc_exeuctable(foo
#              SOURCES foo.cc bar.cc
#              PKG_DEPENDS gtkmm
#              CMAKE_DEPENDS boost
#              TARGET_DEPENDS baz)
function(cc_executable target_name)
  set(zero_value_args )
  set(one_value_args PACKAGE)
  set(multi_value_args SOURCES PKG_DEPENDS CMAKE_DEPENDS TARGET_DEPENDS
                       RAW_DEPENDS)
  cmake_parse_arguments(cc "${zero_value_args}" "${one_value_args}"
                        "${multi_value_args}" ${ARGN} )

  # add INCLUDE_DIRS for all external dependencies
  foreach(dep ${cc_PKG_DEPENDS} ${cc_CMAKE_DEPENDS})
    include_directories(${${dep}_INCLUDE_DIRS})
  endforeach()

  # create library targets
  add_executable(${target_name}_exe ${cc_SOURCES})

  # set the output file basename the same for static and shared
  set_target_properties(${target_name}_exe
                        PROPERTIES OUTPUT_NAME ${target_name})

  # link libraries from pkg-config imports
  foreach(dep ${cc_PKG_DEPENDS})
    target_link_libraries(${target_name}_exe ${${dep}_LDFLAGS})
  endforeach()

  # link libraries from cmake imports
  foreach(dep ${cc_CMAKE_DEPENDS})
    target_link_libraries(${target_name}_exe ${${dep}_LIBRARY}
                                             ${${dep}_LIBRARIES})
  endforeach()

  # link libraries that are build as part of this project
  target_link_libraries(${target_name}_exe ${cc_TARGET_DEPENDS}
                                           ${cc_RAW_DEPENDS})

  # install targets
  install(TARGETS ${target_name}_exe
          RUNTIME DESTINATION bin
          LIBRARY DESTINATION lib
          ARCHIVE DESTINATION lib)
endfunction()

# These are the libraries to link for all tests
set(GTEST_LIBS
  gtest
  gtest_main
  pthread)

function(cc_test test_name)
  cc_executable(${test_name} ${ARGN})
  target_link_libraries(${test_name}_exe ${GTEST_LIBS})
  add_test(NAME ${test_name} COMMAND ${test_name}_exe)
endfunction()

