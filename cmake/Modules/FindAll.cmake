
find_package(PkgConfig)

# Find all packages in one go
# usage:
# find_all(<OUTPUT_PREFIX>
#          PACKAGE <NAME>
#             [<ARGUMENTS TO FIND_PACKAGE>]
#             [PREFIX <NAME>]
#          PKG <NAME>
# )
macro(find_all FINDALL_OUTPUT_PREFIX)
  set(FINDALL_STATE_NEW_PACKAGE  FALSE) # token is 'PACKAGE' or 'PKG'
  set(FINDALL_STATE_PACKAGE_NAME FALSE) # token is a package name
  set(FINDALL_STATE_PACKAGE_ARGS FALSE) # token is a package argument
  set(FINDALL_STATE_NEW_PREFIX   FALSE) # token is a 'PREFIX'
  set(FINDALL_STATE_PREFIX_ARG   FALSE) # token is a prefix override

  set(FINDALL_IS_PKG      FALSE)
  set(FINDALL_IS_CMAKE    FALSE)
  set(FINDALL_PACKAGE_ARGS  "")
  set(FINDALL_PACKAGE_LIST  "" )

  set(FINDALL_${FINDALL_OUTPUT_PREFIX}_FOUND   TRUE)
  set(FINDALL_${FINDALL_OUTPUT_PREFIX}_MISSING ""  )

  foreach(tok ${ARGN})
    string(COMPARE EQUAL ${tok} "PACKAGE" FINDALL_STATE_NEW_PACKAGE)
    string(COMPARE EQUAL ${tok} "PKG"     FINDALL_STATE_NEW_PKG)
    string(COMPARE EQUAL ${tok} "PREFIX"  FINDALL_STATE_NEW_PREFIX)

    # if this is a new package, then process the previously queued on
    if((FINDALL_STATE_NEW_PKG) OR (FINDALL_STATE_NEW_PACKAGE))
      # if there is a previous package then find it and add it's
      # name to the list
      if(FINDALL_STATE_IS_CMAKE)
        message(STATUS "find_all : ${FINDALL_PACKAGE_ARGS}")
        find_package(${FINDALL_PACKAGE_ARGS})
        list(APPEND FINDALL_PACKAGE_LIST ${FINDALL_PACKAGE_NAME})
      endif()

      # if there is a previous package then find it and add it's
      # name to the list
      if(FINDALL_STATE_IS_PKG)
        message(STATUS "find_all : ${FINDALL_PACKAGE_ARGS}")
        pkg_check_modules(${FINDALL_PREFIX_${FINDALL_PACKAGE_NAME}}
                          ${FINDALL_PACKAGE_ARGS})
        list(APPEND FINDALL_PACKAGE_LIST ${FINDALL_PACKAGE_NAME})
      endif()

      # clear out the name and args
      set(FINDALL_PACKAGE_NAME  "")
      set(FINDALL_PACKAGE_ARGS  "")

      # set package type
      set(FINDALL_STATE_IS_PKG   FALSE)
      set(FINDALL_STATE_IS_CMAKE FALSE)
      if(FINDALL_STATE_NEW_PKG)
        set(FINDALL_STATE_IS_PKG TRUE)
      else()
        set(FINDALL_STATE_IS_CMAKE TRUE)
      endif()

      # advance the state
      set(FINDALL_STATE_NEW_PACKAGE  FALSE)
      set(FINDALL_STATE_PACKAGE_NAME TRUE)

    # if the next token is the package name
    elseif(FINDALL_STATE_PACKAGE_NAME)
      set(FINDALL_PACKAGE_NAME "${tok}")
      list(APPEND FINDALL_PACKAGE_ARGS ${tok})

      # advance the state
      set(FINDALL_STATE_PACKAGE_NAME FALSE)
      set(FINDALL_STATE_PACKAGE_ARGS TRUE)
      set(FINDALL_PREFIX_${FINDALL_PACKAGE_NAME} ${FINDALL_PACKAGE_NAME})

    # if there's a prefix override
    elseif(FINDALL_STATE_NEW_PREFIX)
      set(FINDALL_STATE_NEW_PREFIX FALSE)
      set(FINDALL_STATE_PREFIX_ARG TRUE)

    # if the next token is the prefix override
    elseif(FINDALL_STATE_PREFIX_ARG)
      set(FINDALL_PREFIX_${FINDALL_PACKAGE_NAME} ${tok})

      # advance the state
      set(FINDALL_STATE_PREFIX_ARG FALSE)

    # if the token is a package argument
    elseif(FINDALL_STATE_PACKAGE_ARGS)
      list(APPEND FINDALL_PACKAGE_ARGS ${tok})
    endif()
  endforeach()

  # handle the last package
  if(FINDALL_STATE_IS_CMAKE)
    message(STATUS "find_all : ${FINDALL_PACKAGE_ARGS}")
    find_package(${FINDALL_PACKAGE_ARGS})
    list(APPEND FINDALL_PACKAGE_LIST ${FINDALL_PACKAGE_NAME})
  endif()

  if(FINDALL_STATE_IS_PKG)
    message(STATUS "find_all : ${FINDALL_PACKAGE_ARGS}")
    pkg_check_modules(${FINDALL_PREFIX_${FINDALL_PACKAGE_NAME}}
                      ${FINDALL_PACKAGE_ARGS})
    list(APPEND FINDALL_PACKAGE_LIST ${FINDALL_PACKAGE_NAME})
  endif()

  # check results
  foreach(PACKAGE_NAME ${FINDALL_PACKAGE_LIST})
    if(${FINDALL_PREFIX_${PACKAGE_NAME}}_FOUND)
      message(STATUS "find_all: ${PACKAGE_NAME} -> FOUND")
    else()
      message(STATUS "find_all: ${PACKAGE_NAME} -> NOT-FOUND")
    endif()

    if(NOT (${FINDALL_PREFIX_${PACKAGE_NAME}}_FOUND))
      set(FINDALL_${FINDALL_OUTPUT_PREFIX}_FOUND FALSE)
      set(FINDALL_${FINDALL_OUTPUT_PREFIX}_MISSING 
          "${FINDALL_${FINDALL_OUTPUT_PREFIX}_MISSING} ${PACKAGE_NAME}")
    endif()
  endforeach()
endmacro()

# Verify a set of packages are found
# usage:
# check_found(<OUTPUT_PREFIX>
#             PREFIX [PREFIX2 [PREFIX3 [...]]])
# returns (into calling scope)
#    <OUTPUT_PREFIX>_FOUND  is true if <PREFIX>_FOUND is true for all <PREFIX>
#    <OUTPUT_PREFIX>_MISSING is a list of all inputs for which found was false
function(check_found output_prefix)
  set(missing_list)
  set(all_found TRUE)
  foreach(prefix ${ARGN})
    if(NOT ${prefix}_FOUND)
      list(APPEND missing_list ${prefix})
      set(all_found FALSE)
    endif()
  endforeach()

  string(REGEX REPLACE ";" " " missing_str "${missing_list}")
  set(${output_prefix}_FOUND ${all_found} PARENT_SCOPE)
  set(${output_prefix}_MISSING ${missing_str} PARENT_SCOPE)
endfunction()
