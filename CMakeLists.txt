# suggest running with 
#   gcc 4.6 : -DCMAKE_CXX_FLAGS="-std=c++0x"
#   gcc 4.8 : -DCMAKE_CXX_FLAGS="-std=c++11"

cmake_minimum_required(VERSION 2.8)

project (cheshirekow)
set( ${CMAKE_PROJECT_NAME}_VERSION_MAJOR 0 )
set( ${CMAKE_PROJECT_NAME}_VERSION_MINOR 1 )
set( ${CMAKE_PROJECT_NAME}_VERSION_BUGFIX 0 )

set(VERSION_STRING "${${CMAKE_PROJECT_NAME}_VERSION_MAJOR}")
set(VERSION_STRING "${VERSION_STRING}.${${CMAKE_PROJECT_NAME}_VERSION_MINOR}")
set(VERSION_STRING "${VERSION_STRING}.${${CMAKE_PROJECT_NAME}_VERSION_BUGFIX}")

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_SOURCE_DIR}/cmake/Modules")

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
SET(LCMGEN_OUTPUT_PATH ${PROJECT_BINARY_DIR}/include)

include(FindAll)

find_all(cheshirekow
         PKG cairomm-xlib-1.0 PREFIX cairomm
         PKG eigen3
         PKG fontconfig
         PKG freetype2
         PKG fuse
         PKG glib-2.0 PREFIX glib
         PKG glew
         PKG gtkmm-3.0 PREFIX GTKmm
         PKG gthread-2.0 PREFIX gthread
         PKG libcrypto++ 
         # PKG libglfw PREFIX glfw
         PKG libglog PREFIX glog
         PKG libgflags PREFIX gflags
         PKG pangomm-1.4 PREFIX pangomm
         PKG protobuf
         PKG re2
         PKG sqlite3
         PKG sigc++-2.0 PREFIX sigc++
         PKG tinyxml2 PREFIX TinyXML2
         PKG tclap
         PKG x11
         PKG xcomposite
         PKG xdamage
         PKG xext # includes XShape
         PKG xfixes
         PKG xrender
         PKG xrandr
         PKG yaml-cpp PREFIX yamlcpp
         PACKAGE Boost COMPONENTS filesystem system random
         PACKAGE OpenGL
         PACKAGE CUDA
         PACKAGE FastCGI
         PACKAGE Java
         PACKAGE ProtobufCompiler
         PACKAGE Soci COMPONENTS sqlite3
         PACKAGE Threads)

include(UseJava)
include(UseLCM)
include(cc_rules)

# string(APPEND ...) not added until cmake 3.2
function(string_append outvar)
  set(outval ${${outvar}})
  foreach(arg ${ARGN})
    set(outval "${outval}${arg}")
  endforeach()
  set(${outvar} ${outval} PARENT_SCOPE)
endfunction()

# Append a message to a global string indicating that a component has been
# excluded from the build and why
function(mark_excluded)
  set(temp "${EXCLUDED_MSG}\n")
  foreach(arg "${ARGN}")
    set(temp "${temp}${arg}")
  endforeach()
  set(EXCLUDED_MSG ${temp} CACHE STRING "" FORCE)
endfunction()

# Initilalize message of excluded components
set(EXCLUDED_MSG CACHE STRING "" FORCE)

set(CUDA_PROPAGATE_HOST_FLAGS OFF)
enable_testing()

include_directories(src)
include_directories(${PROJECT_BINARY_DIR}/include)
include_directories(BEFORE SYSTEM third_party/cppformat)
include_directories(BEFORE SYSTEM third_party/gtest)
include_directories(BEFORE SYSTEM third_party/lcm)
include_directories(third_party)

add_subdirectory(third_party)

# Configuration options
set(FIBER_USE_VARIADIC_TEMPLATES TRUE CACHE BOOL
    "If true, fiber will use variadic templates for additional functionality")
if(FIBER_USE_VARIADIC_TEMPLATES)
  add_definitions(-DFIBER_USE_VARIADIC_TEMPLATES)
endif()

string_append(CMAKE_CXX_FLAGS " -std=c++11")
add_subdirectory(src)

# configure the doxygen configuration
configure_file("${PROJECT_SOURCE_DIR}/doxy.config.in"
               "${PROJECT_BINARY_DIR}/doxy.config")

add_custom_target(doc
                  COMMAND doxygen ${PROJECT_BINARY_DIR}/doxy.config
                  WORKING_DIRECTORY ${PROJECT_BINARY_DIR})

message("lcmgen_ALL: ${lcmgen_ALL}")
add_custom_target(lcmtypes DEPENDS ${lcmgen_ALL})

# If anything was excluded:
if(EXCLUDED_MSG)
  message("The following components were excluded from the build:\n "
                  "${EXCLUDED_MSG}")
  set(EXCLUDED_MSG "" CACHE STRING "" FORCE)
endif()

