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

include("${CMAKE_SOURCE_DIR}/cmake/Modules/FindAll.cmake")

find_all(cheshirekow
         PKG cairomm-xlib-1.0 PREFIX cairomm
         PKG eigen3
         PKG fontconfig
         PKG freetype2
         PKG fuse
         PKG glew
         PKG gtkmm-3.0 PREFIX GTKmm
         PKG libcrypto++ 
         PKG libglfw PREFIX glfw
         PKG libglog PREFIX glog
         PKG libgflags PREFIX gflags
         PKG pangomm-1.4 PREFIX pangomm
         PKG protobuf
         PKG re2
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
         PACKAGE ProtobufCompiler
         PACKAGE Soci COMPONENTS sqlite3
         PACKAGE Threads)

set(CUDA_PROPAGATE_HOST_FLAGS OFF)

enable_testing()

include_directories(include BEFORE SYSTEM)
include_directories(src BEFORE SYSTEM)
include_directories(BEFORE SYSTEM third_party/gtest)

add_subdirectory(third_party)

set(GTEST_LIBS
  gtest
  gtest_main
  pthread)

macro(cc_test TEST_NAME)
  add_executable(${TEST_NAME} ${ARGN})
  target_link_libraries(${TEST_NAME} ${GTEST_LIBS})
  add_test(${TEST_NAME} ${TEST_NAME})
endmacro()

# Configuration options
set(FIBER_USE_VARIADIC_TEMPLATES TRUE CACHE BOOL
    "If true, fiber will use variadic templates for additional functionality")
if(FIBER_USE_VARIADIC_TEMPLATES)
  add_definitions(-DFIBER_USE_VARIADIC_TEMPLATES)
endif()

add_subdirectory(src)

# configure the doxygen configuration
configure_file("${PROJECT_SOURCE_DIR}/doxy.config.in"
               "${PROJECT_BINARY_DIR}/doxy.config")

add_custom_target(doc
                  COMMAND doxygen ${PROJECT_BINARY_DIR}/doxy.config
                  WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
