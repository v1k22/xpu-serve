cmake_minimum_required(VERSION 3.22)
project(xpu-serve VERSION 0.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(XS_WITH_OPENVINO "Build the OpenVINO backend" ON)
option(XS_BUILD_TESTS "Build unit tests" ON)
option(XS_WERROR "Treat warnings as errors" OFF)

find_package(Threads REQUIRED)

if(XS_WITH_OPENVINO)
  # Point OpenVINO_DIR at <openvino>/runtime/cmake, or source <openvino>/setupvars.sh first.
  find_package(OpenVINO COMPONENTS Runtime)
  if(NOT OpenVINO_FOUND)
    message(WARNING "OpenVINO not found: building without the OpenVINO backend")
    set(XS_WITH_OPENVINO OFF)
  endif()
endif()

# Warnings for our own code only (third_party is compiled through its own targets).
add_library(xs_warnings INTERFACE)
target_compile_options(xs_warnings INTERFACE -Wall -Wextra -Wpedantic -Wshadow -Wnon-virtual-dtor
                                             -Wno-missing-field-initializers)
if(XS_WERROR)
  target_compile_options(xs_warnings INTERFACE -Werror)
endif()

add_subdirectory(third_party)
add_subdirectory(src)

if(XS_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

message(STATUS "xpu-serve ${PROJECT_VERSION}: OpenVINO backend = ${XS_WITH_OPENVINO}, tests = ${XS_BUILD_TESTS}")
