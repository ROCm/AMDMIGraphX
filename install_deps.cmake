#!/usr/bin/cmake -P

# ####################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# ####################################################################################

set(ARGS)

foreach(i RANGE 3 ${CMAKE_ARGC})
    list(APPEND ARGS ${CMAKE_ARGV${i}})
endforeach()

include(CMakeParseArguments)

set(options help)
set(oneValueArgs --prefix)
set(multiValueArgs)

cmake_parse_arguments(PARSE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGS})

if(PARSE_help)
    message("Usage: install_deps.cmake [options] [cmake-args]")
    message("")
    message("Options:")
    message("  --prefix               Set the prefix to install the dependencies.")
    message("")
    message("Commands:")
    message("  help                   Show this message and exit.")
    message("")
    return()
endif()

set(_PREFIX /usr/local)

if(PARSE_--prefix)
    set(_PREFIX ${PARSE_--prefix})
endif()

get_filename_component(PREFIX ${_PREFIX} ABSOLUTE)

find_package(CMakeGet QUIET PATHS ${PREFIX})

if(NOT CMakeGet_FOUND)
    set(FILENAME ${PREFIX}/tmp/cmake-get-install.cmake)
    file(DOWNLOAD https://raw.githubusercontent.com/pfultz2/cmake-get/master/install.cmake ${FILENAME} STATUS RESULT_LIST)
    list(GET RESULT_LIST 0 RESULT)
    list(GET RESULT_LIST 1 RESULT_MESSAGE)

    if(NOT RESULT EQUAL 0)
        message(FATAL_ERROR "Download for install.cmake failed: ${RESULT_MESSAGE}")
    endif()

    execute_process(COMMAND ${CMAKE_COMMAND} -P ${FILENAME} ${PREFIX})
    file(REMOVE ${FILENAME})
    find_package(CMakeGet REQUIRED PATHS ${PREFIX})
endif()

# Set compiler to clang++ if not set
if(NOT DEFINED ENV{CXX} AND NOT DEFINED CMAKE_CXX_COMPILER AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    find_program(CLANG clang++ PATHS /opt/rocm /opt/rocm/llvm PATH_SUFFIXES bin)
    if(CLANG)
        set(ENV{CXX} ${CLANG})
    else()
        message(FATAL_ERROR "Cannot find clang++")
    endif()
endif()

cmake_get_from(${CMAKE_CURRENT_LIST_DIR}/dev-requirements.txt PREFIX ${PREFIX} CMAKE_ARGS -DCMAKE_INSTALL_RPATH=${PREFIX}/lib ${PARSE_UNPARSED_ARGUMENTS})
