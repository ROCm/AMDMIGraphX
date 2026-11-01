#####################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#####################################################################################
#
# Toolchain file for hermetic builds: CMake never reaches the network on its
# own. FetchContent does not download or update anything (its source dirs must
# already exist), ExternalProject does not re-fetch (update) downloaded content,
# and an ExternalProject_Add() that would download fails at configure time.
#
# Usage:
#   cmake -DCMAKE_TOOLCHAIN_FILE=cmake/hermetic.cmake ...
#   rbuild prepare -d depend -t cmake/hermetic.cmake
#   tools/install_prereqs.sh --toolchain cmake/hermetic.cmake
#
# rbuild/cget include this file from the cget.cmake toolchain they write, so
# every dependency build and the MIGraphX build itself pick up these settings.
#
# Requires CMake 3.20. Not covered: a custom DOWNLOAD_COMMAND, file(DOWNLOAD),
# and nested configures (ExternalProject or FetchContent sub-builds) unless the
# project forwards CMAKE_TOOLCHAIN_FILE to them.

# Cache entries so option() in FetchContent.cmake keeps them under every CMP0077
# setting. An explicit -D on the command line still wins.
set(FETCHCONTENT_FULLY_DISCONNECTED ON CACHE BOOL
    "Hermetic build: never download or update FetchContent sources")
set(FETCHCONTENT_UPDATES_DISCONNECTED ON CACHE BOOL
    "Hermetic build: never update FetchContent sources")

# ExternalProject reads EP_UPDATE_DISCONNECTED as an inherited directory
# property, which chains up to the global scope from every directory.
set_property(GLOBAL PROPERTY EP_UPDATE_DISCONNECTED ON)

# Refuse ExternalProject downloads at configure time by wrapping
# ExternalProject_Add. ExternalProject.cmake carries include_guard(GLOBAL) since
# CMake 3.20, so including it here first makes the project's own
# include(ExternalProject) a no-op and the wrapper below survives.
if(CMAKE_VERSION VERSION_LESS 3.20)
    message(FATAL_ERROR "hermetic.cmake requires CMake 3.20 or newer, found ${CMAKE_VERSION}")
endif()

# This file is read several times per configure. Wrap only once, or the wrapper
# would call itself. A global property rather than include_guard() so the same
# file reached through two paths still counts once.
get_property(hermetic_externalproject_guarded GLOBAL PROPERTY HERMETIC_EXTERNALPROJECT_GUARDED)
if(hermetic_externalproject_guarded)
    unset(hermetic_externalproject_guarded)
    return()
endif()
unset(hermetic_externalproject_guarded)
set_property(GLOBAL PROPERTY HERMETIC_EXTERNALPROJECT_GUARDED ON)

include(ExternalProject)

# ExternalProject_Add reads CMP0097, CMP0114, CMP0135 and CMP0150 from its
# caller's policy scope, which is now this wrapper, so record it with every
# policy set to NEW instead of whatever this file was included under.
cmake_policy(PUSH)
cmake_policy(VERSION 3.20...${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}.${CMAKE_PATCH_VERSION})
function(ExternalProject_Add name)
    if(NOT "DOWNLOAD_COMMAND" IN_LIST ARGN)
        foreach(keyword URL GIT_REPOSITORY SVN_REPOSITORY HG_REPOSITORY CVS_REPOSITORY)
            if("${keyword}" IN_LIST ARGN)
                message(FATAL_ERROR
                    "Hermetic build: ExternalProject_Add(${name}) would download via ${keyword}. "
                    "Provide the sources locally with SOURCE_DIR and DOWNLOAD_COMMAND \"\", "
                    "or disable the dependency.")
            endif()
        endforeach()
    endif()
    # Forward every argument verbatim as a bracket argument; a plain ${ARGN}
    # would drop the empty strings in DOWNLOAD_COMMAND "" or INSTALL_COMMAND "".
    set(call "_ExternalProject_Add([==[${name}]==]")
    if(ARGC GREATER 1)
        math(EXPR last "${ARGC} - 1")
        foreach(i RANGE 1 ${last})
            string(APPEND call " [==[${ARGV${i}}]==]")
        endforeach()
    endif()
    cmake_language(EVAL CODE "${call})")
endfunction()
cmake_policy(POP)
