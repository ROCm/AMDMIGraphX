#####################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
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

if(NOT WIN32)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(SQLITE3 REQUIRED IMPORTED_TARGET sqlite3)
else()
    # https://cmake.org/cmake/help/latest/policy/CMP0135.html
    cmake_policy(SET CMP0135 NEW)

    if(NOT NMAKE_DIR)
        set(NMAKE_DIR "$ENV{NMAKE_DIR}")
    endif()

    find_program(NMAKE_EXECUTABLE NAMES nmake.exe REQUIRED HINTS "${NMAKE_DIR}")

    include(ExternalProject)

    ExternalProject_Add(
            sqlite3
            GIT_REPOSITORY https://github.com/sqlite/sqlite.git
            GIT_TAG version-3.40.0
            GIT_SHALLOW true
            UPDATE_DISCONNECTED true
            CONFIGURE_COMMAND ""
            BUILD_COMMAND ${NMAKE_EXECUTABLE} /f ..\\sqlite3\\Makefile.msc USE_AMALGAMATION=1 NO_TCL=1 TOP=..\\sqlite3 libsqlite3.lib
            INSTALL_COMMAND "")

    ExternalProject_Get_Property(sqlite3 BINARY_DIR)

    # For compatibility with PkgConfig
    add_library(PkgConfig::SQLITE3 INTERFACE IMPORTED GLOBAL)
    add_dependencies(PkgConfig::SQLITE3 sqlite3)
    target_link_directories(PkgConfig::SQLITE3 INTERFACE ${BINARY_DIR})
    target_link_libraries(PkgConfig::SQLITE3 INTERFACE libsqlite3.lib)
    target_include_directories(PkgConfig::SQLITE3 INTERFACE ${BINARY_DIR})
endif()
