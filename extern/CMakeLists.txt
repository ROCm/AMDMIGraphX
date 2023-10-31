#####################################################################################
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
#####################################################################################

# Windows build requires CMake version 3.24 at minimum, due to SYSTEM argument to FetchContent
cmake_minimum_required(VERSION 3.24.0 FATAL_ERROR)

# https://cmake.org/cmake/help/latest/policy/CMP0135.html
cmake_policy(SET CMP0135 NEW)

include(FetchContent)

set(CMAKE_CXX_STANDARD 17 CACHE INTERNAL "")
set(CMAKE_CXX_STANDARD_REQUIRED ON CACHE INTERNAL "")
set(CMAKE_CXX_EXTENSIONS OFF CACHE INTERNAL "")

FetchContent_Declare(
        HALF
        URL https://downloads.sourceforge.net/project/half/half/2.2.0/half-2.2.0.zip?ts=gAAAAABjwYR1XMSTpEV_SFhJCBhkdlggkBl9O2GQawgvJmGwqYTAsgEacimyfdBymL6etwelv_kYY7BdlCpy4mgcqceBbymbeQ%3D%3D&r=https%3A%2F%2Fsourceforge.net%2Fprojects%2Fhalf%2Ffiles%2Flatest%2Fdownload
        URL_MD5 b169bcfda2c65e90f73167e06368a27a
        SYSTEM)

set(JSON_BuildTests OFF CACHE INTERNAL "")

FetchContent_Declare(
        nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.11.2
        SYSTEM)

set(protobuf_BUILD_SHARED_LIBS OFF CACHE INTERNAL "")
set(protobuf_MSVC_STATIC_RUNTIME OFF CACHE INTERNAL "")
set(ABSL_BUILD_TESTING OFF CACHE INTERNAL "")
set(ABSL_USE_SYSTEM_INCLUDES ON CACHE INTERNAL "")
set(ABSL_PROPAGATE_CXX_STD ON CACHE INTERNAL "")
set(ABSL_ENABLE_INSTALL OFF CACHE INTERNAL "")
set(protobuf_BUILD_LIBPROTOC ON CACHE INTERNAL "")
set(protobuf_BUILD_TESTS OFF CACHE INTERNAL "")
set(protobuf_BUILD_PROTOC_BINARIES ON CACHE INTERNAL "")
set(protobuf_BUILD_PROTOBUF_BINARIES ON CACHE INTERNAL "")
set(protobuf_INSTALL OFF CACHE INTERNAL "")

FetchContent_Declare(
        protobuf
        GIT_REPOSITORY https://github.com/protocolbuffers/protobuf.git
        GIT_TAG v22.2
        SYSTEM)

set(MSGPACK_USE_BOOST OFF CACHE INTERNAL "")
set(MSGPACK_CXX17 ON CACHE INTERNAL "")
set(MSGPACK_BUILD_TESTS OFF CACHE INTERNAL "")
set(MSGPACK_BUILD_DOCS OFF CACHE INTERNAL "")

FetchContent_Declare(
        msgpack
        GIT_REPOSITORY https://github.com/msgpack/msgpack-c.git
        GIT_TAG cpp-5.0.0
        SYSTEM)

# Workaround: we only need Blaze header file, the part below helps to pass CMake configuration.
set(BLAS_LIBRARIES rocBLAS CACHE INTERNAL "")
set(LAPACK_LIBRARIES rocLAPACK CACHE INTERNAL "")

FetchContent_Declare(
        blaze
        GIT_REPOSITORY https://bitbucket.org/blaze-lib/blaze.git
        GIT_TAG v3.8.2
        SYSTEM)

# Store the current value of BUILD_SHARED_LIBS
set(__build_shared_libs ${BUILD_SHARED_LIBS})
set(BUILD_SHARED_LIBS OFF CACHE INTERNAL "")

FetchContent_MakeAvailable(HALF nlohmann_json protobuf msgpack blaze)

# Restore the old value of BUILD_SHARED_LIBS
set(BUILD_SHARED_LIBS ${__build_shared_libs} CACHE BOOL "" FORCE)

if(MIGRAPHX_ENABLE_CPU)
    if(NOT MIGRAPHX_ENABLE_ZENDNN)
        set(DNNL_BUILD_TESTS OFF CACHE INTERNAL "")

        if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
            # oneDNN version 1.7 workaround on clang++ error: no such file or directory: '/Od'
            set(MINGW ON CACHE INTERNAL "")
        endif()
        FetchContent_Declare(
                dnnl
                GIT_REPOSITORY https://github.com/oneapi-src/oneDNN.git
                GIT_TAG v1.7
                SYSTEM)

        FetchContent_MakeAvailable(dnnl)

        if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
            unset(MINGW)
        endif()
        set(__dnnl_compile_options
                -fdelayed-template-parsing
                -fms-extensions
                -fms-compatibility
                -Wno-unused-but-set-variable
                -Wno-unused-variable)
        target_compile_options(dnnl
            PRIVATE
                ${__dnnl_compile_options})
        target_compile_options(dnnl_common
            PRIVATE
                ${__dnnl_compile_options})
        target_compile_options(dnnl_cpu
            PRIVATE
                ${__dnnl_compile_options})
        target_compile_options(dnnl_cpu_x64
            PRIVATE
                ${__dnnl_compile_options})
        unset(__dnnl_compile_options)

        # For compatibility with find_package()
        add_library(DNNL::dnnl INTERFACE IMPORTED GLOBAL)
        target_link_libraries(DNNL::dnnl INTERFACE dnnl)

        add_library(DNNL::dnnl_common INTERFACE IMPORTED GLOBAL)
        target_link_libraries(DNNL::dnnl_common INTERFACE dnnl_common)

        add_library(DNNL::dnnl_cpu INTERFACE IMPORTED GLOBAL)
        target_link_libraries(DNNL::dnnl_cpu INTERFACE dnnl_cpu)

        add_library(DNNL::dnnl_cpu_x64 INTERFACE IMPORTED GLOBAL)
        target_link_libraries(DNNL::dnnl_cpu_x64 INTERFACE dnnl_cpu_x64)
    endif()
endif()

set(__protobuf_compile_options
        -Wno-undef
        -Wno-deprecated-declarations)

target_compile_options(libprotobuf
    PRIVATE
        ${__protobuf_compile_options})
target_compile_options(libprotobuf-lite
    PRIVATE
        ${__protobuf_compile_options})
target_compile_options(libprotoc
    PRIVATE
        ${__protobuf_compile_options})

set(HALF_INCLUDE_DIR ${CMAKE_BINARY_DIR}/_deps/half-src/include PARENT_SCOPE)
set(BLAZE_INCLUDE ${CMAKE_BINARY_DIR}/_deps/blaze-src PARENT_SCOPE)
set(protobuf_PROTOC_EXE ${protobuf_BINARY_DIR}/protoc.exe PARENT_SCOPE)

set(protobuf_DIR ${protobuf_SOURCE_DIR}/cmake PARENT_SCOPE)

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

add_library(SQLite::SQLite3 INTERFACE IMPORTED GLOBAL)
add_dependencies(SQLite::SQLite3 sqlite3)

ExternalProject_Get_Property(sqlite3 BINARY_DIR)

target_link_directories(SQLite::SQLite3 INTERFACE ${BINARY_DIR})
target_link_libraries(SQLite::SQLite3 INTERFACE libsqlite3.lib)
target_include_directories(SQLite::SQLite3 INTERFACE ${BINARY_DIR})
