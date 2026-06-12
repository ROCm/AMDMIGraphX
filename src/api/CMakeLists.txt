#####################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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

# The C API (migraphx.h and api.cpp) is generated from the templates in
# tools/api. `make generate` still writes them into the source tree so the
# output can be reviewed, but migraphx_c is built from a freshly generated copy
# placed in the build directory instead of the checked-in src/api copies.
find_package(Python 3 COMPONENTS Interpreter REQUIRED)
find_program(CLANG_FORMAT clang-format PATHS /opt/rocm/llvm ENV HIP_PATH PATH_SUFFIXES bin)

set(MIGRAPHX_API_TOOLS_DIR ${PROJECT_SOURCE_DIR}/tools)
set(MIGRAPHX_API_GEN_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/include)
set(MIGRAPHX_API_GEN_HEADER ${MIGRAPHX_API_GEN_INCLUDE_DIR}/migraphx/migraphx.h)
set(MIGRAPHX_API_GEN_SOURCE ${CMAKE_CURRENT_BINARY_DIR}/api.cpp)

# clang-format only affects readability of the generated files, so it is
# optional for the build; pass it along only when it is available.
set(MIGRAPHX_API_FORMAT_ARG)
if(CLANG_FORMAT)
    set(MIGRAPHX_API_FORMAT_ARG -f ${CLANG_FORMAT})
endif()

add_custom_command(
    OUTPUT ${MIGRAPHX_API_GEN_HEADER} ${MIGRAPHX_API_GEN_SOURCE}
    COMMAND ${Python_EXECUTABLE} ${MIGRAPHX_API_TOOLS_DIR}/generate.py
        --api-only
        --api-header ${MIGRAPHX_API_GEN_HEADER}
        --api-source ${MIGRAPHX_API_GEN_SOURCE}
        ${MIGRAPHX_API_FORMAT_ARG}
    WORKING_DIRECTORY ${MIGRAPHX_API_TOOLS_DIR}
    DEPENDS
        ${MIGRAPHX_API_TOOLS_DIR}/generate.py
        ${MIGRAPHX_API_TOOLS_DIR}/api.py
        ${MIGRAPHX_API_TOOLS_DIR}/api/migraphx.h
        ${MIGRAPHX_API_TOOLS_DIR}/api/api.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/migraphx.py
    COMMENT "Generating C API (migraphx.h and api.cpp) in build directory"
    VERBATIM
)

add_custom_target(generate_api DEPENDS ${MIGRAPHX_API_GEN_HEADER} ${MIGRAPHX_API_GEN_SOURCE})

# The non-generated public headers (e.g. migraphx.hpp) live alongside the
# generated header so consumers get a complete include directory.
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/include/migraphx/migraphx.hpp
    ${MIGRAPHX_API_GEN_INCLUDE_DIR}/migraphx/migraphx.hpp
    COPYONLY)

add_library(migraphx_c
    ${MIGRAPHX_API_GEN_SOURCE}
)
add_dependencies(migraphx_c generate_api)
set_target_properties(migraphx_c PROPERTIES EXPORT_NAME c)
migraphx_generate_export_header(migraphx_c DIRECTORY migraphx/api)

# migraphx_c is stable API interface library. SO version of this should be 
# bumped when binary compatibility is broken. 
rocm_set_soversion(migraphx_c 3.0)   

if(BUILD_TESTING)
    target_compile_definitions(migraphx_c PRIVATE MIGRAPHX_BUILD_TESTING)
endif()

rocm_clang_tidy_check(migraphx_c)
target_link_libraries(migraphx_c PRIVATE migraphx migraphx_tf migraphx_onnx)
target_link_libraries(migraphx_c PUBLIC migraphx_version)

rocm_install_targets(
  TARGETS migraphx_c
  INCLUDE
    ${MIGRAPHX_API_GEN_INCLUDE_DIR}
)
