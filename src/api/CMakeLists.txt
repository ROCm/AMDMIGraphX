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

find_package(Python 3 REQUIRED COMPONENTS Interpreter)
find_program(CLANG_FORMAT clang-format PATHS /opt/rocm/llvm ENV HIP_PATH PATH_SUFFIXES bin)

set(TOOLS_DIR "${PROJECT_SOURCE_DIR}/tools")
set(BIN_DIR "${PROJECT_BINARY_DIR}/src/api")
set(SRC_DIR "${PROJECT_SOURCE_DIR}/src/api")

set(GENERATE_COMMAND ${Python_EXECUTABLE} generate.py)
if (CLANG_FORMAT)
    list(APPEND GENERATE_COMMAND -f ${CLANG_FORMAT})
endif()
if(MIGRAPHX_ENABLE_ONNX)
    list(APPEND GENERATE_COMMAND -Denable_onnx)
endif()
if(MIGRAPHX_ENABLE_TENSORFLOW)
    list(APPEND GENERATE_COMMAND -Denable_tensorflow)
endif()
list(APPEND GENERATE_COMMAND -o ${PROJECT_BINARY_DIR}/src)

add_custom_command(
    OUTPUT ${BIN_DIR}/api.cpp
    COMMAND ${GENERATE_COMMAND}
    DEPENDS ${TOOLS_DIR}/generate.py ${TOOLS_DIR}/api.py ${TOOLS_DIR}/te.py ${SRC_DIR}/migraphx.py
    WORKING_DIRECTORY ${TOOLS_DIR}
    COMMENT "Generate MIGraphX C API source files...")

add_custom_target(generate DEPENDS ${BIN_DIR}/api.cpp)

add_library(migraphx_c ${BIN_DIR}/api.cpp)
add_dependencies(migraphx_c generate)

set_target_properties(migraphx_c PROPERTIES EXPORT_NAME c)
migraphx_generate_export_header(migraphx_c DIRECTORY migraphx/api)

# migraphx_c is stable API interface library. SO version of this should be 
# bumped when binary compatibility is broken. 
rocm_set_soversion(migraphx_c 3.0)   

if(BUILD_TESTING)
    target_compile_definitions(migraphx_c PRIVATE MIGRAPHX_BUILD_TESTING)
endif()

rocm_clang_tidy_check(migraphx_c)
target_link_libraries(migraphx_c PRIVATE migraphx migraphx_all_frontends)
target_link_libraries(migraphx_c PUBLIC migraphx_version)

rocm_install_targets(
  TARGETS migraphx_c
  INCLUDE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
