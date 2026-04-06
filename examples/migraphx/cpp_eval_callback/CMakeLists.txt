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
cmake_minimum_required(VERSION 3.5)
project(cpp_eval_callback)

set(CMAKE_CXX_STANDARD 17)

# Root of the MIGraphX source and build trees
set(MIGRAPHX_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../.." CACHE PATH "MIGraphX source root")
set(MIGRAPHX_BUILD_DIR "${MIGRAPHX_SRC_DIR}/build" CACHE PATH "MIGraphX build directory")

add_executable(eval_callback_example eval_callback_example.cpp)

target_include_directories(eval_callback_example PRIVATE
    ${MIGRAPHX_SRC_DIR}/src/include
    ${MIGRAPHX_BUILD_DIR}/src/include
)

target_link_directories(eval_callback_example PRIVATE
    ${MIGRAPHX_BUILD_DIR}/lib
)

target_link_libraries(eval_callback_example
    migraphx
    migraphx_gpu
)
