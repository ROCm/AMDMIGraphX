#####################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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
project(MGXLlama2)
cmake_minimum_required(VERSION 3.22)

set(TARGET_NAME mgxllama2)

set(HARNESS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/harness)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CXX /opt/rocm/llvm/bin/clang++)
 
list (APPEND CMAKE_PREFIX_PATH /opt/rocm ${HIP_PATH})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3 -W -Wall -pthread -D__HIP_PLATFORM_HCC__=1")

find_package(migraphx REQUIRED)
find_package(hip REQUIRED)

include_directories(${HARNESS_DIR})

add_executable(${TARGET_NAME}
    mgxllama2.cc
)

target_include_directories(${TARGET_NAME}
    PUBLIC ${HARNESS_DIR}
)

target_link_libraries(${TARGET_NAME}
    migraphx::c
    hip::device
    pthread
)

