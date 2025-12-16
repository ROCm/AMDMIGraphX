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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#####################################################################################

cmake_minimum_required(VERSION 3.15)
project(paged_attention_example)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add default ROCm install path hint
list(APPEND CMAKE_PREFIX_PATH "/opt/rocm")

# Find MIGraphX and HIP
find_package(migraphx REQUIRED)
find_package(hip REQUIRED)

# Create the decode executable
add_executable(paged_attention_decode paged_attention_decode.cpp)

# Set HIP platform define
target_compile_options(paged_attention_decode PRIVATE -D__HIP_PLATFORM_AMD__=1)

# Link MIGraphX and HIP runtime
target_link_libraries(paged_attention_decode PRIVATE migraphx::c hip::host)

# Create the prefill executable (combined KV format)
add_executable(paged_attention_prefill paged_attention_prefill.cpp)

# Set HIP platform define
target_compile_options(paged_attention_prefill PRIVATE -D__HIP_PLATFORM_AMD__=1)

# Link MIGraphX and HIP runtime
target_link_libraries(paged_attention_prefill PRIVATE migraphx::c hip::host)

# Create the verification executable
add_executable(verify_paged_attention verify_paged_attention.cpp)

# Set HIP platform define
target_compile_options(verify_paged_attention PRIVATE -D__HIP_PLATFORM_AMD__=1)

# Link MIGraphX and HIP runtime
target_link_libraries(verify_paged_attention PRIVATE migraphx::c hip::host)

# Copy test ONNX file if available
if(EXISTS "${CMAKE_SOURCE_DIR}/../../test/onnx/group_query_attention_defaults_test.onnx")
    configure_file(
        "${CMAKE_SOURCE_DIR}/../../test/onnx/group_query_attention_defaults_test.onnx"
        "${CMAKE_BINARY_DIR}/group_query_attention_defaults_test.onnx"
        COPYONLY
    )
endif()

