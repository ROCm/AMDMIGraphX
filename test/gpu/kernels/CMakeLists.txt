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

file(GLOB_RECURSE KERNELS_TESTS LIST_DIRECTORIES false CONFIGURE_DEPENDS *.cpp)
list(REMOVE_ITEM KERNELS_TESTS ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)

message("KERNELS_TESTS: ${KERNELS_TESTS}")

add_embed_library(kernel_tests ${KERNELS_TESTS} RELATIVE ${CMAKE_CURRENT_SOURCE_DIR})
rocm_add_test_executable(test_gpu_kernels main.cpp)

target_link_libraries(test_gpu_kernels migraphx migraphx_gpu kernel_tests)

add_library(migraphx_gpu_test_kernel_file_check EXCLUDE_FROM_ALL ${KERNELS_TESTS})
target_link_libraries(migraphx_gpu_test_kernel_file_check compile_migraphx_gpu_kernels)
rocm_clang_tidy_check(migraphx_gpu_test_kernel_file_check)

