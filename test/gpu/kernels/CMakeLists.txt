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

file(GLOB KERNELS_TESTS CONFIGURE_DEPENDS *.cpp)
list(REMOVE_ITEM KERNELS_TESTS ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)

message("KERNELS_TESTS: ${KERNELS_TESTS}")

add_embed_library(kernel_tests ${KERNELS_TESTS} RELATIVE ${CMAKE_CURRENT_SOURCE_DIR})
rocm_add_test_executable(test_gpu_kernels main.cpp)

target_link_libraries(test_gpu_kernels migraphx migraphx_gpu kernel_tests)



# foreach(TEST ${KERNELS_TESTS})
#     get_filename_component(BASE_NAME ${TEST} NAME_WE)
#     rocm_add_test_executable(test_gpu_kernels_${BASE_NAME} main.cpp)
#     rocm_clang_tidy_check(test_gpu_kernels_${BASE_NAME})
# endforeach()





# function(add_gpu_kernels_test ${TEST_NAME})
#     add_executable(${TEST_NAME} ${ARGN})
#     rocm_clang_tidy_check(${TEST_NAME})
#     target_link_libraries(${TEST_NAME} migraphx migraphx_all_targets)
#     target_include_directories(${TEST_NAME} PUBLIC ../include)
# endfunction(add_kernels_test)


# add_executable(test_kernels ${VERIFY_TESTS})
# rocm_mark_as_test(test_kernels)
# rocm_install_test(TARGETS test_kernels)
# target_link_libraries(test_kernels migraphx migraphx_all_targets)
# target_include_directories(test_kernels PUBLIC ../include)
# rocm_clang_tidy_check(test_kernels)

# foreach(SECTION general reduce rnn conv gemm)
#     rocm_add_test(NAME test_kernels_${SECTION} COMMAND test_kernels ${SECTION})
#     set_tests_properties(test_kernels_${SECTION} PROPERTIES 
#         COST 100 
#     )
#     if(MIGRAPHX_ENABLE_GPU)
#         set_tests_properties(test_kernels_${SECTION} PROPERTIES 
#             RESOURCE_LOCK gpu
#         )
#     endif()
# endforeach()
