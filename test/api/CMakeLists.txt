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
function(add_api_test TEST_NAME TEST_SRC TEST_DIR)
    set(NAME test_api_${TEST_NAME})
    add_executable(${NAME} EXCLUDE_FROM_ALL ${TEST_SRC})
    rocm_clang_tidy_check(${NAME})
    target_link_libraries(${NAME} migraphx_c migraphx migraphx_all_targets)
    if(MIGRAPHX_ENABLE_ONNX)
        target_link_libraries(${NAME} onnx_files)
    endif()
    if(MIGRAPHX_ENABLE_TENSORFLOW)
        target_link_libraries(${NAME} pb_files)
    endif()
    target_include_directories(${NAME} PUBLIC ../include include ${PROJECT_BINARY_DIR}/src/include)
    add_test(NAME ${NAME} COMMAND $<TARGET_FILE:${NAME}> WORKING_DIRECTORY ${TEST_DIR}) 
    add_dependencies(tests ${NAME})
    add_dependencies(check ${NAME})
    if(WIN32)
        target_compile_definitions(${NAME} PRIVATE _CRT_SECURE_NO_WARNINGS)
    endif()
endfunction()

set(MIGRAPHX_ENABLE_C_API_TEST On CACHE BOOL "")

# Workaround: C file dont work with clang-tidy right now, need a fix in rocm-cmake
function(add_c_api_test TEST_NAME TEST_SRC TEST_DIR)
    if(MIGRAPHX_ENABLE_C_API_TEST)
        set(NAME test_api_${TEST_NAME})
        add_executable(${NAME} EXCLUDE_FROM_ALL ${TEST_SRC})
        target_link_libraries(${NAME} migraphx_c)
        target_include_directories(${NAME} PUBLIC ../include ${PROJECT_BINARY_DIR}/src/include)
        add_test(NAME ${NAME} COMMAND $<TARGET_FILE:${NAME}> WORKING_DIRECTORY ${TEST_DIR}) 
        add_dependencies(tests ${NAME})
        add_dependencies(check ${NAME})
        if(WIN32)
            target_compile_definitions(${NAME} PRIVATE _CRT_SECURE_NO_WARNINGS)
        endif()
    endif()
endfunction()

if(MIGRAPHX_ENABLE_ONNX)
add_api_test(array_base test_array_base.cpp ${TEST_ONNX_DIR})
add_api_test(assign test_assign.cpp ${TEST_ONNX_DIR})
add_api_test(compile_options test_compile_options.cpp ${TEST_ONNX_DIR})
add_api_test(lookup test_lookup.cpp ${TEST_ONNX_DIR})
add_api_test(module_construct test_module_construct.cpp ${TEST_ONNX_DIR})
add_api_test(dynamic_shape test_dynamic_shape.cpp ${TEST_ONNX_DIR})
add_api_test(ref test_cpu.cpp ${TEST_ONNX_DIR})
add_api_test(save_load test_save_load.cpp ${TEST_ONNX_DIR})
add_api_test(op test_op_construct.cpp ${TEST_ONNX_DIR})
add_c_api_test(c_op test_c_op_construct.c ${TEST_ONNX_DIR})
add_api_test(custom_op test_custom_op.cpp ${TEST_ONNX_DIR})
add_api_test(onnx_op_list test_onnx_op_list.cpp ${TEST_ONNX_DIR})
endif()
if(MIGRAPHX_ENABLE_PYTHON AND MIGRAPHX_ENABLE_TENSORFLOW)
add_api_test(tf_parser test_tf_parser.cpp ${TEST_TF_DIR})
endif()
# GPU-based tests
if(MIGRAPHX_ENABLE_GPU AND MIGRAPHX_ENABLE_ONNX)
add_api_test(gpu test_gpu.cpp ${TEST_ONNX_DIR})
add_api_test(custom_op_gpu test_custom_op_gpu.cpp ${TEST_ONNX_DIR})
endif()
