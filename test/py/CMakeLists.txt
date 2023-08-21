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

include(PythonModules)

function(add_py_test NAME SCRIPT)
    foreach(PYTHON_VERSION ${PYTHON_VERSIONS})
        set (ENV_COMMAND ${CMAKE_COMMAND} -E env 
            "PYTHONPATH=$<TARGET_FILE_DIR:migraphx_pybind_${PYTHON_VERSION}>"
            "PYTHONMALLOC=debug"
            "MALLOC_CHECK_=3"
        )
        set(PYTHON_EXECUTABLE ${PYTHON_${PYTHON_VERSION}_EXECUTABLE})
        add_test(
            NAME test_py_${PYTHON_VERSION}_${NAME} 
            COMMAND ${ENV_COMMAND} ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${SCRIPT} ${ARGN})
        add_custom_target(test_py_${PYTHON_VERSION}_${NAME} 
            COMMAND ${ENV_COMMAND} ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${SCRIPT} ${ARGN} 
            COMMENT "${PYTHON_EXECUTABLE} ${SCRIPT}")

    endforeach()
endfunction()
add_dependencies(tests migraphx_py)
add_dependencies(check migraphx_py)

add_py_test(ref test_cpu.py WORKING_DIRECTORY ${TEST_ONNX_DIR})
add_py_test(save_load test_save_load.py WORKING_DIRECTORY ${TEST_ONNX_DIR})
add_py_test(op test_op.py WORKING_DIRECTORY ${TEST_ONNX_DIR})
add_py_test(shape test_shape.py WORKING_DIRECTORY ${TEST_ONNX_DIR})
add_py_test(module_construct test_module_construct.py WORKING_DIRECTORY ${TEST_ONNX_DIR})
add_py_test(literal test_literal.py WORKING_DIRECTORY ${TEST_ONNX_DIR})
if(MIGRAPHX_ENABLE_GPU)
add_py_test(gpu_offload test_gpu_offload.py WORKING_DIRECTORY ${TEST_ONNX_DIR})
add_py_test(gpu test_gpu.py WORKING_DIRECTORY ${TEST_ONNX_DIR})
add_py_test(array test_array.py WORKING_DIRECTORY ${TEST_ONNX_DIR})
add_py_test(backend onnx_backend_test.py WORKING_DIRECTORY ${TEST_ONNX_DIR})
add_py_test(gpu_async test_gpu_async.py WORKING_DIRECTORY ${TEST_ONNX_DIR})
endif()
