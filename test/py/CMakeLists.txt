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
set(FIX_NAME SETUP_VIRTUAL_ENV)
set(VENV_DIR ${CMAKE_BINARY_DIR}/test/py/venv)

function(add_py_venv_fixture FIXTURE_NAME VIRTUAL_ENV_DIR REQUIREMENTS_FILE)
    message("Attempting to create virtual environment: ${VIRTUAL_ENV_DIR}")
    find_package (Python3 COMPONENTS Interpreter)
    add_test (NAME INITIALIZE_VENV COMMAND "${Python3_EXECUTABLE}" -m venv ${VIRTUAL_ENV_DIR})
    set_tests_properties(INITIALIZE_VENV PROPERTIES FIXTURES_SETUP INIT_VENV)

    set (ENV{VIRTUAL_ENV} ${VIRTUAL_ENV_DIR})
    set (Python3_FIND_VIRTUALENV ONLY)
    unset (Python3_EXECUTABLE)
    find_package (Python3 COMPONENTS Interpreter Development)
    add_test (NAME SETUP_VENV COMMAND "${Python3_EXECUTABLE}" -m pip install -r ${REQUIREMENTS_FILE})
    set_tests_properties(SETUP_VENV PROPERTIES FIXTURES_REQUIRED INIT_VENV)
    set_tests_properties(SETUP_VENV PROPERTIES FIXTURES_SETUP ${FIXTURE_NAME})
endfunction()

function(add_py_test NAME SCRIPT)
    foreach(PYTHON_VERSION ${PYTHON_VERSIONS})
        set (ENV_COMMAND ${CMAKE_COMMAND} -E env 
            "PYTHONPATH=$<TARGET_FILE_DIR:migraphx_pybind_${PYTHON_VERSION}>"
            "PYTHONMALLOC=debug"
            "MALLOC_CHECK_=3"
        )
        set (ENV{VIRTUAL_ENV} ${VENV_DIR})
        set (Python3_FIND_VIRTUALENV ONLY)
        unset (Python3_EXECUTABLE)
        find_package (Python3 COMPONENTS Interpreter Development)
        set(PYTHON_EXECUTABLE "${Python3_EXECUTABLE}")
        add_test(
            NAME test_py_${PYTHON_VERSION}_${NAME} 
            COMMAND ${ENV_COMMAND} ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${SCRIPT} ${ARGN})
        set_tests_properties(test_py_${PYTHON_VERSION}_${NAME} PROPERTIES FIXTURES_REQUIRED ${FIX_NAME})
        add_custom_target(test_py_${PYTHON_VERSION}_${NAME} 
            COMMAND ${ENV_COMMAND} ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${SCRIPT} ${ARGN} 
            COMMENT "${PYTHON_EXECUTABLE} ${SCRIPT}")

    endforeach()
endfunction()
add_dependencies(tests migraphx_py)
add_dependencies(check migraphx_py)

add_py_venv_fixture(
    ${FIX_NAME} ${VENV_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/REQUIREMENTS.TXT)

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
