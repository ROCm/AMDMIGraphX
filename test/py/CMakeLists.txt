#####################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
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
set(VENV ${CMAKE_BINARY_DIR}/test/py/venv)
set(BASE_REQUIREMENTS ${CMAKE_CURRENT_SOURCE_DIR}/base-requirements.txt)
set(REQUIREMENTS_PYTHON_ALL ${CMAKE_CURRENT_SOURCE_DIR}/requirements-python-all.txt)
set(REQUIREMENTS_PYTHON_36 ${CMAKE_CURRENT_SOURCE_DIR}/requirements-python-3.6.txt)

function(add_py_venv_fixture FIXTURE_NAME PYTHON_EXECUTABLE VIRTUAL_ENV_DIR REQUIREMENTS_FILE)
    if(NOT TEST py_${FIXTURE_NAME}_initialize_env)
        add_test(NAME py_${FIXTURE_NAME}_initialize_env COMMAND ${PYTHON_EXECUTABLE} -m venv ${VIRTUAL_ENV_DIR} --clear)
        set_tests_properties(py_${FIXTURE_NAME}_initialize_env PROPERTIES FIXTURES_SETUP ${FIXTURE_NAME}_INIT_VENV)
        set(ENV{VIRTUAL_ENV} ${VIRTUAL_ENV_DIR})
        set(PYTHON_EXECUTABLE ${VIRTUAL_ENV_DIR}/bin/python)
        add_test(
            NAME py_${FIXTURE_NAME}_setup_env
            COMMAND ${PYTHON_EXECUTABLE} -m pip install -r ${BASE_REQUIREMENTS} -r ${REQUIREMENTS_FILE})
        set_tests_properties(py_${FIXTURE_NAME}_setup_env PROPERTIES FIXTURES_REQUIRED ${FIXTURE_NAME}_INIT_VENV)
        set_tests_properties(py_${FIXTURE_NAME}_setup_env PROPERTIES FIXTURES_SETUP ${FIXTURE_NAME}_VENV)
    endif()
endfunction()

function(add_py_test NAME SCRIPT VENV_DIR)
    foreach(PYTHON_VERSION ${PYTHON_VERSIONS})
        set (ENV_COMMAND ${CMAKE_COMMAND} -E env 
            "PYTHONPATH=$<TARGET_FILE_DIR:migraphx_pybind_${PYTHON_VERSION}>"
            "PYTHONMALLOC=debug"
            "MALLOC_CHECK_=3"
        )
        set(PYTHON_EXECUTABLE ${PYTHON_${PYTHON_VERSION}_EXECUTABLE})
        if(${PYTHON_VERSION} STREQUAL "3.6")
            add_py_venv_fixture(${PYTHON_VERSION} ${PYTHON_EXECUTABLE} ${VENV_DIR}/${PYTHON_VERSION} ${REQUIREMENTS_PYTHON_36})
        else()
            add_py_venv_fixture(${PYTHON_VERSION} ${PYTHON_EXECUTABLE} ${VENV_DIR}/${PYTHON_VERSION} ${REQUIREMENTS_PYTHON_ALL})
        endif()
        unset(PYTHON_EXECUTABLE)
        set(PYTHON_EXECUTABLE ${VENV_DIR}/${PYTHON_VERSION}/bin/python)
        add_test(
            NAME test_py_${PYTHON_VERSION}_${NAME} 
            COMMAND ${ENV_COMMAND} ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${SCRIPT} ${ARGN})
        set_tests_properties(test_py_${PYTHON_VERSION}_${NAME} PROPERTIES FIXTURES_REQUIRED ${PYTHON_VERSION}_VENV)
        add_custom_target(test_py_${PYTHON_VERSION}_${NAME} 
            COMMAND ${ENV_COMMAND} ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${SCRIPT} ${ARGN} 
            COMMENT "${PYTHON_EXECUTABLE} ${SCRIPT}")
    endforeach()
endfunction()
add_dependencies(tests migraphx_py)
add_dependencies(check migraphx_py)

add_py_test(ref test_cpu.py ${VENV} WORKING_DIRECTORY ${TEST_ONNX_DIR})
add_py_test(save_load test_save_load.py ${VENV} WORKING_DIRECTORY ${TEST_ONNX_DIR})
add_py_test(op test_op.py ${VENV} WORKING_DIRECTORY ${TEST_ONNX_DIR})
add_py_test(shape test_shape.py ${VENV} WORKING_DIRECTORY ${TEST_ONNX_DIR})
add_py_test(module_construct test_module_construct.py ${VENV} WORKING_DIRECTORY ${TEST_ONNX_DIR})
add_py_test(literal test_literal.py ${VENV} WORKING_DIRECTORY ${TEST_ONNX_DIR})
if(MIGRAPHX_ENABLE_GPU)
add_py_test(gpu_offload test_gpu_offload.py ${VENV} WORKING_DIRECTORY ${TEST_ONNX_DIR})
add_py_test(gpu test_gpu.py ${VENV} WORKING_DIRECTORY ${TEST_ONNX_DIR})
add_py_test(array test_array.py ${VENV} WORKING_DIRECTORY ${TEST_ONNX_DIR})
add_py_test(backend onnx_backend_test.py ${VENV} WORKING_DIRECTORY ${TEST_ONNX_DIR})
add_py_test(gpu_async test_gpu_async.py ${VENV} WORKING_DIRECTORY ${TEST_ONNX_DIR})
endif()
