
include(PythonModules)

function(add_py_test NAME SCRIPT)
    foreach(PYTHON_VERSION ${PYTHON_VERSIONS})
        set (ENV_COMMAND ${CMAKE_COMMAND} -E env 
            "PYTHONPATH=$<TARGET_FILE_DIR:migraphx_py_${PYTHON_VERSION}>"
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
foreach(PYTHON_VERSION ${PYTHON_VERSIONS})
    add_dependencies(tests migraphx_py_${PYTHON_VERSION})
    add_dependencies(check migraphx_py_${PYTHON_VERSION})
endforeach()

add_py_test(ref test_cpu.py WORKING_DIRECTORY ${TEST_ONNX_DIR})
add_py_test(save_load test_save_load.py WORKING_DIRECTORY ${TEST_ONNX_DIR})
add_py_test(op test_op.py WORKING_DIRECTORY ${TEST_ONNX_DIR})
add_py_test(shape test_shape.py WORKING_DIRECTORY ${TEST_ONNX_DIR})
if(MIGRAPHX_ENABLE_GPU)
add_py_test(gpu_offload test_gpu_offload.py WORKING_DIRECTORY ${TEST_ONNX_DIR})
add_py_test(gpu test_gpu.py WORKING_DIRECTORY ${TEST_ONNX_DIR})
add_py_test(array test_array.py WORKING_DIRECTORY ${TEST_ONNX_DIR})
add_py_test(backend onnx_backend_test.py WORKING_DIRECTORY ${TEST_ONNX_DIR})
endif()
