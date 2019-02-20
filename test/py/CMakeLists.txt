
find_package(PythonInterp)

function(add_py_test NAME SCRIPT)
    set (ENV_COMMAND ${CMAKE_COMMAND} -E env 
        "PYTHONPATH=$<TARGET_FILE_DIR:migraphx_py>"
        "PYTHONMALLOC=debug"
        "MALLOC_CHECK_=3"
    )
    add_test(
        NAME test_py_${NAME} 
        COMMAND ${ENV_COMMAND} ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${SCRIPT} ${ARGN})
    add_custom_target(test_py_${NAME} 
        COMMAND ${ENV_COMMAND} ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${SCRIPT} ${ARGN} 
        COMMENT "${PYTHON_EXECUTABLE} ${SCRIPT}")
endfunction()
add_dependencies(tests migraphx_py)
add_dependencies(check migraphx_py)

add_py_test(cpu test_cpu.py WORKING_DIRECTORY ${TEST_ONNX_DIR})
if(MIGRAPHX_ENABLE_GPU)
add_py_test(gpu test_gpu.py WORKING_DIRECTORY ${TEST_ONNX_DIR})
add_py_test(array test_array.py WORKING_DIRECTORY ${TEST_ONNX_DIR})
endif()
