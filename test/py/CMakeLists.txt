
find_package(PythonInterp)

function(add_py_test NAME SCRIPT)
    set (ENV_COMMAND ${CMAKE_COMMAND} -E env 
        "PYTHONPATH=$<TARGET_FILE_DIR:migraphx_py>"
    )
    add_test(
        NAME test_py_${NAME} 
        COMMAND ${ENV_COMMAND} ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${SCRIPT} ${ARGN})
    add_custom_target(test_py_${NAME} 
        COMMAND ${ENV_COMMAND} ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${SCRIPT} ${ARGN} 
        COMMENT "${PYTHON_EXECUTABLE} ${SCRIPT}")
endfunction()

add_py_test(test test.py WORKING_DIRECTORY ${TEST_ONNX_DIR})
