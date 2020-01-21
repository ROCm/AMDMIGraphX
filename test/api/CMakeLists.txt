
function(add_api_test TEST_NAME TEST_SRC)
    add_executable(${TEST_NAME} EXCLUDE_FROM_ALL ${TEST_SRC})
    rocm_clang_tidy_check(${TEST_NAME})
    target_link_libraries(${TEST_NAME} migraphx_c)
    target_include_directories(${TEST_NAME} PUBLIC ../include)
    add_test(NAME ${TEST_NAME} COMMAND $<TARGET_FILE:${TEST_NAME}> WORKING_DIRECTORY ${TEST_ONNX_DIR}) 
    add_dependencies(tests ${TEST_NAME})
    add_dependencies(check ${TEST_NAME})
endfunction()


add_api_test(cpu test_cpu.cpp)
if(MIGRAPHX_ENABLE_GPU)
add_api_test(gpu test_gpu.cpp)
# GPU-based tests
endif()
