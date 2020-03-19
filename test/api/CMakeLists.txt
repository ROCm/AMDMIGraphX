
function(add_api_test TEST_NAME TEST_SRC)
    set(NAME test_api_${TEST_NAME})
    add_executable(${NAME} EXCLUDE_FROM_ALL ${TEST_SRC})
    rocm_clang_tidy_check(${NAME})
    target_link_libraries(${NAME} migraphx_c)
    target_include_directories(${NAME} PUBLIC ../include)
    add_test(NAME ${NAME} COMMAND $<TARGET_FILE:${NAME}> WORKING_DIRECTORY ${TEST_ONNX_DIR}) 
    add_dependencies(tests ${NAME})
    add_dependencies(check ${NAME})
endfunction()


add_api_test(cpu test_cpu.cpp)
if(MIGRAPHX_ENABLE_GPU)
add_api_test(gpu test_gpu.cpp)
# GPU-based tests
endif()
