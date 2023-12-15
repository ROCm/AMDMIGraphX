

function(add_onnx_test TEST_NAME)
    add_executable(${TEST_NAME} ${ARGN})
    rocm_clang_tidy_check(${TEST_NAME})
    target_link_libraries(${TEST_NAME} migraphx_onnx migraphx_ref)
    target_include_directories(${TEST_NAME} PUBLIC ../include include)
    add_test(NAME ${TEST_NAME} COMMAND $<TARGET_FILE:${TEST_NAME}> WORKING_DIRECTORY ${TEST_ONNX_DIR})
    rocm_mark_as_test(${TEST_NAME})
endfunction()

file(GLOB ONNX_PARSE_TESTS CONFIGURE_DEPENDS parse/*.cpp)

add_onnx_test(test_onnx_test ${ONNX_PARSE_TESTS})
add_onnx_test(test_onnx_rnn_test onnx_rnn_test.cpp)
add_onnx_test(test_verify_onnx verify_onnx.cpp)

