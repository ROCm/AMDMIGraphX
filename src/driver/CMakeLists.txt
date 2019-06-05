
add_executable(driver main.cpp)
target_link_libraries(driver migraphx_cpu migraphx_onnx migraphx_tf)
