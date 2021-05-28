
add_executable(gpu-jit-driver jit_driver.cpp)
target_link_libraries(gpu-jit-driver PRIVATE migraphx_gpu)
