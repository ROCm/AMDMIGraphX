
add_executable(driver 
    main.cpp
    verify.cpp
    perf.cpp
    resnet50.cpp)
rocm_clang_tidy_check(driver)
target_link_libraries(driver migraphx_cpu migraphx_onnx migraphx_tf)
if(MIGRAPHX_ENABLE_GPU)
target_link_libraries(driver migraphx_gpu)
target_compile_definitions(driver PRIVATE -DHAVE_GPU)
endif()
