
add_executable(gpu-driver
    action.cpp
    compile_pointwise.cpp
    main.cpp
    parser.cpp
)
target_include_directories(gpu-driver PRIVATE include)
target_link_libraries(gpu-driver PRIVATE migraphx_gpu)
