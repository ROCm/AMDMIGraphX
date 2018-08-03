
add_library(migraph 
    auto_contiguous.cpp
    dead_code_elimination.cpp
    generate.cpp
    program.cpp
    shape.cpp
)
rocm_clang_tidy_check(migraph)
target_include_directories(migraph PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

add_subdirectory(onnx)
add_subdirectory(targets/cpu)
if(MIGRAPH_ENABLE_GPU)
add_subdirectory(targets/gpu)
endif()
