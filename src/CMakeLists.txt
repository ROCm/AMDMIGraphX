
add_library(rtg 
    program.cpp
    shape.cpp
)
rocm_clang_tidy_check(rtg)
target_include_directories(rtg PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

add_subdirectory(onnx)
add_subdirectory(targets/cpu)
add_subdirectory(targets/miopen)
