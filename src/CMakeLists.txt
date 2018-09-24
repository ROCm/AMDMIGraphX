
add_library(migraph 
    auto_contiguous.cpp
    dead_code_elimination.cpp
    eliminate_allocation.cpp
    eliminate_contiguous.cpp
    fwd_conv_batchnorm_rewrite.cpp
    env.cpp
    generate.cpp
    instruction.cpp
    program.cpp
    shape.cpp
    simplify_reshapes.cpp
    opt/memory_coloring.cpp
    opt/memory_coloring_impl.cpp
)
rocm_clang_tidy_check(migraph)
target_include_directories(migraph PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

add_subdirectory(onnx)
add_subdirectory(targets/cpu)
if(MIGRAPH_ENABLE_GPU)
add_subdirectory(targets/gpu)
endif()

#install (TARGETS migraph
#   LIBRARY DESTINATION /opt/rocm/lib)

#install (DIRECTORY include/migraph DESTINATION /opt/rocm/include)
 

