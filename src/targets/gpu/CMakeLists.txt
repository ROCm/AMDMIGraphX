
list(APPEND CMAKE_PREFIX_PATH /opt/rocm /opt/rocm/hip /opt/rocm/hcc)
find_package(miopen)

# rocblas
find_package(rocblas REQUIRED PATHS /opt/rocm)
message(STATUS "Build with rocblas")

if(NOT TARGET MIOpen)
    message(SEND_ERROR "Cant find miopen")
endif()

add_library(migraph_device 
    hip_contiguous.cpp
)
rocm_clang_tidy_check(migraph_device)
target_link_libraries(migraph_device migraph hip::device)
target_include_directories(migraph_device PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

add_library(migraph_gpu
    eliminate_allocation.cpp
    eliminate_workspace.cpp
    hip.cpp
    target.cpp
    lowering.cpp
    write_literals.cpp
    rocblas.cpp
)
rocm_clang_tidy_check(migraph_gpu)
target_link_libraries(migraph_gpu migraph MIOpen migraph_device roc::rocblas)
target_include_directories(migraph_gpu PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
