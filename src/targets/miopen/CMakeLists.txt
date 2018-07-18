
list(APPEND CMAKE_PREFIX_PATH /opt/rocm /opt/rocm/hip /opt/rocm/hcc)
find_package(miopen)

# rocblas
find_package(rocblas REQUIRED PATHS /opt/rocm)
message(STATUS "Build with rocblas")

if(NOT TARGET MIOpen)
    message(SEND_ERROR "Cant find miopen")
endif()

add_library(migraph_miopen
    hip.cpp
    miopen_target.cpp
    miopen_lowering.cpp
    miopen_write_literals.cpp
    rocblas.cpp
)
rocm_clang_tidy_check(migraph_miopen)
target_link_libraries(migraph_miopen migraph MIOpen rocblas)
target_include_directories(migraph_miopen PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
