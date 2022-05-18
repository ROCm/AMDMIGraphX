cmake_minimum_required(VERSION 3.5)
project (custom_rocblas_kernel)

set (CMAKE_CXX_STANDARD 14)
set (EXAMPLE custom_op_rocblas_kernel)


list (APPEND CMAKE_PREFIX_PATH /opt/rocm/rocblas /opt/rocm/hip /opt/rocm)
find_package (migraphx REQUIRED)
find_package (rocblas REQUIRED)

message("source file: " ${EXAMPLE}.cpp " ---> bin: " ${EXAMPLE})
add_executable(${EXAMPLE} ${EXAMPLE}.cpp)

target_link_libraries(${EXAMPLE} migraphx::c roc::rocblas hip::device)
