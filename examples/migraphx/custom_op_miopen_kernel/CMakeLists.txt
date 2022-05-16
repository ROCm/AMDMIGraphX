cmake_minimum_required(VERSION 3.5)
project (custom_miopen_kernel)

set (CMAKE_CXX_STANDARD 14)
set (EXAMPLE custom_op_miopen_kernel)


list (APPEND CMAKE_PREFIX_PATH /home/umayadav/migraphx-install/opt/rocm /opt/rocm/miopen /opt/rocm/hip /opt/rocm)
find_package (migraphx REQUIRED)
find_package (miopen REQUIRED)

message("source file: " ${EXAMPLE}.cpp " ---> bin: " ${EXAMPLE})
add_executable(${EXAMPLE} ${EXAMPLE}.cpp)

target_link_libraries(${EXAMPLE} migraphx::c MIOpen hip::device)
