cmake_minimum_required(VERSION 3.5)
project (CAI)

set (MIGRAPHX_FOLDER AMDMIGraphX)
set (MIGRAPHX_BUILD ${MIGRAPHX_FOLDER}/build)

set (CMAKE_CXX_STANDARD 14)

include_directories(../../../${MIGRAPHX_FOLDER}/src/api/include)
link_directories(../../../${MIGRAPHX_BUILD}/lib)

set (mnist_inference mnist_inference.cpp)

message("source file: " mnist_inference.cpp " ---> bin: " mnist_inference)
add_executable(mnist_inference mnist_inference.cpp)
target_link_libraries(mnist_inference migraphx_c)
