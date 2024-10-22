project(MGXLlama2)
cmake_minimum_required(VERSION 3.22)

set(TARGET_NAME mgxllama2)

set(HARNESS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/harness)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CXX /opt/rocm/llvm/bin/clang++)
 
list (APPEND CMAKE_PREFIX_PATH /opt/rocm ${HIP_PATH})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3 -W -Wall -pthread -D__HIP_PLATFORM_HCC__=1")

find_package(migraphx REQUIRED)
find_package(hip REQUIRED)

include_directories(${HARNESS_DIR})

add_executable(${TARGET_NAME}
    mgxllama2.cc
)

target_include_directories(${TARGET_NAME}
    PUBLIC ${HARNESS_DIR}
)

target_link_libraries(${TARGET_NAME}
    migraphx::c
    hip::device
    pthread
)

