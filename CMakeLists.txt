cmake_minimum_required(VERSION 3.5)

project(rtglib)

add_compile_options(-std=c++14)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(EnableCompilerWarnings)

add_library(rtg 
    src/program.cpp
    src/shape.cpp
)
target_include_directories(rtg PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

add_subdirectory(onnx)
add_subdirectory(test)
