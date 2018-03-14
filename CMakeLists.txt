cmake_minimum_required(VERSION 3.5)

project(rtglib)

find_package(Protobuf REQUIRED)

protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS onnx.proto)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_executable(read_onnx src/read_onnx.cpp ${PROTO_SRCS})
target_include_directories(read_onnx PUBLIC ${PROTOBUF_INLCUDE_DIR})
target_link_libraries(read_onnx ${PROTOBUF_LIBRARY})
