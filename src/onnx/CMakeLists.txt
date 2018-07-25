find_package(Protobuf REQUIRED)

protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS onnx.proto)
add_library(onnx-proto STATIC ${PROTO_SRCS})
target_include_directories(onnx-proto SYSTEM PUBLIC ${CMAKE_CURRENT_BINARY_DIR} ${PROTOBUF_INCLUDE_DIR})
target_compile_options(onnx-proto PRIVATE -w)
target_link_libraries(onnx-proto PRIVATE ${PROTOBUF_LIBRARY})
set_target_properties(onnx-proto PROPERTIES POSITION_INDEPENDENT_CODE On)

add_library(migraph_onnx onnx.cpp)
rocm_clang_tidy_check(migraph_onnx)
target_link_libraries(migraph_onnx PRIVATE onnx-proto)
target_link_libraries(migraph_onnx PUBLIC migraph)

add_executable(read_onnx read_onnx.cpp)
rocm_clang_tidy_check(read_onnx)
target_link_libraries(read_onnx migraph_onnx)

add_executable(mnist mnist.cpp)
rocm_clang_tidy_check(mnist)
target_link_libraries(mnist migraph_cpu migraph_onnx)

if(MIGRAPH_ENABLE_GPU)
add_executable(verify_onnx verify_onnx.cpp)
rocm_clang_tidy_check(verify_onnx)
target_link_libraries(verify_onnx migraph_onnx migraph_cpu migraph_miopen)
endif()
