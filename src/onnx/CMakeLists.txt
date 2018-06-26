find_package(Protobuf REQUIRED)

protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS onnx.proto)
add_library(onnx-proto STATIC ${PROTO_SRCS})
target_include_directories(onnx-proto SYSTEM PUBLIC ${CMAKE_CURRENT_BINARY_DIR} ${PROTOBUF_INCLUDE_DIR})
target_compile_options(onnx-proto PRIVATE -w)
target_link_libraries(onnx-proto PRIVATE ${PROTOBUF_LIBRARY})

add_library(rtg_onnx onnx.cpp)
rocm_clang_tidy_check(rtg_onnx)
target_link_libraries(rtg_onnx onnx-proto rtg)

add_executable(read_onnx read_onnx.cpp)
rocm_clang_tidy_check(read_onnx)
target_link_libraries(read_onnx rtg_onnx)

if(RTG_ENABLE_MIOPEN)
add_executable(verify_onnx verify_onnx.cpp)
rocm_clang_tidy_check(verify_onnx)
target_link_libraries(verify_onnx rtg_onnx rtg_cpu rtg_miopen)
endif()
