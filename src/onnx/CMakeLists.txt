find_package(Protobuf REQUIRED)

protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS onnx.proto)
add_library(onnx-proto STATIC ${PROTO_SRCS})
target_include_directories(onnx-proto SYSTEM PUBLIC ${CMAKE_CURRENT_BINARY_DIR} ${PROTOBUF_INCLUDE_DIR})
target_compile_options(onnx-proto PRIVATE -w)
target_link_libraries(onnx-proto PRIVATE ${PROTOBUF_LIBRARY})
set_target_properties(onnx-proto PROPERTIES POSITION_INDEPENDENT_CODE On)

add_library(migraphx_onnx onnx.cpp)
set_target_properties(migraphx_onnx PROPERTIES EXPORT_NAME onnx)
rocm_set_soversion(migraphx_onnx ${PROJECT_VERSION})
rocm_clang_tidy_check(migraphx_onnx)
target_link_libraries(migraphx_onnx PRIVATE onnx-proto)
target_link_libraries(migraphx_onnx PUBLIC migraphx)

rocm_install_targets(
  TARGETS migraphx_onnx
)

if(MIGRAPHX_ENABLE_GPU)
add_executable(mnist mnist.cpp)
rocm_clang_tidy_check(mnist)
target_link_libraries(mnist migraphx_cpu migraphx_gpu migraphx_onnx)

add_executable(cifar10 cifar10.cpp)
rocm_clang_tidy_check(cifar10)
target_link_libraries(cifar10 migraphx_cpu migraphx_gpu migraphx_onnx)

endif()