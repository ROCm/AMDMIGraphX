find_package(Protobuf REQUIRED)

protobuf_generate_cpp(
    PROTO_SRCS PROTO_HDRS 
    graph.proto
    node_def.proto
    attr_value.proto
    tensor.proto
    tensor_shape.proto
    resource_handle.proto
    types.proto
    function.proto
    op_def.proto
    versions.proto
)
add_library(tf-proto STATIC ${PROTO_SRCS})
target_include_directories(tf-proto SYSTEM PUBLIC ${CMAKE_CURRENT_BINARY_DIR} ${PROTOBUF_INCLUDE_DIR})
target_compile_options(tf-proto PRIVATE -w)
target_link_libraries(tf-proto PRIVATE ${PROTOBUF_LIBRARY})
set_target_properties(tf-proto PROPERTIES POSITION_INDEPENDENT_CODE On)

add_library(migraphx_tf tf.cpp)
set_target_properties(migraphx_tf PROPERTIES EXPORT_NAME tf)
rocm_clang_tidy_check(migraphx_tf)
target_link_libraries(migraphx_tf PRIVATE tf-proto)
target_link_libraries(migraphx_tf PUBLIC migraphx)

rocm_install_targets(
  TARGETS migraphx_tf
)

add_executable(read_tf read_tf.cpp)
rocm_clang_tidy_check(read_tf)
target_link_libraries(read_tf migraphx_tf)


# add_executable(verify_onnx verify_onnx.cpp)
# rocm_clang_tidy_check(verify_onnx)
# target_link_libraries(verify_onnx migraphx_onnx migraphx_cpu migraphx_gpu)

# add_executable(perf_onnx perf_onnx.cpp)
# rocm_clang_tidy_check(perf_onnx)
# target_link_libraries(perf_onnx migraphx_onnx migraphx_cpu migraphx_gpu)
# endif()
