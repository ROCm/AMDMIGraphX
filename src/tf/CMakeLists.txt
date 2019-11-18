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
rocm_set_soversion(migraphx_tf ${MIGRAPHX_SO_VERSION})
rocm_clang_tidy_check(migraphx_tf)
target_link_libraries(migraphx_tf PRIVATE tf-proto)
target_link_libraries(migraphx_tf PUBLIC migraphx)

rocm_install_targets(
  TARGETS migraphx_tf
)

