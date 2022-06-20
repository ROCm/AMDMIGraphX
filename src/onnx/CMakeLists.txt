find_package(Protobuf REQUIRED)

protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS onnx.proto)
add_library(onnx-proto STATIC ${PROTO_SRCS})
target_include_directories(onnx-proto SYSTEM PUBLIC ${CMAKE_CURRENT_BINARY_DIR} ${PROTOBUF_INCLUDE_DIR})
target_compile_options(onnx-proto PRIVATE -w)
target_link_libraries(onnx-proto PRIVATE ${PROTOBUF_LIBRARY})
set_target_properties(onnx-proto PROPERTIES POSITION_INDEPENDENT_CODE On)

file(GLOB ONNX_SRCS ${CONFIGURE_DEPENDS} *.cpp)
add_library(migraphx_onnx ${ONNX_SRCS})
target_include_directories(migraphx_onnx PRIVATE include)
set_target_properties(migraphx_onnx PROPERTIES EXPORT_NAME onnx)
rocm_set_soversion(migraphx_onnx ${MIGRAPHX_SO_VERSION})
rocm_clang_tidy_check(migraphx_onnx)
target_link_libraries(migraphx_onnx PRIVATE onnx-proto "-Wl,--exclude-libs,ALL")
target_link_libraries(migraphx_onnx PUBLIC migraphx)

rocm_install_targets(
  TARGETS migraphx_onnx
)
