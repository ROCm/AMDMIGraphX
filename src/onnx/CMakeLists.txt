#####################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#####################################################################################

add_library(onnx-proto STATIC onnx.proto)
if(WIN32)
    include(${protobuf_DIR}/protobuf-generate.cmake)
    add_dependencies(onnx-proto protobuf::protoc)
    target_compile_options(onnx-proto
        PRIVATE
            -Wno-used-but-marked-unused
            -Wno-switch-enum
            -Wno-reserved-identifier
            -Wno-inconsistent-missing-destructor-override
            -Wno-suggest-destructor-override
            -Wno-zero-as-null-pointer-constant
            -Wno-undef
            -Wno-unused-parameter
            -Wno-shadow-field-in-constructor
            -Wno-covered-switch-default)
    protobuf_generate(TARGET onnx-proto)
else()
    find_package(Protobuf REQUIRED)
    protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS onnx.proto)
    target_sources(onnx-proto PRIVATE ${PROTO_SRCS} ${PROTO_HDRS})
    target_include_directories(onnx-proto SYSTEM PUBLIC ${CMAKE_CURRENT_BINARY_DIR} ${PROTOBUF_INCLUDE_DIR})
    target_compile_options(onnx-proto PRIVATE -w)
endif()

target_link_libraries(onnx-proto PRIVATE protobuf::libprotobuf)
set_target_properties(onnx-proto PROPERTIES POSITION_INDEPENDENT_CODE On)

file(GLOB ONNX_SRCS CONFIGURE_DEPENDS *.cpp)
add_library(migraphx_onnx ${ONNX_SRCS})
target_include_directories(migraphx_onnx PRIVATE include)
set_target_properties(migraphx_onnx PROPERTIES EXPORT_NAME onnx)
migraphx_generate_export_header(migraphx_onnx)
rocm_set_soversion(migraphx_onnx ${MIGRAPHX_SO_VERSION})
rocm_clang_tidy_check(migraphx_onnx)
if(WIN32)
    # (!!!) Link protobuf statically on Windows (!!!)
    # Refer to protobuf/cmake/README.md for details.
    target_link_libraries(migraphx_onnx PRIVATE onnx-proto protobuf::libprotobuf)
    target_include_directories(migraphx_onnx SYSTEM PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
    target_compile_options(migraphx_onnx
        PRIVATE
            -Wno-used-but-marked-unused
            -Wno-switch-enum
            -Wno-reserved-identifier
            -Wno-inconsistent-missing-destructor-override
            -Wno-suggest-destructor-override
            -Wno-zero-as-null-pointer-constant
            -Wno-undef
            -Wno-unused-parameter
            -Wno-shadow-field-in-constructor
            -Wno-covered-switch-default)
else()
    target_link_libraries(migraphx_onnx PRIVATE onnx-proto "-Wl,--exclude-libs,ALL")
endif()
target_link_libraries(migraphx_onnx PUBLIC migraphx)

rocm_install_targets(
  TARGETS migraphx_onnx
)
