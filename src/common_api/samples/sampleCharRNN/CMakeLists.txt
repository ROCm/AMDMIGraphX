file(GLOB SAMPLE_SRCS *.cpp)
file(GLOB COMMON_SRCS ../common/*.cpp)
file(GLOB UTILS_SRC ../utils/*.cpp)
add_executable(sampleCharRNN ${SAMPLE_SRCS} ${COMMON_SRCS} ${UTILS_SRC})

target_include_directories(sampleCharRNN
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../common
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_compile_definitions(sampleCharRNN PRIVATE DEFINE_TRT_ENTRYPOINTS=1)

target_compile_options(sampleCharRNN PRIVATE -Wno-unused-parameter -Wno-deprecated-declarations -Wno-documentation)

target_link_libraries(sampleCharRNN PRIVATE commonapi migraphx_onnx migraphx_ref migraphx_c migraphx migraphx_all_targets)