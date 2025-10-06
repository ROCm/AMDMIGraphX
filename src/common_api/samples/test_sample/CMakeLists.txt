file(GLOB SAMPLE_SRCS *.cpp)
file(GLOB COMMON_SRCS ../common/*.cpp)
file(GLOB UTILS_SRC ../utils/*.cpp)
add_executable(test_sample ${SAMPLE_SRCS} ${COMMON_SRCS} ${UTILS_SRC})
# rocm_clang_tidy_check(test_sample)

target_include_directories(test_sample
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../common
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(test_sample PRIVATE commonapi migraphx_onnx migraphx_ref migraphx_c migraphx migraphx_all_targets)