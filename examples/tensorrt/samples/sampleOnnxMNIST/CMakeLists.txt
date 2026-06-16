file(GLOB SAMPLE_SRCS *.cpp)
file(GLOB COMMON_SRCS ../common/*.cpp)
file(GLOB UTILS_SRC ../utils/*.cpp)
add_executable(sampleOnnxMNIST ${SAMPLE_SRCS} ${COMMON_SRCS} ${UTILS_SRC})

target_include_directories(sampleOnnxMNIST
    PRIVATE
        ${MIGRAPHX_SAMPLE_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/../common
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_compile_definitions(sampleOnnxMNIST PRIVATE DEFINE_TRT_ENTRYPOINTS=1)

target_compile_options(sampleOnnxMNIST PRIVATE -Wno-unused-parameter -Wno-deprecated-declarations -Wno-documentation)

target_link_libraries(sampleOnnxMNIST PRIVATE ${MIGRAPHX_SAMPLE_LIBS})
