file(GLOB COMMONAPI_SRC *.cpp)

add_library(commonapi SHARED ${COMMONAPI_SRC})

target_include_directories(commonapi
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# only temporary:
target_compile_options(commonapi PRIVATE -Wno-unused-parameter -Wno-deprecated-declarations -Wno-documentation)

# set_target_properties(commonapi PROPERTIES OUTPUT_NAME "commonapi")
set_target_properties(commonapi PROPERTIES EXPORT_NAME commonapi)
migraphx_generate_export_header(commonapi)
rocm_set_soversion(commonapi ${MIGRAPHX_SO_VERSION})
rocm_clang_tidy_check(commonapi)

# find_package(hiprtc REQUIRED)
# target_link_libraries(commonapi PUBLIC migraphx)

find_package(hiprtc REQUIRED)
# target_compile_definitions(commonapi PRIVATE -DMIGRAPHX_USE_HIPRTC=1)
target_link_libraries(commonapi PUBLIC hiprtc::hiprtc migraphx_onnx)

install(TARGETS commonapi LIBRARY DESTINATION lib)
install(DIRECTORY ../include/migraphx/common_api DESTINATION include/migraphx)

add_subdirectory(layers)
