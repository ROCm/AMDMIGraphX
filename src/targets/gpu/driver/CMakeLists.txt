
file(GLOB GPU_DRIVER_SRCS ${CONFIGURE_DEPENDS} ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
add_executable(gpu-driver
    ${GPU_DRIVER_SRCS}
)
target_include_directories(gpu-driver PRIVATE include)
target_link_libraries(gpu-driver PRIVATE migraphx_gpu)
