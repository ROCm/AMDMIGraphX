
list(APPEND CMAKE_PREFIX_PATH /opt/rocm /opt/rocm/hip /opt/rocm/hcc)
find_package(miopen)

# rocblas
find_package(rocblas REQUIRED PATHS /opt/rocm)
message(STATUS "Build with rocblas")

if(NOT TARGET MIOpen)
    message(SEND_ERROR "Cant find miopen")
endif()

add_library(migraphx_device
    device/add.cpp
    device/argmax.cpp
    device/argmin.cpp
    device/max.cpp
    device/min.cpp
    device/exp.cpp
    device/erf.cpp
    device/log.cpp
    device/sin.cpp
    device/cos.cpp
    device/tan.cpp
    device/sinh.cpp
    device/cosh.cpp
    device/asin.cpp
    device/acos.cpp
    device/atan.cpp
    device/add_relu.cpp
    device/contiguous.cpp
    device/logsoftmax.cpp
    device/softmax.cpp
    device/convert.cpp
    device/mul.cpp
    device/concat.cpp
    device/pad.cpp
    device/gather.cpp
    device/sub.cpp
    device/div.cpp
    device/clip.cpp
    device/reduce_sum.cpp
)
set_target_properties(migraphx_device PROPERTIES EXPORT_NAME device)
rocm_clang_tidy_check(migraphx_device)
target_link_libraries(migraphx_device migraphx hip::device -Wno-invalid-command-line-argument -amdgpu-target=gfx803 -amdgpu-target=gfx900 -amdgpu-target=gfx906)
target_include_directories(migraphx_device PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_include_directories(migraphx_device PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/device/include>)

add_library(migraphx_gpu
    argmax.cpp
    argmin.cpp
    eliminate_workspace.cpp
    fuse_ops.cpp
    hip.cpp
    target.cpp
    lowering.cpp
    gemm.cpp
    pooling.cpp
    convolution.cpp
    softmax.cpp
    logsoftmax.cpp
    contiguous.cpp
    concat.cpp
    relu.cpp
    leaky_relu.cpp
    tanh.cpp
    batchnorm.cpp
    write_literals.cpp
    rocblas.cpp
    sigmoid.cpp
    abs.cpp
    elu.cpp
    pad.cpp
    gather.cpp
    lrn.cpp
    schedule_model.cpp
    adjust_allocation.cpp
    clip.cpp
    reduce_sum.cpp
)
set_target_properties(migraphx_gpu PROPERTIES EXPORT_NAME gpu)
rocm_clang_tidy_check(migraphx_gpu)
target_link_libraries(migraphx_gpu PUBLIC migraphx MIOpen roc::rocblas)
target_link_libraries(migraphx_gpu PRIVATE migraphx_device)

rocm_install_targets(
  TARGETS migraphx_gpu migraphx_device
  INCLUDE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

