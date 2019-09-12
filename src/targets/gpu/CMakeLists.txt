
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
    device/mul_add.cpp
    device/exp.cpp
    device/erf.cpp
    device/log.cpp
    device/sin.cpp
    device/cos.cpp
    device/tan.cpp
    device/sinh.cpp
    device/cosh.cpp
    device/tanh.cpp
    device/asin.cpp
    device/acos.cpp
    device/atan.cpp
    device/relu.cpp
    device/add_relu.cpp
    device/contiguous.cpp
    device/logsoftmax.cpp
    device/softmax.cpp
    device/sigmoid.cpp
    device/convert.cpp
    device/mul.cpp
    device/concat.cpp
    device/pad.cpp
    device/gather.cpp
    device/sub.cpp
    device/int8_gemm_pack.cpp
    device/div.cpp
    device/clip.cpp
    device/reduce_sum.cpp
    device/rsqrt.cpp
    device/round.cpp
    device/sqrt.cpp
    device/reduce_mean.cpp
    device/pow.cpp
    device/sqdiff.cpp
    device/sign.cpp
)
set_target_properties(migraphx_device PROPERTIES EXPORT_NAME device)
rocm_set_soversion(migraphx_device ${PROJECT_VERSION})
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
    pooling.cpp
    convolution.cpp
    quant_convolution.cpp
    softmax.cpp
    logsoftmax.cpp
    contiguous.cpp
    concat.cpp
    leaky_relu.cpp
    batchnorm.cpp
    write_literals.cpp
    rocblas.cpp
    abs.cpp
    elu.cpp
    pad.cpp
    gather.cpp
    convert.cpp
    lrn.cpp
    schedule_model.cpp
    adjust_allocation.cpp
    pack_int8_args.cpp
    clip.cpp
    reduce_sum.cpp
    reduce_mean.cpp
    int8_gemm_pack.cpp
    int8_conv_pack.cpp
    gemm_impl.cpp
)
set_target_properties(migraphx_gpu PROPERTIES EXPORT_NAME gpu)
rocm_set_soversion(migraphx_gpu ${PROJECT_VERSION})
rocm_clang_tidy_check(migraphx_gpu)
target_link_libraries(migraphx_gpu PUBLIC migraphx MIOpen roc::rocblas)
target_link_libraries(migraphx_gpu PRIVATE migraphx_device)

rocm_install_targets(
  TARGETS migraphx_gpu migraphx_device
  INCLUDE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

