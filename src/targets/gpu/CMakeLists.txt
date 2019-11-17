
list(APPEND CMAKE_PREFIX_PATH /opt/rocm /opt/rocm/hip /opt/rocm/hcc)
find_package(miopen)

# rocblas
find_package(rocblas REQUIRED PATHS /opt/rocm)
message(STATUS "Build with rocblas")

if(NOT TARGET MIOpen)
    message(SEND_ERROR "Cant find miopen")
endif()

add_library(migraphx_device
    device/acos.cpp
    device/add.cpp
    device/add_clip.cpp
    device/add_relu.cpp
    device/add_sigmoid.cpp
    device/add_tanh.cpp
    device/argmax.cpp
    device/argmin.cpp
    device/asin.cpp
    device/atan.cpp
    device/ceil.cpp
    device/clip.cpp
    device/concat.cpp
    device/contiguous.cpp
    device/convert.cpp
    device/cos.cpp
    device/cosh.cpp
    device/div.cpp
    device/erf.cpp
    device/exp.cpp
    device/floor.cpp
    device/gather.cpp
    device/int8_gemm_pack.cpp
    device/log.cpp
    device/logsoftmax.cpp
    device/max.cpp
    device/min.cpp
    device/mul.cpp
    device/mul_add.cpp
    device/mul_add_relu.cpp
    device/pad.cpp
    device/pow.cpp
    device/reduce_max.cpp
    device/reduce_mean.cpp
    device/reduce_min.cpp
    device/reduce_sum.cpp
    device/relu.cpp
    device/round.cpp
    device/rsqrt.cpp
    device/sigmoid.cpp
    device/sign.cpp
    device/sin.cpp
    device/sinh.cpp
    device/softmax.cpp
    device/sqdiff.cpp
    device/sqrt.cpp
    device/sub.cpp
    device/tan.cpp
    device/tanh.cpp
)
set_target_properties(migraphx_device PROPERTIES EXPORT_NAME device)
rocm_set_soversion(migraphx_device ${MIGRAPHX_SO_VERSION})
rocm_clang_tidy_check(migraphx_device)
target_compile_options(migraphx_device PRIVATE -std=c++17)
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
    int8_gemm_pack.cpp
    int8_conv_pack.cpp
    gemm_impl.cpp
    preallocate_param.cpp
)
set_target_properties(migraphx_gpu PROPERTIES EXPORT_NAME gpu)
rocm_set_soversion(migraphx_gpu ${MIGRAPHX_SO_VERSION})
rocm_clang_tidy_check(migraphx_gpu)
target_link_libraries(migraphx_gpu PUBLIC migraphx MIOpen roc::rocblas)
target_link_libraries(migraphx_gpu PRIVATE migraphx_device)

rocm_install_targets(
  TARGETS migraphx_gpu migraphx_device
  INCLUDE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

