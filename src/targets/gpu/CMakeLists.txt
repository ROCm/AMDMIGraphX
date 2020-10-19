
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
    device/acosh.cpp
    device/add.cpp
    device/add_clip.cpp
    device/add_relu.cpp
    device/add_sigmoid.cpp
    device/add_tanh.cpp
    device/argmax.cpp
    device/argmin.cpp
    device/asin.cpp
    device/asinh.cpp
    device/atan.cpp
    device/atanh.cpp
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
    device/gelu.cpp
    device/greater.cpp
    device/int8_gemm_pack.cpp
    device/layernorm.cpp
    device/less.cpp
    device/log.cpp
    device/logsoftmax.cpp
    device/max.cpp
    device/min.cpp
    device/mul.cpp
    device/mul_add.cpp
    device/mul_add_relu.cpp
    device/pad.cpp
    device/pow.cpp
    device/prelu.cpp
    device/recip.cpp
    device/reduce_max.cpp
    device/reduce_mean.cpp
    device/reduce_min.cpp
    device/reduce_sum.cpp
    device/reduce_prod.cpp
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
    device/rnn_variable_seq_lens.cpp
    device/equal.cpp
)
set_target_properties(migraphx_device PROPERTIES EXPORT_NAME device)
rocm_set_soversion(migraphx_device ${MIGRAPHX_SO_VERSION})
rocm_clang_tidy_check(migraphx_device)
target_compile_options(migraphx_device PRIVATE -std=c++17 -fno-gpu-rdc -Wno-unused-command-line-argument -Xclang -fallow-half-arguments-and-returns)
target_link_libraries(migraphx_device migraphx hip::device -fno-gpu-rdc -Wno-invalid-command-line-argument -Wno-unused-command-line-argument)
if(CMAKE_CXX_COMPILER MATCHES ".*hcc")
    set(AMDGPU_TARGETS "gfx803;gfx900;gfx906" CACHE STRING "")
    foreach(AMDGPU_TARGET ${AMDGPU_TARGETS})
        target_compile_options(migraphx_device PRIVATE -amdgpu-target=${AMDGPU_TARGET})
        target_link_libraries(migraphx_device -amdgpu-target=${AMDGPU_TARGET})
    endforeach()
else()
    target_compile_options(migraphx_device PRIVATE -Wno-cuda-compat)
endif()
check_cxx_compiler_flag("--cuda-host-only -fhip-lambda-host-device -x hip" HAS_HIP_LAMBDA_HOST_DEVICE)
if(HAS_HIP_LAMBDA_HOST_DEVICE)
  message(STATUS "Enable -fhip-lambda-host-device")
  target_compile_options(migraphx_device PRIVATE -fhip-lambda-host-device)
endif()
target_include_directories(migraphx_device PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_include_directories(migraphx_device PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/device/include>)

add_library(migraphx_gpu
    analyze_streams.cpp
    argmax.cpp
    argmin.cpp
    eliminate_workspace.cpp
    fuse_ops.cpp
    hip.cpp
    target.cpp
    lowering.cpp
    pooling.cpp
    convolution.cpp
    deconvolution.cpp
    quant_convolution.cpp
    softmax.cpp
    logsoftmax.cpp
    concat.cpp
    leaky_relu.cpp
    batch_norm_inference.cpp
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
    rnn_variable_seq_lens.cpp
    sync_device.cpp
)
set_target_properties(migraphx_gpu PROPERTIES EXPORT_NAME gpu)
function(register_migraphx_gpu_ops PREFIX)
    foreach(OP ${ARGN})
        register_op(migraphx_gpu HEADER migraphx/gpu/${OP}.hpp OPERATORS gpu::${PREFIX}${OP} INCLUDES migraphx/gpu/context.hpp)
    endforeach()
endfunction()
register_migraphx_gpu_ops(hip_
    acosh
    acos
    add
    argmax
    argmin
    asinh
    asin
    atanh
    atan
    ceil
    clip
    concat
    convert
    cosh
    cos
    div
    equal
    erf
    exp
    floor
    gather
    greater
    less
    log
    logsoftmax
    max
    min
    mul
    pad
    pow
    prelu
    recip
    reduce_max
    reduce_mean
    reduce_min
    reduce_prod
    reduce_sum
    relu
    round
    rsqrt
    sigmoid
    sign
    sinh
    sin
    softmax
    sqdiff
    sqrt
    sub
    tanh
    tan
)
register_migraphx_gpu_ops(miopen_
    abs
    batch_norm_inference
    contiguous
    convolution
    deconvolution
    elu
    int8_conv_pack
    leaky_relu
    lrn
    pooling
    quant_convolution
)
register_op(migraphx_gpu 
    HEADER migraphx/gpu/rnn_variable_seq_lens.hpp 
    OPERATORS gpu::hip_rnn_var_sl_shift_sequence gpu::hip_rnn_var_sl_shift_output gpu::hip_rnn_var_sl_last_output
    INCLUDES migraphx/gpu/context.hpp)
register_op(migraphx_gpu 
    HEADER migraphx/gpu/int8_gemm_pack.hpp 
    OPERATORS gpu::hip_int8_gemm_pack_a gpu::hip_int8_gemm_pack_b
    INCLUDES migraphx/gpu/context.hpp)
register_op(migraphx_gpu 
    HEADER migraphx/gpu/gemm.hpp 
    OPERATORS gpu::rocblas_gemm<op::dot> gpu::rocblas_gemm<op::quant_dot>
    INCLUDES migraphx/gpu/context.hpp)
rocm_set_soversion(migraphx_gpu ${MIGRAPHX_SO_VERSION})
rocm_clang_tidy_check(migraphx_gpu)
# Workaround broken rocblas headers
target_compile_definitions(migraphx_gpu PUBLIC -D__HIP_PLATFORM_HCC__=1)
target_compile_options(migraphx_gpu PRIVATE -std=c++17)
target_link_libraries(migraphx_gpu PUBLIC migraphx MIOpen roc::rocblas)
target_link_libraries(migraphx_gpu PRIVATE migraphx_device)

rocm_install_targets(
  TARGETS migraphx_gpu migraphx_device
  INCLUDE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

