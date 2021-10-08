
list(APPEND CMAKE_PREFIX_PATH /opt/rocm /opt/rocm/hip /opt/rocm/hcc)
find_package(miopen)

# rocblas
find_package(rocblas REQUIRED PATHS /opt/rocm)
message(STATUS "Build with rocblas")

if(NOT TARGET MIOpen)
    message(SEND_ERROR "Cant find miopen")
endif()

include(Embed)
file(GLOB KERNEL_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/include/migraphx/kernels/*.hpp)
message(STATUS "KERNEL_FILES: ${KERNEL_FILES}")
add_embed_library(migraphx_kernels ${KERNEL_FILES})

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
    device/equal.cpp
    device/erf.cpp
    device/exp.cpp
    device/fill.cpp
    device/floor.cpp
    device/gather.cpp
    device/gelu.cpp
    device/greater.cpp
    device/int8_gemm_pack.cpp
    device/layernorm.cpp
    device/less.cpp
    device/log.cpp
    device/logical_and.cpp
    device/logical_or.cpp
    device/logical_xor.cpp
    device/logsoftmax.cpp
    device/max.cpp
    device/min.cpp
    device/mul.cpp
    device/mul_add.cpp
    device/mul_add_relu.cpp
    device/multinomial.cpp
    device/nonzero.cpp
    device/pad.cpp
    device/pow.cpp
    device/prelu.cpp
    device/prefix_scan_sum.cpp
    device/recip.cpp
    device/reduce_max.cpp
    device/reduce_mean.cpp
    device/reduce_min.cpp
    device/reduce_sum.cpp
    device/reduce_prod.cpp
    device/relu.cpp
    device/reverse.cpp
    device/rnn_variable_seq_lens.cpp
    device/round.cpp
    device/rsqrt.cpp
    device/scatter.cpp
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
    device/topk.cpp
    device/unary_not.cpp
    device/where.cpp
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
    abs.cpp
    analyze_streams.cpp
    allocation_model.cpp
    argmax.cpp
    argmin.cpp
    batch_norm_inference.cpp
    clip.cpp
    code_object_op.cpp
    compile_hip.cpp
    compile_hip_code_object.cpp
    compile_pointwise.cpp
    concat.cpp
    convert.cpp
    convolution.cpp
    deconvolution.cpp
    device_name.cpp
    eliminate_workspace.cpp
    elu.cpp
    fuse_ops.cpp
    gather.cpp
    gemm_impl.cpp
    hip.cpp
    int8_conv_pack.cpp
    int8_gemm_pack.cpp
    kernel.cpp
    lowering.cpp
    logsoftmax.cpp
    loop.cpp
    lrn.cpp
    leaky_relu.cpp
    mlir_conv.cpp
    multinomial.cpp
    nonzero.cpp
    pack_args.cpp
    pack_int8_args.cpp
    pad.cpp
    pooling.cpp
    quant_convolution.cpp
    reverse.cpp
    rnn_variable_seq_lens.cpp
    rocblas.cpp
    scatter.cpp
    schedule_model.cpp
    softmax.cpp
    sync_device.cpp
    target.cpp
    topk.cpp
    write_literals.cpp
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
    logical_and
    logical_or
    logical_xor
    loop
    max
    min
    mul
    multinomial
    nonzero
    pad
    pow
    prelu
    prefix_scan_sum
    recip
    reduce_max
    reduce_mean
    reduce_min
    reduce_prod
    reduce_sum
    relu
    reverse
    round
    rsqrt
    scatter
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
    topk
    unary_not
    where
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

# look for offload bundler
get_filename_component(CMAKE_CXX_COMPILER_PATH "${CMAKE_CXX_COMPILER}" PATH)
if(CMAKE_CXX_COMPILER MATCHES ".*clang\\+\\+$")
    find_program(MIGRAPHX_OFFLOADBUNDLER_BIN clang-offload-bundler
        HINTS ${CMAKE_CXX_COMPILER_PATH}
        PATH_SUFFIXES bin
        PATHS /opt/rocm/llvm
    )
else()
    find_program(MIGRAPHX_EXTRACT_KERNEL extractkernel
        PATH_SUFFIXES bin
        HINTS ${CMAKE_CXX_COMPILER_PATH}
        PATHS
            /opt/rocm/hip
            /opt/rocm/hcc
            /opt/rocm
    )
endif()

message(STATUS "clang-offload-bundler: ${MIGRAPHX_OFFLOADBUNDLER_BIN}")
message(STATUS "extractkernel: ${MIGRAPHX_EXTRACT_KERNEL}")

set(MIGRAPHX_ENABLE_MLIR OFF CACHE BOOL "")
if(MIGRAPHX_ENABLE_MLIR)
    find_library(LIBMLIRMIOPEN MLIRMIOpenThin REQUIRED)
    # REQUIRED is not supported before cmake 3.18
    if(NOT LIBMLIRMIOPEN)
        message(FATAL_ERROR "libMLIRMIOpenThin not found")
    else()
        message(STATUS "Build with libMLIRMIOpenThin: " ${LIBMLIRMIOPEN})
    endif()

    target_compile_definitions(migraphx_gpu PRIVATE "-DMIGRAPHX_MLIR_MIOPEN_SUPPORT")
    target_link_libraries(migraphx_gpu PUBLIC ${LIBMLIRMIOPEN})
endif()

set(MIGRAPHX_USE_HIPRTC OFF CACHE BOOL "")
if(MIGRAPHX_USE_HIPRTC)
target_compile_definitions(migraphx_gpu PRIVATE -DMIGRAPHX_USE_HIPRTC=1)
else()
# Get flags needed to compile hip
include(TargetFlags)
target_flags(HIP_COMPILER_FLAGS hip::device)
# Remove cuda arch flags
string(REGEX REPLACE --cuda-gpu-arch=[a-z0-9]+ "" HIP_COMPILER_FLAGS "${HIP_COMPILER_FLAGS}")
string(REGEX REPLACE --offload-arch=[a-z0-9:+-]+ "" HIP_COMPILER_FLAGS "${HIP_COMPILER_FLAGS}")
string(REPLACE "$<LINK_LANGUAGE:CXX>" "1" HIP_COMPILER_FLAGS "${HIP_COMPILER_FLAGS}")
string(REPLACE "SHELL:" "" HIP_COMPILER_FLAGS "${HIP_COMPILER_FLAGS}")
message(STATUS "Hip compiler flags: ${HIP_COMPILER_FLAGS}")
target_compile_definitions(migraphx_gpu PRIVATE 
    "-DMIGRAPHX_HIP_COMPILER=${CMAKE_CXX_COMPILER}" 
    "-DMIGRAPHX_HIP_COMPILER_FLAGS=${HIP_COMPILER_FLAGS}" 
    "-DMIGRAPHX_OFFLOADBUNDLER_BIN=${MIGRAPHX_OFFLOADBUNDLER_BIN}"
    "-DMIGRAPHX_EXTRACT_KERNEL=${MIGRAPHX_EXTRACT_KERNEL}"
    "-DMIGRAPHX_USE_HIPRTC=0"
)
endif()

# Check miopen find mode api
include(CheckLibraryExists)
get_target_property(MIOPEN_LOCATION MIOpen LOCATION)
check_library_exists(MIOpen "miopenHiddenSetConvolutionFindMode" "${MIOPEN_LOCATION}" HAS_FIND_MODE_API)
if(HAS_FIND_MODE_API)
    target_compile_definitions(migraphx_gpu PUBLIC -DMIGRAPHX_HAS_FIND_MODE_API)
    message(STATUS "MIOpen has find mode api")
else()
    message(STATUS "MIOpen does not have find mode api")
endif()

# Workaround broken rocblas headers
target_compile_definitions(migraphx_gpu PUBLIC -D__HIP_PLATFORM_HCC__=1)
target_link_libraries(migraphx_gpu PUBLIC migraphx MIOpen roc::rocblas)
target_link_libraries(migraphx_gpu PRIVATE migraphx_device migraphx_kernels)

add_subdirectory(driver)

rocm_install_targets(
  TARGETS migraphx_gpu migraphx_device
  INCLUDE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

