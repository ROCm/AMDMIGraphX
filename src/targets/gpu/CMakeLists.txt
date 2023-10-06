# ####################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# ####################################################################################

list(APPEND CMAKE_PREFIX_PATH /opt/rocm)
find_package(hip)
if(NOT GPU_TARGETS)
    message(FATAL_ERROR "HIP package is broken and has no GPU_TARGETS, please pass -DGPU_TARGETS=$(/opt/rocm/bin/rocminfo | grep -o -m1 'gfx.*') to cmake to build for your gpu.")
endif()
find_package(miopen)

# rocblas
find_package(rocblas REQUIRED PATHS /opt/rocm)
message(STATUS "Build with rocblas")

if(NOT TARGET MIOpen)
    message(SEND_ERROR "Cant find miopen")
endif()

if(NOT WIN32)
    # TODO: re-enable when CK is ported to Windows
    find_package(composable_kernel 1.0.0 REQUIRED COMPONENTS jit_library)
endif()

if(BUILD_DEV)
    set(MIGRAPHX_USE_HIPRTC OFF CACHE BOOL "Use hipRTC APIs")
else()
    set(MIGRAPHX_USE_HIPRTC ON CACHE BOOL "Use hipRTC APIs")
endif()

file(GLOB KERNEL_FILES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/include/migraphx/kernels/*.hpp)
message(STATUS "KERNEL_FILES: ${KERNEL_FILES}")

if(WIN32)
    # TODO: re-enable when CK is ported to Windows
    list(REMOVE_ITEM KERNEL_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels/include/migraphx/kernels/ck_gemm.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels/include/migraphx/kernels/ck.hpp)
endif()

include(Embed)
add_embed_library(migraphx_kernels ${KERNEL_FILES} RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/kernels/include/)

configure_file(device/targets.hpp.in include/migraphx/gpu/device/targets.hpp)
file(GLOB DEVICE_GPU_SRCS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/device/*.cpp)
add_library(migraphx_device ${DEVICE_GPU_SRCS})

add_library(compile_for_gpu INTERFACE)
target_compile_options(compile_for_gpu INTERFACE -std=c++17 -fno-gpu-rdc -Wno-cuda-compat -Wno-unused-command-line-argument -Xclang -fallow-half-arguments-and-returns)
target_link_libraries(compile_for_gpu INTERFACE hip::device -fno-gpu-rdc -Wno-invalid-command-line-argument -Wno-unused-command-line-argument -Wno-option-ignored)
check_cxx_compiler_flag("--cuda-host-only -fhip-lambda-host-device -x hip" HAS_HIP_LAMBDA_HOST_DEVICE)

if(HAS_HIP_LAMBDA_HOST_DEVICE)
    message(STATUS "Enable -fhip-lambda-host-device")
    target_compile_options(compile_for_gpu INTERFACE -fhip-lambda-host-device)
endif()

set_target_properties(migraphx_device PROPERTIES EXPORT_NAME device)
rocm_set_soversion(migraphx_device ${MIGRAPHX_SO_VERSION})
rocm_clang_tidy_check(migraphx_device)
target_link_libraries(migraphx_device PUBLIC migraphx)
target_link_libraries(migraphx_device PRIVATE compile_for_gpu)
target_include_directories(migraphx_device PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_include_directories(migraphx_device PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_BINAR_DIR}/include>)
target_include_directories(migraphx_device PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/device/include>)
target_compile_options(migraphx_device PRIVATE -Wno-ignored-attributes)
migraphx_generate_export_header(migraphx_device DIRECTORY migraphx/gpu/device)

add_library(kernel_file_check EXCLUDE_FROM_ALL)

foreach(KERNEL_FILE ${KERNEL_FILES})
    get_filename_component(KERNEL_BASE_FILE ${KERNEL_FILE} NAME_WE)
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/kernels/include/migraphx/kernels/${KERNEL_BASE_FILE}.cpp "#include <migraphx/kernels/${KERNEL_BASE_FILE}.hpp>\n")
    target_sources(kernel_file_check PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/kernels/include/migraphx/kernels/${KERNEL_BASE_FILE}.cpp)
endforeach()

target_compile_definitions(kernel_file_check PRIVATE -DMIGRAPHX_NLOCAL=256)
target_include_directories(kernel_file_check PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/kernels/include/>)
target_link_libraries(kernel_file_check compile_for_gpu)

rocm_clang_tidy_check(kernel_file_check)

file(GLOB JIT_GPU_SRCS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/jit/*.cpp)

if(WIN32)
    # TODO: re-enable when CK is ported to Windows
    list(REMOVE_ITEM JIT_GPU_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/jit/ck_gemm.cpp)
endif()

add_library(migraphx_gpu
    abs.cpp
    analyze_streams.cpp
    allocation_model.cpp
    argmax.cpp
    argmin.cpp
    code_object_op.cpp
    compile_ops.cpp
    compile_gen.cpp
    compile_hip.cpp
    compile_hip_code_object.cpp
    compile_miopen.cpp
    compiler.cpp
    device_name.cpp
    fuse_ck.cpp
    fuse_mlir.cpp
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
    mlir.cpp
    multinomial.cpp
    no_device.cpp
    nonzero.cpp
    pack_args.cpp
    pack_int8_args.cpp
    prefuse_ops.cpp
    pad.cpp
    perfdb.cpp
    pooling.cpp
    reverse.cpp
    rnn_variable_seq_lens.cpp
    rocblas.cpp
    scatter.cpp
    schedule_model.cpp
    sync_device.cpp
    target.cpp
    time_op.cpp
    topk.cpp
    write_literals.cpp
    ${JIT_GPU_SRCS}
)

set_target_properties(migraphx_gpu PROPERTIES EXPORT_NAME gpu)
migraphx_generate_export_header(migraphx_gpu)

function(register_migraphx_gpu_ops PREFIX)
    foreach(OP ${ARGN})
        register_op(migraphx_gpu HEADER migraphx/gpu/${OP}.hpp OPERATORS gpu::${PREFIX}${OP} INCLUDES migraphx/gpu/context.hpp)
    endforeach()
endfunction()

register_migraphx_gpu_ops(hip_
    argmax
    argmin
    gather
    logsoftmax
    loop
    multinomial
    nonzero
    pad
    prefix_scan_sum
    reverse
    scatter
    topk
)
register_migraphx_gpu_ops(miopen_
    abs
    contiguous
    int8_conv_pack
    lrn
    pooling
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
register_op(migraphx_gpu HEADER migraphx/gpu/convolution.hpp
    OPERATORS gpu::miopen_convolution<op::convolution> gpu::miopen_convolution<op::convolution_backwards> gpu::miopen_convolution<op::quant_convolution>
    INCLUDES migraphx/gpu/context.hpp)
rocm_set_soversion(migraphx_gpu ${MIGRAPHX_SO_VERSION})
rocm_clang_tidy_check(migraphx_gpu)

set(MIGRAPHX_ENABLE_MLIR ON CACHE BOOL "")

if(MIGRAPHX_ENABLE_MLIR)
    # Find package rocMLIR
    find_package(rocMLIR 1.0.0 CONFIG REQUIRED)
    message(STATUS "Build with rocMLIR::rockCompiler ${rocMLIR_VERSION}")
    target_compile_definitions(migraphx_gpu PRIVATE "-DMIGRAPHX_MLIR")
    # Make this private to avoid multiple inclusions of LLVM symbols.
    # TODO: Fix rocMLIR's library to hide LLVM internals.
    target_link_libraries(migraphx_gpu PRIVATE rocMLIR::rockCompiler)
endif()

if(MIGRAPHX_USE_HIPRTC)
    message(STATUS "MIGraphX is using hipRTC")
    target_compile_definitions(migraphx_gpu PRIVATE -DMIGRAPHX_USE_HIPRTC=1)
else()
    message(STATUS "MIGraphX is using HIP Clang")

    # Get flags needed to compile hip
    include(TargetFlags)
    target_flags(HIP_COMPILER_FLAGS hip::device)

    # Remove cuda arch flags
    string(REGEX REPLACE --cuda-gpu-arch=[a-z0-9]+ "" HIP_COMPILER_FLAGS "${HIP_COMPILER_FLAGS}")
    string(REGEX REPLACE --offload-arch=[a-z0-9:+-]+ "" HIP_COMPILER_FLAGS "${HIP_COMPILER_FLAGS}")

    # Skip library paths since hip will incorrectly treat it as a source file
    string(APPEND HIP_COMPILER_FLAGS " ")

    foreach(_unused RANGE 2)
        string(REGEX REPLACE " /[^ ]+\\.(a|so) " " " HIP_COMPILER_FLAGS "${HIP_COMPILER_FLAGS}")
    endforeach()

    message(STATUS "Hip compiler flags: ${HIP_COMPILER_FLAGS}")
    target_compile_definitions(migraphx_gpu PRIVATE
        "-DMIGRAPHX_HIP_COMPILER=${CMAKE_CXX_COMPILER}"
        "-DMIGRAPHX_HIP_COMPILER_FLAGS=${HIP_COMPILER_FLAGS}"
    )

    if(DEFINED CMAKE_CXX_COMPILER_LAUNCHER)
        execute_process(COMMAND which ${CMAKE_CXX_COMPILER_LAUNCHER} OUTPUT_VARIABLE MIGRAPHX_HIP_COMPILER_LAUNCHER)
        string(STRIP "${MIGRAPHX_HIP_COMPILER_LAUNCHER}" MIGRAPHX_HIP_COMPILER_LAUNCHER)
        target_compile_definitions(migraphx_gpu PRIVATE "-DMIGRAPHX_HIP_COMPILER_LAUNCHER=${MIGRAPHX_HIP_COMPILER_LAUNCHER}")
    endif()
endif()

# Check miopen find mode api
include(CheckLibraryExists)
get_target_property(MIOPEN_LOCATION MIOpen LOCATION)
check_library_exists(MIOpen "miopenHiddenSetConvolutionFindMode" "${MIOPEN_LOCATION}" HAS_FIND_MODE_API)
check_library_exists(MIOpen "miopenFindSolutions" "${MIOPEN_LOCATION}" HAS_FIND_2_API)

set(MIGRAPHX_USE_FIND_2_API "${HAS_FIND_2_API}" CACHE BOOL "")

if(MIGRAPHX_USE_FIND_2_API)
    check_library_exists(MIOpen "miopenSetFindOptionPreallocatedTensor" "${MIOPEN_LOCATION}" HAS_PREALLOCATION_API)
    if(HAS_PREALLOCATION_API)
        target_compile_definitions(migraphx_gpu PUBLIC -DMIGRAPHX_HAS_FIND_2_API -DMIGRAPHX_PREALLOCATE_MIOPEN_BUFFERS)
    else()
        target_compile_definitions(migraphx_gpu PUBLIC -DMIGRAPHX_HAS_FIND_2_API)
    endif()
    message(STATUS "MIGraphx is using Find-2.0 API of MIOpen")
else()
    message(STATUS "MIGraphx is using legacy Find API in MIOpen")
endif()

if(HAS_FIND_MODE_API)
    target_compile_definitions(migraphx_gpu PUBLIC -DMIGRAPHX_HAS_FIND_MODE_API)
    message(STATUS "MIGraphx is using Find Mode API of MIOpen")
else()
    message(STATUS "MIOpen does not have find mode api")
endif()

target_link_libraries(migraphx_gpu PUBLIC migraphx MIOpen roc::rocblas)
target_link_libraries(migraphx_gpu PRIVATE migraphx_device migraphx_kernels)
if(NOT WIN32)
    # TODO: re-enable when CK is ported to Windows
    target_link_libraries(migraphx_gpu PRIVATE composable_kernel::jit_library)
endif()

add_subdirectory(driver)
add_subdirectory(hiprtc)

rocm_install_targets(
    TARGETS migraphx_gpu migraphx_device compile_for_gpu
    INCLUDE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
