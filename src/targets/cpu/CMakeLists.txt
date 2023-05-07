#####################################################################################
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
#####################################################################################

include(CheckCXXCompilerFlag)
include(GenerateExportHeader)

add_library(migraphx_cpu
    allocate.cpp
    allocation_model.cpp
    binary.cpp
    concat.cpp
    convolution.cpp
    copy.cpp
    deconvolution.cpp
    dnnl.cpp
    eltwise.cpp
    erf.cpp
    fmod.cpp
    fuse_ops.cpp
    gather.cpp
    gemm.cpp
    layernorm.cpp
    logsoftmax.cpp
    lowering.cpp
    lrn.cpp
    mod.cpp
    preallocate.cpp
    pooling.cpp
    reduction.cpp
    reorder.cpp
    softmax.cpp
    sub.cpp
    target.cpp
    write_literals.cpp
)

generate_export_header(migraphx_cpu
    EXPORT_FILE_NAME
        ${CMAKE_BINARY_DIR}/include/migraphx/cpu/export.h)

if(WIN32)
    target_compile_options(migraphx_cpu
        PRIVATE
            -fdelayed-template-parsing        
            -fms-extensions
            -fms-compatibility)
endif()        

set_target_properties(migraphx_cpu PROPERTIES EXPORT_NAME cpu)

if(NOT WIN32)
    rocm_set_soversion(migraphx_cpu ${MIGRAPHX_SO_VERSION})
endif()

set(MIGRAPHX_ENABLE_ZENDNN Off CACHE BOOL "")
target_include_directories(migraphx_cpu PRIVATE ${CMAKE_BINARY_DIR}/include)

if(WIN32)
    if(MIGRAPHX_ENABLE_ZENDNN)
        target_link_libraries(migraphx_cpu PRIVATE amdZenDNN)
    else()
        target_link_libraries(migraphx_cpu PRIVATE dnnl)
    endif()
else()
    if(MIGRAPHX_ENABLE_ZENDNN)
        find_path(ZENDNN_INC_PATH zendnn.hpp)
        find_library(ZENDNN_LIB amdZenDNN)
        find_library(BLIS_LIB blis)
        target_include_directories(migraphx_cpu PRIVATE ${ZENDNN_INC_PATH})
        message(STATUS "ZENDNN_LIB: ${ZENDNN_LIB}")
        target_link_libraries(migraphx_cpu PRIVATE ${BLIS_LIB})
        target_link_libraries(migraphx_cpu PRIVATE ${ZENDNN_LIB})
    else()
        find_package(dnnl REQUIRED)
        target_link_libraries(migraphx_cpu PRIVATE DNNL::dnnl)
    endif()
endif()

if(CLANG_TIDY_EXE)
    rocm_clang_tidy_check(migraphx_cpu)
endif()

if(MIGRAPHX_ENABLE_ZENDNN)
    target_compile_definitions(migraphx_cpu PRIVATE -DMIGRAPHX_ENABLE_ZENDNN)
endif()

target_link_libraries(migraphx_cpu PRIVATE migraphx)

find_package(OpenMP REQUIRED COMPONENTS CXX)
if(WIN32)
    target_link_libraries(migraphx_cpu PRIVATE libomp)
    target_include_directories(migraphx_cpu PRIVATE ${OpenMP_CXX_INCLUDE_DIRS})
    target_compile_options(migraphx_cpu PRIVATE ${OpenMP_CXX_FLAGS})
else()
    target_link_libraries(migraphx_cpu PRIVATE OpenMP::OpenMP_CXX)
    # Add library path to rpath to workaround issues with our broken packages
    foreach(LIBRARY ${OpenMP_CXX_LIBRARIES})
        if(LIBRARY MATCHES "libomp")
            get_filename_component(LIBRARY_PATH "${LIBRARY}" PATH)
            target_link_libraries(migraphx_cpu PUBLIC -Wl,-rpath=${LIBRARY_PATH} -Wl,-rpath-link=${LIBRARY_PATH})
        endif()
    endforeach()
endif()

rocm_install_targets(
  TARGETS migraphx_cpu
  INCLUDE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
