# ####################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
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

# Default locations of the two rocMLIR installs (rocmlir-config.cmake dirs).
# Override on the command line if installed elsewhere.
set(MIGRAPHX_MLIR_LEGACY_ROCMLIR_DIR "C:/opt/rocmlir/WinRelease/lib/cmake/rocmlir"
    CACHE PATH "rocMLIR config dir for the legacy backend plugin")
set(MIGRAPHX_MLIR_TRITON_ROCMLIR_DIR "C:/opt/rocmlirTriton/WinRelease/lib/cmake/rocmlir"
    CACHE PATH "rocMLIR config dir for the Triton backend plugin")

# Builds one MLIR backend plugin DLL (migraphx_mlir_<BACKEND_NAME>) that links
# the rocMLIR install whose config lives at ROCMLIR_CONFIG_DIR. Each invocation
# must happen in its own directory scope (separate add_subdirectory) so that the
# rocMLIR::rockCompiler imported target is created freshly per install and does
# not clash between the two backends.
function(add_mlir_backend_plugin BACKEND_NAME ROCMLIR_CONFIG_DIR)
    set(_plugin migraphx_mlir_${BACKEND_NAME})

    # Force find_package(rocMLIR) to use this specific install.
    set(rocMLIR_DIR "${ROCMLIR_CONFIG_DIR}")
    find_package(rocMLIR 1.0.0 CONFIG REQUIRED)
    message(STATUS "MLIR backend plugin '${BACKEND_NAME}' using rocMLIR from ${ROCMLIR_CONFIG_DIR}")

    add_library(${_plugin} MODULE
        ${MIGRAPHX_GPU_DIR}/mlir.cpp
        ${MIGRAPHX_GPU_DIR}/prepare_mlir.cpp
        ${MIGRAPHX_GPU_DIR}/hsa_chiplet.cpp
        ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/plugin_helpers.cpp
        ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/plugin_export.cpp)

    target_include_directories(${_plugin} PRIVATE
        ${MIGRAPHX_GPU_DIR}/include
        ${MIGRAPHX_GPU_BINARY_DIR}/include)

    # Mirror migraphx_gpu's public compile definitions (MIGRAPHX_USE_*), which
    # determine the ABI/layout of gpu::context, and mark MIGRAPHX_GPU_EXPORT as
    # dllexport so the implementation in mlir.cpp is defined, not imported.
    target_compile_definitions(${_plugin} PRIVATE
        migraphx_gpu_EXPORTS
        MIGRAPHX_MLIR
        $<TARGET_PROPERTY:migraphx_gpu,INTERFACE_COMPILE_DEFINITIONS>)

    # Link migraphx core (shared, so types match the host) + this rocMLIR.
    # Note: deliberately NOT linking migraphx_gpu to avoid duplicate definitions
    # of the public mlir.hpp symbols.
    target_link_libraries(${_plugin} PRIVATE
        migraphx
        hip::host
        rocMLIR::rockCompiler)
    if(WIN32)
        # rocMLIR pulls in symbols that resolve against ntdll on Windows.
        target_link_libraries(${_plugin} PRIVATE ntdll)
    endif()

    # Place the plugin next to migraphx_gpu so the loader finds it. Keep the
    # default library prefix (empty on Windows, "lib" on Linux) to match the
    # name the loader in mlir_backend_loader.cpp computes.
    set_target_properties(${_plugin} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY $<TARGET_FILE_DIR:migraphx_gpu>
        LIBRARY_OUTPUT_DIRECTORY $<TARGET_FILE_DIR:migraphx_gpu>)

    add_dependencies(${_plugin} migraphx_gpu)
endfunction()

add_subdirectory(legacy)
add_subdirectory(triton)
