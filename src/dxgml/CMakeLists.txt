#####################################################################################
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
#####################################################################################

file(GLOB DXGML_SRCS CONFIGURE_DEPENDS *.cpp)
add_library(migraphx_dxgml ${DXGML_SRCS})
set_target_properties(migraphx_dxgml PROPERTIES EXPORT_NAME dxgml)
migraphx_generate_export_header(migraphx_dxgml)
rocm_set_soversion(migraphx_dxgml ${MIGRAPHX_SO_VERSION})
rocm_clang_tidy_check(migraphx_dxgml)

target_link_libraries(migraphx_dxgml PUBLIC migraphx)

# We parse DxGML MLIR text using the MLIR C API from rocMLIR.
# When DXGML_DROP_DIR is set, we also link dxgml.lib and register the DxGML
# dialects via the typed C++ API (Option 2), enabling typed attribute/type access.
find_package(rocMLIR 1.0.0 REQUIRED CONFIG)

# DxGML drop directory — provides dxgml.lib + Source/include headers
set(DXGML_DROP_DIR "$ENV{DXGML_DROP_DIR}" CACHE PATH
    "Path to DXGML-Drop4.1-x64 directory")

if(DXGML_DROP_DIR)
    message(STATUS "DxGML: DXGML_DROP_DIR = ${DXGML_DROP_DIR}")
    # DxGML C++ headers (dxgml/dxgml_dialect.h, dxgml/dxgml_types.h, etc.)
    target_include_directories(migraphx_dxgml SYSTEM PRIVATE "${DXGML_DROP_DIR}/Source/include")
    # Generated TableGen headers already in build dir
    target_include_directories(migraphx_dxgml PRIVATE
        "${CMAKE_BINARY_DIR}/dxgml_generated")
    # Link dxgml.lib (import library for dxgml.dll — 12 MB, contains DxGMLDialect)
    target_link_libraries(migraphx_dxgml PRIVATE "${DXGML_DROP_DIR}/lib/dxgml.lib")
    target_compile_definitions(migraphx_dxgml PRIVATE MIGRAPHX_DXGML_HAS_IR_LIB)
else()
    message(STATUS "DxGML: DXGML_DROP_DIR not set — using allowUnregisteredDialects fallback")
endif()

# Get include directories from rockCompiler target (contains mlir-c/ and mlir/CAPI/ headers)
get_target_property(_rocmlir_incs rocMLIR::rockCompiler INTERFACE_INCLUDE_DIRECTORIES)
target_include_directories(migraphx_dxgml SYSTEM PRIVATE ${_rocmlir_incs})

# Resolve the rocMLIR versioned install prefix from one of the known include dirs.
# The first include dir is typically: <prefix>/include/rocmlir
# and the second:                    <prefix>/include/rocmlir/external/include
# We strip 2 components from the first (rocmlir, include) to get <prefix>.
list(GET _rocmlir_incs 0 _first_inc)
get_filename_component(_p1 "${_first_inc}" DIRECTORY)  # strip /rocmlir
get_filename_component(_rocmlir_prefix "${_p1}" DIRECTORY) # strip /include

message(STATUS "DxGML: rocMLIR prefix = ${_rocmlir_prefix}")

# Link only the minimal MLIR C API static libs needed for parsing.
# These provide: mlirContextCreate, mlirModuleCreateParse, mlirAttributePrint, etc.
foreach(_lib
    MLIRCAPIIR
    MLIRIR
    MLIRAsmParser
    MLIRBytecodeReader
    MLIRBytecodeWriter
    MLIRParser
    MLIRSupport
    MLIRBytecodeOpInterface
    LLVMCore
    LLVMSupport
)
    find_library(_found_${_lib} NAMES "${_lib}"
        PATHS
            "${_rocmlir_prefix}/lib"
            "${_rocmlir_prefix}/lib/llvm"
        NO_DEFAULT_PATH
        REQUIRED)
    message(STATUS "DxGML: found ${_lib} = ${_found_${_lib}}")
    target_link_libraries(migraphx_dxgml PRIVATE "${_found_${_lib}}")
endforeach()

# On Windows, LLVMSupport needs ntdll for RtlGetLastNtStatus etc.
if(WIN32)
    target_link_libraries(migraphx_dxgml PRIVATE ntdll.lib)
endif()

rocm_install_targets(
  PRIVATE
  TARGETS migraphx_dxgml
)
