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
cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
    message(FATAL_ERROR "The binary and source directroy cannot be the same")
endif()

get_property(_GENERATOR_IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)

# This has to be initialized before the project() command appears
# Set the default of CMAKE_BUILD_TYPE to be release, unless user specifies with -D.  MSVC_IDE does not use CMAKE_BUILD_TYPE
if(_GENERATOR_IS_MULTI_CONFIG)
    if (NOT CMAKE_CONFIGURATION_TYPES)
        set(CMAKE_CONFIGURATION_TYPES "Debug;Release;RelWithDebInfo;MinSizeRel" CACHE STRING
            "Available build types (configurations) on multi-config generators")
    endif()
else()
    if(NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE Release CACHE STRING
            "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel.")
    endif()
endif()

set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "")

set(CMAKE_BUILD_RPATH "${CMAKE_BINARY_DIR}/lib")

project(migraphx LANGUAGES C CXX)
include(CTest)

find_package(ROCM REQUIRED)
find_package(Threads REQUIRED)

find_path(HALF_INCLUDE_DIR half.hpp PATH_SUFFIXES half)
if (NOT HALF_INCLUDE_DIR)
    message(FATAL_ERROR "Could not find half.hpp - Please check that the install path of half.hpp has been added to CMAKE_PREFIX_PATH")
else()
	message(STATUS "half.hpp is at ${HALF_INCLUDE_DIR}")
endif()

include(CheckTypeSize)
set(CMAKE_REQUIRED_INCLUDES ${HALF_INCLUDE_DIR})
set(CMAKE_EXTRA_INCLUDE_FILES half.hpp)
check_type_size("half_float::detail::expr" HALF_EXPR LANGUAGE CXX)
set(CMAKE_REQUIRED_INCLUDES)
set(CMAKE_EXTRA_INCLUDE_FILES)

include(ROCMSetupVersion)

option(BUILD_DEV "Build for development purpose only" OFF)

rocm_setup_version(VERSION 2.7.0)
set(MIGRAPHX_SO_VERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH})

option( BUILD_SHARED_LIBS "Build as a shared library" ON )

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("--cuda-host-only -x hip" HAS_HIP)
if(HAS_HIP)
    message(STATUS "Enable gpu backend")
    set(MIGRAPHX_ENABLE_GPU On CACHE BOOL "")
else()
    set(MIGRAPHX_ENABLE_GPU Off CACHE BOOL "")
endif()

# Disable cpu backend by default
set(MIGRAPHX_ENABLE_CPU Off CACHE BOOL "")

# Disable fpga backend by default
set(MIGRAPHX_ENABLE_FPGA Off CACHE BOOL "")

set(CMAKE_CXX_STANDARD_DEFAULT "")
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-std=c++17>)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(EnableCompilerWarnings)
include(ROCMClangTidy)
if(CMAKE_CXX_COMPILER MATCHES ".*clang\\+\\+")
    set(MIGRAPHX_TIDY_ERRORS ERRORS * -readability-inconsistent-declaration-parameter-name)
# Enable tidy on hip
elseif(MIGRAPHX_ENABLE_GPU)
    set(MIGRAPHX_TIDY_ERRORS ALL)
endif()
rocm_enable_clang_tidy(
    CHECKS
        boost-*
        bugprone-*
        cert-*
        clang-analyzer-*
        clang-diagnostic-*
        cppcoreguidelines-*
        google-*
        hicpp-multiway-paths-covered
        hicpp-signed-bitwise
        llvm-namespace-comment
        misc-*
	-misc-confusable-identifiers
        -misc-use-anonymous-namespace
        modernize-*
        performance-*
        readability-*
        -bugprone-easily-swappable-parameters
        -bugprone-implicit-widening-of-multiplication-result
        -bugprone-macro-parentheses
        -bugprone-signed-char-misuse
        -bugprone-unchecked-optional-access
        # Disable the aliased reserved identifiers
        -cert-dcl37-c
        -cert-dcl51-cpp
        -cert-err33-c
        -cert-str34-c
        # Disable all alpha checks by default
        -clang-analyzer-alpha*
        # Enable some alpha checks
        clang-analyzer-alpha.core.CallAndMessageUnInitRefArg
        clang-analyzer-alpha.core.Conversion
        clang-analyzer-alpha.core.IdenticalExpr
        clang-analyzer-alpha.core.PointerArithm
        clang-analyzer-alpha.core.PointerSub
        clang-analyzer-alpha.core.TestAfterDivZero
        clang-analyzer-alpha.cplusplus.InvalidIterator
        clang-analyzer-alpha.cplusplus.IteratorRange
        clang-analyzer-alpha.cplusplus.MismatchedIterator
        clang-analyzer-alpha.cplusplus.MisusedMovedObject
        -clang-analyzer-optin.performance.Padding
        -clang-diagnostic-deprecated-declarations
        -clang-diagnostic-extern-c-compat
        -clang-diagnostic-disabled-macro-expansion
        -clang-diagnostic-unused-command-line-argument
        -cppcoreguidelines-avoid-do-while
        -cppcoreguidelines-avoid-const-or-ref-data-members
        -cppcoreguidelines-explicit-virtual-functions
        -cppcoreguidelines-init-variables
        -cppcoreguidelines-pro-bounds-array-to-pointer-decay
        -cppcoreguidelines-pro-bounds-constant-array-index
        -cppcoreguidelines-pro-bounds-pointer-arithmetic
        -cppcoreguidelines-pro-type-member-init
        -cppcoreguidelines-pro-type-reinterpret-cast
        -cppcoreguidelines-pro-type-union-access
        -cppcoreguidelines-pro-type-vararg
        -cppcoreguidelines-special-member-functions
        -cppcoreguidelines-virtual-class-destructor
        -google-readability-*
        -google-runtime-int
        -google-runtime-references
        -misc-macro-parentheses
        -misc-no-recursion
        -modernize-concat-nested-namespaces
        -modernize-pass-by-value
        -modernize-use-default-member-init
        -modernize-use-nodiscard
        -modernize-use-override
        -modernize-use-trailing-return-type
        -modernize-use-transparent-functors
        -performance-type-promotion-in-math-fn
        -readability-braces-around-statements
        -readability-convert-member-functions-to-static
        -readability-else-after-return
        -readability-function-cognitive-complexity
        -readability-identifier-length
        -readability-named-parameter
        -readability-redundant-string-init
        -readability-suspicious-call-argument
        -readability-uppercase-literal-suffix
        -*-avoid-c-arrays
        -*-explicit-constructor
        -*-magic-numbers
        -*-narrowing-conversions
        -*-non-private-member-variables-in-classes
        -*-use-auto
        -*-use-emplace
        -*-use-equals-default
    ${MIGRAPHX_TIDY_ERRORS}
    HEADER_FILTER
        ".*hpp"
    EXTRA_ARGS
        -UNDEBUG
        -DMIGRAPHX_USE_CLANG_TIDY
    CLANG_ARGS
        -analyzer-max-loop 10
        -analyzer-inline-max-stack-depth 10
        -analyzer-config optin.cplusplus.UninitializedObject:Pedantic=true
        -analyzer-config widen-loops=true
        -analyzer-config unroll-loops=true
        -analyzer-config cfg-lifetime=true
        -analyzer-config cfg-scopes=true
)
include(ROCMCppCheck)
rocm_enable_cppcheck(
    CHECKS
        warning
        style
        performance
        portability
    SUPPRESS
        ConfigurationNotChecked
        unmatchedSuppression
        unusedFunction
        ctuPointerArith
        noExplicitConstructor
        passedByValue
        unusedStructMember
        functionStatic
        functionConst
        shadowFunction
        shadowVar
        shadowVariable
        unsafeClassDivZero
        # Disable because of too many FPs
        arithOperationsOnVoidPointer
        definePrefix:*test/include/test.hpp
        ctuOneDefinitionRuleViolation:*test/*
        useSmartPointer:*src/api/api.cpp
        useSmartPointer:*make_shared_array.hpp
    FORCE
    INCONCLUSIVE
    RULE_FILE
        ${CMAKE_CURRENT_SOURCE_DIR}/cppcheck.rules
    SOURCES
        examples/
        src/
        test/
    INCLUDE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src/targets/cpu/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src/targets/gpu/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src/targets/gpu/device/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src/targets/gpu/kernels/include
        ${CMAKE_CURRENT_SOURCE_DIR}/test/include
    DEFINE
        MIGRAPHX_MLIR=1
        CPPCHECK=1
        __device__=
        __host__=
        __global__=
    UNDEFINE
        MIGRAPHX_USE_CLANG_TIDY
)

enable_testing()

include(ROCMCreatePackage)
include(ROCMTest)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)
add_subdirectory(src)
add_subdirectory(docs)
if(BUILD_TESTING)
    rocm_enable_test_package(migraphx)
    add_subdirectory(test)
endif()
add_subdirectory(tools)

set(DEST_DIR ${CMAKE_BINARY_DIR})
file(GLOB backend_files ${CMAKE_SOURCE_DIR}/src/py/backend/*.py)
file(MAKE_DIRECTORY ${DEST_DIR}/lib/onnx_migraphx)
foreach(py_file ${backend_files})
    configure_file(${py_file} ${DEST_DIR}/lib/onnx_migraphx/. COPYONLY)
endforeach(py_file)
configure_file(${CMAKE_SOURCE_DIR}/test/py/onnx_backend_test.py ${DEST_DIR}/onnx_backend_test.py COPYONLY)

rocm_create_package(
    NAME MIGraphX
    DESCRIPTION "AMD's graph optimizer"
    MAINTAINER "AMDMIGraphX Maintainer <migraphx-lib.support@amd.com>"
    LDCONFIG
    PTH
    DEPENDS miopen-hip rocblas hip-rocclr hip-base half
)
