#####################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#
#####################################################################################
cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
    message(FATAL_ERROR "The binary and source directroy cannot be the same")
endif()

# Setup valid strings for build type
if (NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_CONFIGURATION_TYPES "Debug;Release;RelWithDebInfo;MinSizeRel" CACHE STRING "Configs")
endif()

get_property(ROCM_ENERATOR_IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
# This has to be initialized before the project() command appears
# Set the default of CMAKE_BUILD_TYPE to be release, unless user specifies with -D.  MSVC_IDE does not use CMAKE_BUILD_TYPE
if(NOT ROCM_ENERATOR_IS_MULTI_CONFIG)
    set(CMAKE_BUILD_TYPE Release CACHE STRING
        "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel.")
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS ${CMAKE_CONFIGURATION_TYPES})
endif()

if(NOT WIN32)
    set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "")
    set(CMAKE_BUILD_RPATH "${CMAKE_BINARY_DIR}/lib")
endif()

list(APPEND CMAKE_PREFIX_PATH /opt/rocm /opt/rocm/llvm $ENV{ROCM_PATH} $ENV{HIP_PATH})

project(tiny-rocm-cxx LANGUAGES C CXX)
include(CTest)

find_package(ROCM REQUIRED)
find_package(Threads REQUIRED)
include(ROCMSetupVersion)

include(ROCMClangTidy)
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
        -cppcoreguidelines-macro-usage
        -cppcoreguidelines-pro-bounds-array-to-pointer-decay
        -cppcoreguidelines-pro-bounds-constant-array-index
        -cppcoreguidelines-pro-bounds-pointer-arithmetic
        -cppcoreguidelines-pro-type-member-init
        -cppcoreguidelines-pro-type-reinterpret-cast
        -cppcoreguidelines-pro-type-union-access
        -cppcoreguidelines-pro-type-vararg
        -cppcoreguidelines-special-member-functions
        -cppcoreguidelines-virtual-class-destructor
        -cppcoreguidelines-avoid-capture-default-when-capturing-this
        -cppcoreguidelines-rvalue-reference-param-not-moved
        -google-readability-*
        -google-runtime-int
        -google-runtime-references
        -hicpp-signed-bitwise
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
    ERRORS
        *
    HEADER_FILTER
        ".*hpp"
    EXTRA_ARGS
        -UNDEBUG
        -DROCM_USE_CLANG_TIDY
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
    FORCE
    INCONCLUSIVE
    SOURCES
        examples/
        src/
        test/
    INCLUDE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include
        ${CMAKE_CURRENT_SOURCE_DIR}/test/include
    DEFINE
        CPPCHECK=1
        __device__=
        __host__=
        __global__=
    UNDEFINE
        ROCM_USE_CLANG_TIDY
)

include(ROCMCreatePackage)
include(ROCMTest)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)

add_subdirectory(src)
if(BUILD_TESTING)
    rocm_enable_test_package(tiny-rocm-cxx)
    add_subdirectory(test)
endif()

rocm_create_package(
    NAME tiny-rocm-cxx
    DESCRIPTION "Small C++ library for runtime compliation"
    MAINTAINER "AMDMIGraphX Maintainer <migraphx-lib.support@amd.com>"
    LDCONFIG
    PTH
)
