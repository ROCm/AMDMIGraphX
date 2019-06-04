cmake_minimum_required(VERSION 3.5)

if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
    message(FATAL_ERROR "The binary and source directroy cannot be the same")
endif()

# This has to be initialized before the project() command appears
# Set the default of CMAKE_BUILD_TYPE to be release, unless user specifies with -D.  MSVC_IDE does not use CMAKE_BUILD_TYPE
if( NOT MSVC_IDE AND NOT CMAKE_BUILD_TYPE )
    set( CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." )
endif()

# Default installation path
if(WIN32)
    set(CMAKE_INSTALL_PREFIX "/opt/rocm/x86_64-w64-mingw32" CACHE PATH "")
else()
    set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "")
endif()

project(migraphx)
find_package(ROCM REQUIRED)

include(ROCMSetupVersion)

rocm_setup_version(VERSION 0.2)

option( BUILD_SHARED_LIBS "Build as a shared library" ON )

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "5.4")
        message(FATAL_ERROR "MIGraph requires at least gcc 5.4")
    endif()
endif()

if(CMAKE_CXX_COMPILER MATCHES ".*hcc")
    message(STATUS "Enable miopen backend")
    set(MIGRAPHX_ENABLE_GPU On CACHE BOOL "")
else()
    set(MIGRAPHX_ENABLE_GPU Off CACHE BOOL "")
endif()

set(MIGRAPHX_ENABLE_TF Off CACHE BOOL "")

add_compile_options(-std=c++14)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(EnableCompilerWarnings)
include(ROCMClangTidy)
rocm_enable_clang_tidy(
    CHECKS
        *
        -android-cloexec-fopen
        -clang-analyzer-alpha.core.CastToStruct
        -clang-analyzer-optin.performance.Padding
        -clang-diagnostic-deprecated-declarations
        -clang-diagnostic-extern-c-compat
        -clang-diagnostic-disabled-macro-expansion
        -clang-diagnostic-unused-command-line-argument
        -cppcoreguidelines-explicit-virtual-functions
        -cppcoreguidelines-pro-bounds-array-to-pointer-decay
        -cppcoreguidelines-pro-bounds-constant-array-index
        -cppcoreguidelines-pro-bounds-pointer-arithmetic
        -cppcoreguidelines-pro-type-member-init
        -cppcoreguidelines-pro-type-reinterpret-cast
        -cppcoreguidelines-pro-type-union-access
        -cppcoreguidelines-pro-type-vararg
        -cppcoreguidelines-special-member-functions
        -fuchsia-*
        -google-readability-braces-around-statements
        -google-readability-todo
        -google-runtime-int
        -google-runtime-references
        -hicpp-braces-around-statements
        -hicpp-explicit-conversions
        -hicpp-no-array-decay
        -hicpp-special-member-functions
        -hicpp-uppercase-literal-suffix
        -hicpp-use-override
        # This check is broken
        -llvm-header-guard
        -llvm-include-order
        -misc-macro-parentheses
        -modernize-use-override
        -modernize-pass-by-value
        -modernize-use-default-member-init
        -modernize-use-transparent-functors
        -performance-type-promotion-in-math-fn
        -readability-braces-around-statements
        -readability-else-after-return
        -readability-named-parameter
        -readability-uppercase-literal-suffix,
        -*-avoid-c-arrays
        -*-explicit-constructor
        -*-magic-numbers
        -*-non-private-member-variables-in-classes
        -*-use-auto
        -*-use-emplace
        -*-use-equals-default
    ERRORS
        *
        -readability-inconsistent-declaration-parameter-name
    HEADER_FILTER
        ".*hpp"
    EXTRA_ARGS
        -DMIGRAPHX_USE_CLANG_TIDY
        "-Dmain\\\\(...\\\\)=main\\\\(__VA_ARGS__\\\\) // NOLINT"

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
        noExplicitConstructor
        passedByValue
        unusedStructMember
        functionStatic
        functionConst:*program.*
        shadowFunction
        shadowVar
        definePrefix:*test/include/test.hpp
    FORCE
    INCONCLUSIVE
    RULE_FILE
        ${CMAKE_CURRENT_SOURCE_DIR}/cppcheck.rules
    SOURCES
        src/
        test/
    INCLUDE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src/targets/cpu/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src/targets/miopen/include
        ${CMAKE_CURRENT_SOURCE_DIR}/test/include
    DEFINE
        CPPCHECK=1
)

enable_testing()

include(ROCMCreatePackage)
rocm_create_package(
    NAME MIGraphX
    DESCRIPTION "AMD's graph optimizer"
    MAINTAINER "Paul Fultz II <paul.fultz@amd.com>"
    LDCONFIG
    PTH
    DEPENDS miopen-hip rocblas hip_hcc half
)

add_subdirectory(src)
add_subdirectory(doc)
add_subdirectory(test)
add_subdirectory(tools)
