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

rocm_setup_version(VERSION 0.5.1)
set(MIGRAPHX_SO_VERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR})

option( BUILD_SHARED_LIBS "Build as a shared library" ON )

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "5.4")
        message(FATAL_ERROR "MIGraph requires at least gcc 5.4")
    endif()
endif()

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("--cuda-host-only -x hip" HAS_HIP)
if(HAS_HIP)
    message(STATUS "Enable miopen backend")
    set(MIGRAPHX_ENABLE_GPU On CACHE BOOL "")
else()
    set(MIGRAPHX_ENABLE_GPU Off CACHE BOOL "")
endif()

add_compile_options(-std=c++14)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(EnableCompilerWarnings)
include(ROCMClangTidy)
if(CMAKE_CXX_COMPILER MATCHES ".*hcc")
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
        modernize-*
        performance-*
        readability-*
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
        -google-readability-*
        -google-runtime-int
        -google-runtime-references
        -misc-macro-parentheses
        -modernize-concat-nested-namespaces
        -modernize-pass-by-value
        -modernize-use-default-member-init
        -modernize-use-nodiscard
        -modernize-use-override
        -modernize-use-trailing-return-type
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
    ${MIGRAPHX_TIDY_ERRORS}
    HEADER_FILTER
        ".*hpp"
    EXTRA_ARGS
        -UNDEBUG
        -DMIGRAPHX_USE_CLANG_TIDY
        "-Dmain\\\\(...\\\\)=main\\\\(__VA_ARGS__\\\\) // NOLINT"
    # CLANG_ARGS
    #     -analyzer-config optin.cplusplus.UninitializedObject:Pedantic=true
    #     -analyzer-config widen-loops=true
    #     -analyzer-config unroll-loops=true
    #     -analyzer-config cfg-lifetime=true
    #     -analyzer-config cfg-scopes=true
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
        shadowVariable
        unsafeClassDivZero
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
        __device__=
        __host__=
)

enable_testing()

include(ROCMCreatePackage)
rocm_create_package(
    NAME MIGraphX
    DESCRIPTION "AMD's graph optimizer"
    MAINTAINER "Paul Fultz II <paul.fultz@amd.com>"
    LDCONFIG
    PTH
    DEPENDS miopen-hip rocblas hip-hcc half
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)

add_subdirectory(src)
add_subdirectory(doc)
add_subdirectory(test)
add_subdirectory(tools)
