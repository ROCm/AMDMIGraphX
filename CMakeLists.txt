cmake_minimum_required(VERSION 3.5)

project(migraphlib)
find_package(ROCM REQUIRED)

option( BUILD_SHARED_LIBS "Build as a shared library" ON )

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "5.4")
        message(FATAL_ERROR "MIGraph requires at least gcc 5.4")
    endif()
endif()

if(CMAKE_CXX_COMPILER MATCHES ".*hcc")
    message(STATUS "Enable miopen backend")
    set(MIGRAPH_ENABLE_GPU On CACHE BOOL "")
else()
    set(MIGRAPH_ENABLE_GPU Off CACHE BOOL "")
endif()

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
        -clang-diagnostic-unused-command-line-argument
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
        -hicpp-use-override
        # This check is broken
        -hicpp-use-auto
        -llvm-header-guard
        -llvm-include-order
        -misc-macro-parentheses
        -modernize-use-auto
        -modernize-use-override
        -modernize-pass-by-value
        -modernize-use-default-member-init
        -modernize-use-transparent-functors
        -readability-braces-around-statements
        -readability-else-after-return
        -readability-named-parameter
        -*-explicit-constructor
        -*-use-emplace
        -*-use-equals-default
    ERRORS
        *
        -readability-inconsistent-declaration-parameter-name
    HEADER_FILTER
        ".*hpp"
    EXTRA_ARGS
        -DMIGRAPH_USE_CLANG_TIDY

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
        definePrefix:*test/include/test.hpp
    FORCE
    RULE_FILE
        ${CMAKE_CURRENT_SOURCE_DIR}/cppcheck.rules
    SOURCES
        src/
        test/
    INCLUDE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src/targets/cpu/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src/targets/miopen/include
    DEFINE
        CPPCHECK=1
)

enable_testing()

add_subdirectory(src)
add_subdirectory(doc)
add_subdirectory(test)
add_subdirectory(tools)
