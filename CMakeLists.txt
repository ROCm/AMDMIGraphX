cmake_minimum_required(VERSION 3.5)

project(rtglib)
find_package(ROCM REQUIRED)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "5.4")
        message(FATAL_ERROR "RTGLib requires at least gcc 5.4")
    endif()
endif()

add_compile_options(-std=c++14)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(EnableCompilerWarnings)
# Override clang-tidy to not find the version from hcc
find_program(CLANG_TIDY_EXE 
    NAMES 
        clang-tidy
        clang-tidy-5.0
        clang-tidy-6.0
        clang-tidy-7.0
    PATHS
        /usr/local/opt/llvm/bin
)
include(ROCMClangTidy)
rocm_enable_clang_tidy(
    CHECKS
        *
        -cert-env33-c
        -android-cloexec-fopen
        -cert-msc50-cpp
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
        -llvm-header-guard
        -llvm-include-order
        -misc-macro-parentheses
        -modernize-use-auto
        -modernize-pass-by-value
        -modernize-use-default-member-init
        -modernize-use-transparent-functors
        -performance-unnecessary-value-param
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
        -DRTG_USE_CLANG_TIDY
    ANALYZE_TEMPORARY_DTORS ON

)
include(ROCMCppCheck)
rocm_enable_cppcheck(
    CHECKS 
        all
    SUPPRESS 
        ConfigurationNotChecked
        unmatchedSuppression
        unusedFunction
        noExplicitConstructor
        passedByValue
        unusedStructMember
    FORCE
    SOURCES
        include/
        src/
        test/
    INCLUDE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_subdirectory(src)
add_subdirectory(doc)
add_subdirectory(test)
add_subdirectory(tools)
