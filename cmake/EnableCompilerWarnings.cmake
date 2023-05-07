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
# - Enable warning all for gcc/clang or use /W4 for visual studio

## Strict warning level
if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(__default_cxx_compile_options /w)
else()
    set(__default_cxx_compile_options
            -Wall
            -Wextra
            -Wcomment
            -Wendif-labels
            -Wformat
            -Winit-self
            -Wreturn-type
            -Wsequence-point
            -Wswitch
            -Wtrigraphs
            -Wundef
            -Wuninitialized
            -Wunreachable-code
            -Wunused
            -Wno-sign-compare
            -Wno-reserved-macro-identifier)
    if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        list(APPEND __default_cxx_compile_options
                -Weverything
                -Wshadow
                -Wno-c++98-compat
                -Wno-c++98-compat-pedantic
                -Wno-conversion
                -Wno-double-promotion
                -Wno-exit-time-destructors
                -Wno-extra-semi
                -Wno-extra-semi-stmt
                -Wno-float-conversion
                -Wno-gnu-anonymous-struct
                -Wno-gnu-zero-variadic-macro-arguments
                -Wno-missing-prototypes
                -Wno-nested-anon-types
                -Wno-option-ignored
                -Wno-padded
                -Wno-shorten-64-to-32
                -Wno-sign-conversion
                -Wno-unused-command-line-argument
                -Wno-weak-vtables
                -Wno-c99-extensions)
        if(WIN32)
            list(APPEND __default_cxx_compile_options
                -fms-extensions
                -fms-compatibility
                -fdelayed-template-parsing)
        endif()
    endif()
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER "7.0")
        list(APPEND __default_cxx_compile_options
                -Wduplicated-branches
                -Wduplicated-cond
                -Wno-noexcept-type
                -Wodr
                -Wshift-negative-value
                -Wshift-overflow=2
                -Wno-missing-field-initializers
                -Wno-maybe-uninitialized)
    endif()
endif()

add_compile_options(${__default_cxx_compile_options})
unset(__default_cxx_compile_options)