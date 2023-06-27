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

if(COMMAND rocm_enable_clang_tidy)
    return()
endif()

include(ROCMClangTidy)

include(CMakeDependentOption)
cmake_dependent_option(MIGRAPHX_ENABLE_CLANG_TIDY "Enable Clang-Tidy checks" ON CLANG_TIDY_EXE OFF)

macro(rocm_enable_clang_tidy)
    if(MIGRAPHX_ENABLE_CLANG_TIDY)
        _rocm_enable_clang_tidy(${ARGN})
    endif()
endmacro()

macro(rocm_clang_tidy_check)
    if(MIGRAPHX_ENABLE_CLANG_TIDY)
        _rocm_clang_tidy_check(${ARGN})
    endif()
endmacro()
