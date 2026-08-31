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
# Linux exploit-mitigation compiler and linker flags (SEC-00802 / SEC-00469)

include(CheckCXXCompilerFlag)

if(NOT WIN32 AND NOT APPLE AND MIGRAPHX_ENABLE_HARDENING)
    message(STATUS "Linux build hardening enabled")
    add_compile_options(
        -fstack-protector-strong
        -Wformat-security
        -fstack-clash-protection
    )
    add_compile_definitions(_FORTIFY_SOURCE=2)
    add_link_options(-Wl,-z,relro -Wl,-z,now -pie)

    check_cxx_compiler_flag("-fcf-protection=full" MIGRAPHX_HAS_FCF_PROTECTION)
    if(MIGRAPHX_HAS_FCF_PROTECTION)
        add_compile_options(-fcf-protection=full)
    endif()
endif()
