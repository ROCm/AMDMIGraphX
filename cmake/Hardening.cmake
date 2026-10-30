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

include(CheckCCompilerFlag)
include(CheckCXXCompilerFlag)

# HIP offload compiles one invocation for the host and the device. A host-only
# flag such as -fcf-protection is an error on amdgcn. Clang accepts -Xarch_host
# so the flag stays on the host pass; GCC does not, and host GCC builds use the
# bare flag instead.
function(migraphx_add_hardening_option flag)
    string(MAKE_C_IDENTIFIER "${flag}" flag_id)
    set(host_flag "-Xarch_host ${flag}")

    check_cxx_compiler_flag("${host_flag}" MIGRAPHX_HARDEN_CXX_HOST_${flag_id})
    check_c_compiler_flag("${host_flag}" MIGRAPHX_HARDEN_C_HOST_${flag_id})
    if(MIGRAPHX_HARDEN_CXX_HOST_${flag_id} AND MIGRAPHX_HARDEN_C_HOST_${flag_id})
        separate_arguments(host_flag_list NATIVE_COMMAND "${host_flag}")
        add_compile_options(${host_flag_list})
        return()
    endif()

    check_cxx_compiler_flag("${flag}" MIGRAPHX_HARDEN_CXX_${flag_id})
    check_c_compiler_flag("${flag}" MIGRAPHX_HARDEN_C_${flag_id})
    if(MIGRAPHX_HARDEN_CXX_${flag_id} AND MIGRAPHX_HARDEN_C_${flag_id})
        add_compile_options("${flag}")
    elseif(MIGRAPHX_HARDEN_CXX_${flag_id})
        add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:${flag}>")
    elseif(MIGRAPHX_HARDEN_C_${flag_id})
        add_compile_options("$<$<COMPILE_LANGUAGE:C>:${flag}>")
    endif()
endfunction()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND MIGRAPHX_ENABLE_HARDENING)
    message(STATUS "Linux build hardening enabled")

    migraphx_add_hardening_option("-fstack-protector-strong")
    migraphx_add_hardening_option("-Wformat-security")
    migraphx_add_hardening_option("-fstack-clash-protection")

    # glibc warns that _FORTIFY_SOURCE needs optimization, and -Werror turns
    # that into a failed build for Debug and for an empty CMAKE_BUILD_TYPE.
    string(TOUPPER "${CMAKE_BUILD_TYPE}" migraphx_hardening_config)
    if(migraphx_hardening_config MATCHES "^(RELEASE|RELWITHDEBINFO|MINSIZEREL)$")
        migraphx_add_hardening_option("-D_FORTIFY_SOURCE=2")
    endif()

    check_cxx_compiler_flag("-Wl,-z,relro" MIGRAPHX_HAS_RELRO)
    check_cxx_compiler_flag("-Wl,-z,now" MIGRAPHX_HAS_BIND_NOW)
    if(MIGRAPHX_HAS_RELRO AND MIGRAPHX_HAS_BIND_NOW)
        add_link_options(-Wl,-z,relro -Wl,-z,now)
    endif()

    # -pie is for executables. Shared libraries are already position-independent.
    check_cxx_compiler_flag("-pie" MIGRAPHX_HAS_PIE)
    if(MIGRAPHX_HAS_PIE)
        string(APPEND CMAKE_EXE_LINKER_FLAGS " -pie")
    endif()

    migraphx_add_hardening_option("-fcf-protection=full")
endif()
