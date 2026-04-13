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

# Detect the packaging backend for MIGraphX.
#
# detect_package_backend() sets MIGRAPHX_PACKAGE_BACKEND as a cache variable:
#   "therock"  - TheRock environment (amdrocm-xxx deb/rpm packages)
#   "default"  - Traditional ROCm with deb/rpm packages
#
# Preferred usage (explicit):
#   cmake -DMIGRAPHX_PACKAGE_BACKEND=therock -DMIGRAPHX_THEROCK_GPU_ARCH=gfx120x ..
#
# If MIGRAPHX_PACKAGE_BACKEND is not set, falls back to auto-detection via
# dpkg/rpm to check for installed amdrocm-runtime packages.
#
# When MIGRAPHX_PACKAGE_BACKEND=therock, MIGRAPHX_THEROCK_GPU_ARCH must be set
# to the target GPU architecture family that follows TheRock packaging requirements.

function(_detect_therock_via_package_manager)
    set(_found FALSE)
    if(NOT WIN32)
        find_program(_dpkg_exe dpkg)
        if(_dpkg_exe)
            execute_process(
                COMMAND ${_dpkg_exe} -s amdrocm-runtime
                RESULT_VARIABLE _result
                OUTPUT_QUIET ERROR_QUIET
            )
            if(_result EQUAL 0)
                set(_found TRUE)
            endif()
        endif()
        if(NOT _found)
            find_program(_rpm_exe rpm)
            if(_rpm_exe)
                execute_process(
                    COMMAND ${_rpm_exe} -q amdrocm-runtime
                    RESULT_VARIABLE _result
                    OUTPUT_QUIET ERROR_QUIET
                )
                if(_result EQUAL 0)
                    set(_found TRUE)
                endif()
            endif()
        endif()
        unset(_dpkg_exe CACHE)
        unset(_rpm_exe CACHE)
    endif()
    set(_THEROCK_DETECTED ${_found} PARENT_SCOPE)
endfunction()

function(detect_package_backend)
    if(NOT DEFINED CACHE{MIGRAPHX_PACKAGE_BACKEND})
        # No explicit -D flag: auto-detect via package manager (fallback)
        _detect_therock_via_package_manager()
        if(_THEROCK_DETECTED)
            set(_default_backend "therock")
            message(STATUS "MIGraphX package backend auto-detected: therock (amdrocm-runtime found)")
            message(STATUS "  Hint: prefer explicit -DMIGRAPHX_PACKAGE_BACKEND=therock -DMIGRAPHX_THEROCK_GPU_ARCH=<arch>")
        else()
            set(_default_backend "default")
        endif()
        set(MIGRAPHX_PACKAGE_BACKEND "${_default_backend}" CACHE STRING
            "Packaging backend: 'default' for traditional ROCm, 'therock' for TheRock amdrocm packages")
    endif()

    set_property(CACHE MIGRAPHX_PACKAGE_BACKEND PROPERTY STRINGS "default" "therock")

    set(_valid_backends "default" "therock")
    if(NOT MIGRAPHX_PACKAGE_BACKEND IN_LIST _valid_backends)
        message(FATAL_ERROR
            "MIGRAPHX_PACKAGE_BACKEND='${MIGRAPHX_PACKAGE_BACKEND}' is not valid. "
            "Must be one of: ${_valid_backends}")
    endif()

    if(MIGRAPHX_PACKAGE_BACKEND STREQUAL "therock")
        if(DEFINED ENV{MIGRAPHX_THEROCK_GPU_ARCH})
            set(_default_gpu_arch "$ENV{MIGRAPHX_THEROCK_GPU_ARCH}")
        else()
            set(_default_gpu_arch "")
        endif()
        set(MIGRAPHX_THEROCK_GPU_ARCH "${_default_gpu_arch}" CACHE STRING
            "TheRock GPU architecture family suffix (e.g. gfx120x ..)")

        if(MIGRAPHX_THEROCK_GPU_ARCH STREQUAL "")
            message(FATAL_ERROR
                "MIGRAPHX_PACKAGE_BACKEND=therock requires MIGRAPHX_THEROCK_GPU_ARCH to be set. "
                "Example: cmake -DMIGRAPHX_PACKAGE_BACKEND=therock -DMIGRAPHX_THEROCK_GPU_ARCH=gfx120x ..")
        endif()

        message(STATUS "MIGraphX package backend: therock (GPU arch: ${MIGRAPHX_THEROCK_GPU_ARCH})")
    else()
        message(STATUS "MIGraphX package backend: default (traditional ROCm)")
    endif()
endfunction()
