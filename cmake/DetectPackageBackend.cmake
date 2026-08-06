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
# Preferred usage:
#   cmake -DMIGRAPHX_PACKAGE_BACKEND=therock -DGPU_TARGETS="gfx942;gfx950" ..
#
# MIGRAPHX_THEROCK_GPU_ARCH is an optional package-architecture override. It is
# primarily for TheRock (<=7.14) repositories whose package suffix is a GPU family
# (for example gfx94x) rather than the raw GPU_TARGETS value (gfx942).
#
# If MIGRAPHX_PACKAGE_BACKEND is not set, falls back to auto-detection via
# dpkg/rpm to check for installed amdrocm-runtime packages.

function(_detect_therock_via_package_manager)
    set(_found FALSE)
    if(NOT WIN32)
        find_program(_migraphx_dpkg_exe dpkg)
        if(_migraphx_dpkg_exe)
            execute_process(
                COMMAND ${_migraphx_dpkg_exe} -s amdrocm-runtime
                RESULT_VARIABLE _result
                OUTPUT_QUIET ERROR_QUIET
            )
            if(_result EQUAL 0)
                set(_found TRUE)
            endif()
        endif()
        if(NOT _found)
            find_program(_migraphx_rpm_exe rpm)
            if(_migraphx_rpm_exe)
                execute_process(
                    COMMAND ${_migraphx_rpm_exe} -q amdrocm-runtime
                    RESULT_VARIABLE _result
                    OUTPUT_QUIET ERROR_QUIET
                )
                if(_result EQUAL 0)
                    set(_found TRUE)
                endif()
            endif()
        endif()
        unset(_migraphx_dpkg_exe CACHE)
        unset(_migraphx_rpm_exe CACHE)
    endif()
    set(_MIGRAPHX_THEROCK_DETECTED ${_found} PARENT_SCOPE)
endfunction()

function(detect_package_backend)
    if(NOT DEFINED CACHE{MIGRAPHX_PACKAGE_BACKEND})
        # No explicit -D flag: auto-detect via package manager (fallback)
        _detect_therock_via_package_manager()
        if(_MIGRAPHX_THEROCK_DETECTED)
            set(_default_backend "therock")
            message(STATUS "MIGraphX package backend auto-detected: therock (amdrocm-runtime found)")
            message(STATUS
                "  Hint: prefer explicit -DMIGRAPHX_PACKAGE_BACKEND=therock "
                "-DGPU_TARGETS=<arch>[;<arch>...]")
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
        if(DEFINED ENV{GPU_ARCH_FOR_THEROCK})
            # TheRock <=7.14 package-architecture override.
            set(_default_gpu_arch "$ENV{GPU_ARCH_FOR_THEROCK}")
        elseif(GPU_TARGETS)
            set(_default_gpu_arch "")
            foreach(_gpu_target IN LISTS GPU_TARGETS)
                string(REGEX REPLACE ":.*$" "" _gpu_arch "${_gpu_target}")
                list(APPEND _default_gpu_arch "${_gpu_arch}")
            endforeach()
            list(REMOVE_DUPLICATES _default_gpu_arch)
        else()
            set(_default_gpu_arch "")
        endif()
        set(MIGRAPHX_THEROCK_GPU_ARCH "${_default_gpu_arch}" CACHE STRING
            "Optional TheRock package architecture override. Defaults to GPU_TARGETS; use a legacy package family such as gfx94x when required.")

        if(MIGRAPHX_THEROCK_GPU_ARCH)
            message(STATUS
                "MIGraphX package backend: therock "
                "(package arches: ${MIGRAPHX_THEROCK_GPU_ARCH})")
        else()
            message(STATUS
                "MIGraphX package backend: therock "
                "(no package arches; using all-device meta packages)")
        endif()
    else()
        message(STATUS "MIGraphX package backend: default (traditional ROCm)")
    endif()
endfunction()
