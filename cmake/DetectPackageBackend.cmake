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

# Detect the packaging backend for MIGraphX and auto-configure TheRock settings.
#
# detect_package_backend() probes the installed system packages and, when not
# overridden on the command line, fills in everything needed to build and
# package against TheRock so that a bare `rbuild package` (or
# `cmake --build build --target package`) works without extra -D flags.
#
# It sets these cache variables (each only when not already provided):
#   MIGRAPHX_PACKAGE_BACKEND      - "therock" if any installed package name
#                                   starts with "amdrocm", else "default".
#   MIGRAPHX_THEROCK_ROCM_VERSION - ROCm version suffix parsed from an installed
#                                   amdrocm-blas<ver>-<arch> package (e.g. 7.13).
#   MIGRAPHX_THEROCK_GPU_ARCH     - the first detected per-GPU arch (e.g. gfx942),
#                                   used to select the per-GPU package dependency.
#   GPU_TARGETS                   - semicolon list of ALL detected arches
#                                   (e.g. gfx942;gfx950), used for compilation.
#
# Precedence per variable: an explicit -D / already-set value wins, then the
# GPU_ARCH_FOR_THEROCK env fallback (GPU arch only), then the detected value.
#
# Explicit usage still works and overrides detection, e.g.:
#   cmake -DMIGRAPHX_PACKAGE_BACKEND=therock -DMIGRAPHX_THEROCK_GPU_ARCH=gfx942 \
#         -DMIGRAPHX_THEROCK_ROCM_VERSION=7.13 -DGPU_TARGETS="gfx942;gfx950" ..

# Probe installed packages via dpkg/rpm. Returns (in PARENT_SCOPE):
#   _AMDROCM_ANY          - TRUE if any installed package name starts with amdrocm
#   _THEROCK_VERSION      - version suffix from the first amdrocm-blas<ver>[-<arch>]
#   _THEROCK_FIRST_ARCH   - arch suffix from the first amdrocm-blas<ver>-<arch>
#   _THEROCK_ALL_ARCHS    - deduped, sorted list of all detected arches
function(_probe_amdrocm_packages)
    set(_names "")
    if(NOT WIN32)
        find_program(_migraphx_dpkg_query_exe dpkg-query)
        if(_migraphx_dpkg_query_exe)
            execute_process(
                COMMAND ${_migraphx_dpkg_query_exe}
                        -W "-f=\${db:Status-Abbrev} \${Package}\n" amdrocm*
                OUTPUT_VARIABLE _dpkg_out
                RESULT_VARIABLE _dpkg_res
                ERROR_QUIET
            )
            if(_dpkg_res EQUAL 0 AND _dpkg_out)
                string(REPLACE "\n" ";" _dpkg_lines "${_dpkg_out}")
                foreach(_line IN LISTS _dpkg_lines)
                    # Keep only fully-installed packages (status starts with "ii").
                    if(_line MATCHES "^ii[^ ]* +([A-Za-z0-9._+-]+)")
                        list(APPEND _names "${CMAKE_MATCH_1}")
                    endif()
                endforeach()
            endif()
        endif()
        unset(_migraphx_dpkg_query_exe CACHE)

        if(NOT _names)
            find_program(_migraphx_rpm_exe rpm)
            if(_migraphx_rpm_exe)
                execute_process(
                    COMMAND ${_migraphx_rpm_exe} -qa --qf "%{NAME}\n" amdrocm*
                    OUTPUT_VARIABLE _rpm_out
                    RESULT_VARIABLE _rpm_res
                    ERROR_QUIET
                )
                if(_rpm_res EQUAL 0 AND _rpm_out)
                    string(REPLACE "\n" ";" _rpm_lines "${_rpm_out}")
                    foreach(_line IN LISTS _rpm_lines)
                        string(STRIP "${_line}" _line)
                        if(_line)
                            list(APPEND _names "${_line}")
                        endif()
                    endforeach()
                endif()
            endif()
            unset(_migraphx_rpm_exe CACHE)
        endif()
    endif()

    set(_any FALSE)
    set(_version "")
    set(_first_arch "")
    set(_all_archs "")
    set(_version_no_arch "")
    if(_names)
        list(SORT _names)
        foreach(_name IN LISTS _names)
            if(_name MATCHES "^amdrocm")
                set(_any TRUE)
            endif()
            # Arch-suffixed blas package: amdrocm-blas<ver>-<arch>
            if(_name MATCHES "^amdrocm-blas([0-9][0-9.]*)-(gfx[0-9a-z]+)$")
                if(NOT _version)
                    set(_version "${CMAKE_MATCH_1}")
                endif()
                if(NOT _first_arch)
                    set(_first_arch "${CMAKE_MATCH_2}")
                endif()
                list(APPEND _all_archs "${CMAKE_MATCH_2}")
            # Non-arch blas package: amdrocm-blas<ver> (version fallback only)
            elseif(_name MATCHES "^amdrocm-blas([0-9][0-9.]*)$")
                if(NOT _version_no_arch)
                    set(_version_no_arch "${CMAKE_MATCH_1}")
                endif()
            endif()
        endforeach()
    endif()

    # Fall back to the non-arch blas package version when no per-GPU blas is found.
    if(NOT _version AND _version_no_arch)
        set(_version "${_version_no_arch}")
    endif()

    if(_all_archs)
        list(REMOVE_DUPLICATES _all_archs)
        list(SORT _all_archs)
    endif()

    set(_AMDROCM_ANY ${_any} PARENT_SCOPE)
    set(_THEROCK_VERSION "${_version}" PARENT_SCOPE)
    set(_THEROCK_FIRST_ARCH "${_first_arch}" PARENT_SCOPE)
    set(_THEROCK_ALL_ARCHS "${_all_archs}" PARENT_SCOPE)
endfunction()

function(detect_package_backend)
    _probe_amdrocm_packages()

    if(NOT DEFINED CACHE{MIGRAPHX_PACKAGE_BACKEND})
        # No explicit -D flag: auto-detect from installed amdrocm* packages.
        if(_AMDROCM_ANY)
            set(_default_backend "therock")
            message(STATUS "MIGraphX package backend auto-detected: therock (amdrocm-* packages found)")
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
        # ROCm version suffix used in all amdrocm-* dependency names.
        if(NOT DEFINED CACHE{MIGRAPHX_THEROCK_ROCM_VERSION})
            set(MIGRAPHX_THEROCK_ROCM_VERSION "${_THEROCK_VERSION}" CACHE STRING
                "TheRock ROCm major.minor version suffix for package dependencies (e.g. 7.13)")
        endif()

        # Per-GPU arch used to select the per-GPU package dependency (first only).
        if(NOT DEFINED CACHE{MIGRAPHX_THEROCK_GPU_ARCH})
            if(DEFINED ENV{GPU_ARCH_FOR_THEROCK})
                # Env name drops MIGRAPHX_ prefix to avoid the "unused MIGRAPHX_* env" warning.
                set(_default_gpu_arch "$ENV{GPU_ARCH_FOR_THEROCK}")
            else()
                set(_default_gpu_arch "${_THEROCK_FIRST_ARCH}")
            endif()
            set(MIGRAPHX_THEROCK_GPU_ARCH "${_default_gpu_arch}" CACHE STRING
                "TheRock GPU arch(es) for per-GPU package dependencies (e.g. gfx942). \
Semicolon-separated list for per-GPU deps, or empty for device-all meta-package deps.")
        endif()

        # GPU_TARGETS (what to compile) defaults to ALL detected arches. Only set
        # when not already provided by -D, the HIP package, or env so we never
        # clobber an explicit choice.
        if(NOT GPU_TARGETS AND _THEROCK_ALL_ARCHS)
            set(GPU_TARGETS "${_THEROCK_ALL_ARCHS}" CACHE STRING
                "GPU architectures to compile for (auto-detected from TheRock amdrocm-blas packages)")
        endif()

        message(STATUS "MIGraphX package backend: therock "
            "(ROCm version: '${MIGRAPHX_THEROCK_ROCM_VERSION}', "
            "package GPU arch: '${MIGRAPHX_THEROCK_GPU_ARCH}', "
            "GPU_TARGETS: '${GPU_TARGETS}')")
    else()
        message(STATUS "MIGraphX package backend: default (traditional ROCm)")
    endif()
endfunction()
