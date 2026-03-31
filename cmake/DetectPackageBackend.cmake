#####################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2026-2026 Advanced Micro Devices, Inc. All rights reserved.
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
# Sets MIGRAPHX_PACKAGE_BACKEND as a cache variable with one of:
#   "therock"       - TheRock sub-project build (amdrocm-xxx packages)
#   "pre-installed" - Libraries in /opt/rocm but not registered in package manager
#   "default"       - Traditional ROCm with deb/rpm packages
#
# Can be overridden by the user via -DMIGRAPHX_PACKAGE_BACKEND=<value>

function(detect_package_backend)
    if (DEFINED MIGRAPHX_PACKAGE_BACKEND)
        message(STATUS "MIGraphX package backend (cached): ${MIGRAPHX_PACKAGE_BACKEND}")
        return()
    endif()

    # TheRock injects these variables into sub-projects
    if(DEFINED THEROCK_PROVIDED_PACKAGES OR DEFINED THEROCK_PACKAGE_VERSION)
        set(MIGRAPHX_PACKAGE_BACKEND "therock"
            CACHE STRING "Auto-detected: TheRock sub-project build")
        message(STATUS "MIGraphX package backend (auto-detected): therock")
        return()
    endif()
    
    if(WIN32)
        set(MIGRAPHX_PACKAGE_BACKEND "default"
            CACHE STRING "Default for Windows")
        message(STATUS "MIGraphX package backend (default): default (Windows)")
        return()
    endif()

    # Check system package managers with deb/rpm for traditional ROCm 
    set(_found_in_pkgmgr FALSE)
    find_program(_dpkg_exe dpkg)
    if(_dpkg_exe)
        execute_process(
            COMMAND ${_dpkg_exe} -s hip-runtime-amd
            RESULT_VARIABLE _dpkg_result
            OUTPUT_QUIET ERROR_QUIET
        )
        if(_dpkg_result EQUAL 0)
            set(_found_in_pkgmgr TRUE)
        endif()
    endif()
    if(NOT _found_in_pkgmgr)
        find_program(_rpm_exe rpm)
        if(_rpm_exe)
            execute_process(
                COMMAND ${_rpm_exe} -q hip-runtime-amd
                RESULT_VARIABLE _rpm_result
                OUTPUT_QUIET ERROR_QUIET
            )
            if(_rpm_result EQUAL 0)
                set(_found_in_pkgmgr TRUE)
            endif()
        endif()
    endif()
    if(_found_in_pkgmgr)
        set(MIGRAPHX_PACKAGE_BACKEND "default"
            CACHE STRING "Auto-detected: default ROCm (packages registered)")
        message(STATUS "MIGraphX package backend (auto-detected): default for deb/rpm packages")
        unset(_dpkg_exe CACHE)
        unset(_rpm_exe CACHE)
        return()
    endif()
    
    # Check for pre-installed libraries but not registered in package manager
    find_library(_hip_runtime_lib amdhip64
        PATHS /opt/rocm/lib
        NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
    if(_hip_runtime_lib)
        set(MIGRAPHX_PACKAGE_BACKEND "pre-installed"
            CACHE STRING "Auto-detected: pre-installed (libraries present, not registered)")
        message(STATUS "MIGraphX package backend (auto-detected): pre-installed (${_hip_runtime_lib} found but not in package manager)")
    else()
        set(MIGRAPHX_PACKAGE_BACKEND "default"
            CACHE STRING "Default: default ROCm")
        message(STATUS "MIGraphX package backend (default): default")
    endif()
    unset(_dpkg_exe CACHE)
    unset(_rpm_exe CACHE)
    unset(_hip_runtime_lib CACHE)
endfunction()