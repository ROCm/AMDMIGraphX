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

# Invoked as:
#   cmake -DOBJDUMP=<path> -DTARGET_FILE=<lib.so> -DARCHS=gfx906,gfx1201 \
#         -P verify_offload_archs.cmake
# On Linux: uses llvm-objdump --offloading on the shared library.
# On Windows: extracts the .hip_fat PE section with llvm-objcopy, unbundles
#             each arch with clang-offload-bundler, and verifies the actual
#             compiled target via llvm-readobj --notes.
# Note: ARCHS is COMMA-separated, not ';'-separated, to survive the
# custom-command argument boundary (CMake would split a ';' list otherwise).

if(NOT TARGET_FILE OR NOT ARCHS)
    message(FATAL_ERROR "verify_offload_archs: TARGET_FILE and ARCHS required")
endif()
string(REPLACE "," ";" _archs "${ARCHS}")

if(WIN32)
    if(NOT OBJCOPY OR NOT BUNDLER OR NOT READOBJ)
        message(FATAL_ERROR "verify_offload_archs: OBJCOPY, BUNDLER and READOBJ required on Windows")
    endif()
    get_filename_component(_dir "${TARGET_FILE}" DIRECTORY)
    set(_hip_fat "${_dir}/migraphx_hip_fat_tmp.bin")
    execute_process(
        COMMAND "${OBJCOPY}" "--dump-section=.hip_fat=${_hip_fat}" "${TARGET_FILE}"
        RESULT_VARIABLE _rc ERROR_VARIABLE _err)
    if(NOT _rc EQUAL 0)
        message(FATAL_ERROR "llvm-objcopy failed to extract .hip_fat from ${TARGET_FILE}:\n${_err}")
    endif()
    if(NOT EXISTS "${_hip_fat}")
        message(FATAL_ERROR "llvm-objcopy did not produce ${_hip_fat}")
    endif()
    set(_missing "")
    foreach(_a IN LISTS _archs)
        set(_unbundled "${_dir}/migraphx_verify_${_a}_tmp.o")
        execute_process(
            COMMAND "${BUNDLER}" --unbundle --type=o
                    --input "${_hip_fat}"
                    --output "${_unbundled}"
                    --targets "hipv4-amdgcn-amd-amdhsa--${_a}"
            RESULT_VARIABLE _rc ERROR_VARIABLE _err)
        if(NOT _rc EQUAL 0)
            list(APPEND _missing "${_a}")
            continue()
        endif()
        execute_process(
            COMMAND "${READOBJ}" --notes "${_unbundled}"
            OUTPUT_VARIABLE _notes RESULT_VARIABLE _rc)
        file(REMOVE "${_unbundled}")
        # Search only the amdhsa.target line to avoid false positives from
        # arch names appearing elsewhere in kernel metadata or symbol names.
        string(REGEX MATCH "amdhsa\\.target:[^\r\n]*" _target_line "${_notes}")
        string(FIND "${_target_line}" "${_a}" _pos)
        if(_pos EQUAL -1)
            list(APPEND _missing "${_a}")
        endif()
    endforeach()
    file(REMOVE "${_hip_fat}")
else()
    if(NOT OBJDUMP)
        message(FATAL_ERROR "verify_offload_archs: OBJDUMP required on non-Windows")
    endif()
    execute_process(
        COMMAND "${OBJDUMP}" --offloading "${TARGET_FILE}"
        OUTPUT_VARIABLE _dump RESULT_VARIABLE _rc ERROR_VARIABLE _err)
    if(NOT _rc EQUAL 0)
        message(FATAL_ERROR "llvm-objdump --offloading failed on ${TARGET_FILE}:\n${_err}")
    endif()
    set(_missing "")
    foreach(_a IN LISTS _archs)
        string(FIND "${_dump}" "${_a}" _pos)
        if(_pos EQUAL -1)
            list(APPEND _missing "${_a}")
        endif()
    endforeach()
endif()

if(_missing)
    message(FATAL_ERROR
        "Offload arch(s) missing from ${TARGET_FILE}: ${_missing}\n"
        "Try setting -DMIGRAPHX_WORKAROUND_HIP_MULTI_ARCH_BUG=ON to fix this issue.\n"
        "--offloading output:\n${_dump}")
endif()
message(STATUS "Offload archs verified in ${TARGET_FILE}: ${_archs}")
