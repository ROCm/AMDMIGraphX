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

# Check that migraphx-driver verify reports its result through the exit code:
# 0 when verification passes and 1 when it fails. The tests compare the ref
# target against itself, so they need no GPU. Forcing every tolerance to -1
# makes the comparison fail deterministically.

set(VERIFY_EXIT_CODE_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/verify_exit_code.cmake)
set(VERIFY_FORCE_FAIL --rms-tol -1 --atol -1 --rtol -1)

function(add_driver_verify_exit_test NAME EXPECTED)
    add_test(NAME test_driver_verify_${NAME}
             COMMAND ${CMAKE_COMMAND}
                     -DDRIVER=$<TARGET_FILE:driver>
                     -DEXPECTED=${EXPECTED}
                     "-DARGS=${ARGN}"
                     -P ${VERIFY_EXIT_CODE_SCRIPT})
endfunction()

foreach(MODE default per-instruction layerwise)
    if(MODE STREQUAL "default")
        set(MODE_FLAG)
    else()
        set(MODE_FLAG --${MODE})
    endif()
    add_driver_verify_exit_test(${MODE}_pass 0 verify --test --ref ${MODE_FLAG})
    add_driver_verify_exit_test(${MODE}_fail 1 verify --test --ref ${MODE_FLAG} ${VERIFY_FORCE_FAIL})
endforeach()

# --reduce and --bisect only verify truncated programs, so they need a model
# with a chain of operations; the single GEMM of --test has nothing to compare.
if(MIGRAPHX_ENABLE_ONNX)
    set(VERIFY_CHAIN_MODEL ${CMAKE_SOURCE_DIR}/test/onnx/instance_norm_test.onnx)
    foreach(MODE reduce bisect)
        add_driver_verify_exit_test(${MODE}_pass 0 verify ${VERIFY_CHAIN_MODEL} --ref --${MODE})
        add_driver_verify_exit_test(${MODE}_fail 1 verify ${VERIFY_CHAIN_MODEL} --ref --${MODE} ${VERIFY_FORCE_FAIL})
    endforeach()
endif()

add_dependencies(tests driver)
add_dependencies(check driver)
