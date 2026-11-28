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

# Run migraphx-driver and fail unless it exits with exactly the EXPECTED code.
# Used instead of WILL_FAIL so that a crash is not mistaken for a reported
# verification failure.
#
# Inputs: DRIVER (path to migraphx-driver), EXPECTED (exit code), ARGS (list).

execute_process(COMMAND ${DRIVER} ${ARGS}
                RESULT_VARIABLE RESULT
                OUTPUT_VARIABLE OUTPUT
                ERROR_VARIABLE OUTPUT)

if(NOT "${RESULT}" STREQUAL "${EXPECTED}")
    message("${OUTPUT}")
    message(FATAL_ERROR "migraphx-driver ${ARGS} exited with '${RESULT}', expected ${EXPECTED}")
endif()
