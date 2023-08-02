# ####################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
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
# ####################################################################################
# https://github.com/RadeonOpenCompute/rocm-cmake/blob/f0717bc1a0182b2ddc9194e3144972ab58a88e99/share/rocm/cmake/ROCMLocalTargets.cmake#L16C57-L16C57
# once it is available from rocm-cmake  or hip, remove this
function(rocm_local_targets_migx VARIABLE)
    # rocm_agent_enumerator is only available on Linux platforms right now
    find_program(_rocm_agent_enumerator rocm_agent_enumerator HINTS /opt/rocm/bin ENV ROCM_PATH)

    if(NOT _rocm_agent_enumerator STREQUAL "_rocm_agent_enumerator-NOTFOUND")
        execute_process(
            COMMAND "${_rocm_agent_enumerator}"
            RESULT_VARIABLE _found_agents
            OUTPUT_VARIABLE _rocm_agents
            ERROR_QUIET
        )

        if(_found_agents EQUAL 0)
            cmake_policy(SET CMP0007 NEW)
            string(REPLACE "\n" ";" _rocm_agents "${_rocm_agents}")
            list(POP_BACK _rocm_agents)
            list(REMOVE_DUPLICATES _rocm_agents)
            unset(result)

            foreach(agent IN LISTS _rocm_agents)
                if(NOT agent STREQUAL "gfx000")
                    list(APPEND result "${agent}")
                endif()
            endforeach()

            if(result)
                set(${VARIABLE} "${result}" PARENT_SCOPE)
            endif()
        endif()
    endif()
endfunction()
