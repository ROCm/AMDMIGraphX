# ######################################################################################################################
# Copyright (C) 2021 Advanced Micro Devices, Inc.
# ######################################################################################################################

include(CheckCXXCompilerFlag)
include(CMakeParseArguments)

function(rocm_check_target_ids VARIABLE)
    set(options)
    set(oneValueArgs)
    set(multiValueArgs TARGETS)

    cmake_parse_arguments(PARSE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(PARSE_UNPARSED_ARGUMENTS)
        message(
            FATAL_ERROR
                "Unknown keywords given to rocm_check_target_ids(): \"${PARSE_UNPARSED_ARGUMENTS}\"")
    endif()

    foreach(_target_id ${PARSE_TARGETS})
        set(_result_var "HAVE_${_target_id}")
        check_cxx_compiler_flag("-xhip --offload-arch=${_target_id}" "${_result_var}")
        if(${_result_var})
            list(APPEND _supported_target_ids "${_target_id}")
        endif()
    endforeach()
    set(${VARIABLE} "${_supported_target_ids}" PARENT_SCOPE)
endfunction()
