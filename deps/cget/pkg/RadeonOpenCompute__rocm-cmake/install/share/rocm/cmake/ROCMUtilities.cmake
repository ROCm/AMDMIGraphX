# ######################################################################################################################
# Copyright (C) 2021 Advanced Micro Devices, Inc.
# ######################################################################################################################

if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.12.0")
    # pretty much just a wrapper around string JOIN
    function(rocm_join_if_set glue inout_variable)
        string(JOIN "${glue}" to_set_parent ${${inout_variable}} ${ARGN})
        set(${inout_variable} "${to_set_parent}" PARENT_SCOPE)
    endfunction()
else()
    # cmake < 3.12 doesn't have string JOIN
    function(rocm_join_if_set glue inout_variable)
        set(accumulator "${${inout_variable}}")
        set(tglue ${glue})
        if(accumulator STREQUAL "")
            set(tglue "")       # No glue needed if initially unset
        endif()
        foreach(ITEM IN LISTS ARGN)
            string(CONCAT accumulator "${accumulator}" "${tglue}" "${ITEM}")
            set(tglue ${glue})  # Always need glue after the first concatenate
        endforeach()
        set(${inout_variable} "${accumulator}" PARENT_SCOPE)
    endfunction()
endif()

function(rocm_find_program_version PROGRAM)
    set(options QUIET REQUIRED)
    set(oneValueArgs GREATER GREATER_EQUAL LESS LESS_EQUAL EQUAL OUTPUT_VARIABLE)
    set(multiValueArgs)

    cmake_parse_arguments(PARSE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT DEFINED PARSE_OUTPUT_VARIABLE)
        set(PARSE_OUTPUT_VARIABLE "${PROGRAM}_VERSION")
    endif()

    execute_process(
        COMMAND ${PROGRAM} --version
        RESULT_VARIABLE PROC_RESULT
        OUTPUT_VARIABLE EVAL_RESULT
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(NOT PROC_RESULT EQUAL "0")
        set(${PARSE_OUTPUT_VARIABLE} "0.0.0" PARENT_SCOPE)
        set(${PARSE_OUTPUT_VARIABLE}_OK FALSE PARENT_SCOPE)
        if(PARSE_REQUIRED)
            message(FATAL_ERROR "Could not determine the version of required program ${PROGRAM}.")
        elseif(NOT PARSE_QUIET)
            message(WARNING "Could not determine the version of program ${PROGRAM}.")
        endif()
    else()
        message("${EVAL_RESULT}")
        string(REGEX MATCH "[0-9]+(\\.[^ \t\r\n]+)*" PROGRAM_VERSION "${EVAL_RESULT}")
        set(${PARSE_OUTPUT_VARIABLE} "${PROGRAM_VERSION}" PARENT_SCOPE)
        set(${PARSE_OUTPUT_VARIABLE}_OK TRUE PARENT_SCOPE)
        foreach(COMP GREATER GREATER_EQUAL LESS LESS_EQUAL EQUAL)
            if(DEFINED PARSE_${COMP} AND NOT PROGRAM_VERSION VERSION_${COMP} PARSE_${COMP})
                set(${PARSE_OUTPUT_VARIABLE}_OK FALSE PARENT_SCOPE)
            endif()
        endforeach()
    endif()
endfunction()

function(rocm_set_os_id OS_ID)
    set(_os_id "unknown")
    if(EXISTS "/etc/os-release")
        rocm_read_os_release(_os_id "ID")
    endif()
    set(${OS_ID}
        ${_os_id}
        PARENT_SCOPE)
    set(os_id_out ${OS_ID}_${_os_id})
    set(${os_id_out}
        TRUE
        PARENT_SCOPE)
endfunction()

function(rocm_read_os_release OUTPUT KEYVALUE)
    # finds the line with the keyvalue
    if(EXISTS "/etc/os-release")
        file(STRINGS /etc/os-release _keyvalue_line REGEX "^${KEYVALUE}=")
    endif()

    # remove keyvalue=
    string(REGEX REPLACE "^${KEYVALUE}=\"?(.*)" "\\1" _output "${_keyvalue_line}")

    # remove trailing quote
    string(REGEX REPLACE "\"$" "" _output "${_output}")
    set(${OUTPUT}
        ${_output}
        PARENT_SCOPE)
endfunction()
