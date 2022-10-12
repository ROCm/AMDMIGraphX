# ######################################################################################################################
# Copyright (C) 2017 Advanced Micro Devices, Inc.
# ######################################################################################################################

macro(rocm_set_parent VAR)
    set(${VAR}
        ${ARGN}
        PARENT_SCOPE)
    set(${VAR} ${ARGN})
endmacro()

find_program(GIT NAMES git)

function(rocm_get_rev_count OUTPUT_COUNT)
    set(options)
    set(oneValueArgs DIRECTORY)
    set(multiValueArgs REV)

    cmake_parse_arguments(PARSE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    if(PARSE_DIRECTORY)
        set(DIRECTORY ${PARSE_DIRECTORY})
    endif()

    set(_count 0)
    if(GIT)
        execute_process(
            COMMAND git rev-list --count ${PARSE_REV}
            WORKING_DIRECTORY ${DIRECTORY}
            OUTPUT_VARIABLE REV_COUNT
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE RESULT
            ERROR_QUIET)
        if(${RESULT} EQUAL 0)
            set(_count ${REV_COUNT})
        endif()
    endif()
    rocm_set_parent(${OUTPUT_COUNT} ${_count})
endfunction()

function(rocm_get_commit_count OUTPUT_COUNT)
    set(options)
    set(oneValueArgs PARENT DIRECTORY)
    set(multiValueArgs)

    cmake_parse_arguments(PARSE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    if(PARSE_DIRECTORY)
        set(DIRECTORY ${PARSE_DIRECTORY})
    endif()

    rocm_get_rev_count(
        ALL_COUNT
        DIRECTORY ${DIRECTORY}
        REV HEAD)
    set(_count ${ALL_COUNT})

    if(PARSE_PARENT)
        rocm_get_rev_count(
            PARENT_COUNT
            DIRECTORY ${DIRECTORY}
            REV HEAD ^${PARSE_PARENT})
        set(_count ${PARENT_COUNT})
    endif()
    rocm_set_parent(${OUTPUT_COUNT} ${_count})
endfunction()

function(rocm_get_build_info OUTPUT DELIM)
    set(_info)
    if(DEFINED ENV{ROCM_BUILD_ID})
        set(_info ${_info}${DELIM}$ENV{ROCM_BUILD_ID})
    endif()
    rocm_set_parent(${OUTPUT} ${_info})
endfunction()

function(rocm_version_regex_parse REGEX OUTPUT_VARIABLE INPUT)
    string(REGEX REPLACE ${REGEX} "\\1" OUTPUT "${INPUT}")
    if("${OUTPUT}" STREQUAL "${INPUT}")
        rocm_set_parent(${OUTPUT_VARIABLE} 0)
    else()
        rocm_set_parent(${OUTPUT_VARIABLE} ${OUTPUT})
    endif()
endfunction()

function(rocm_get_git_commit_tag OUTPUT_VERSION)
    set(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

    if(GIT)
        set(GIT_COMMAND ${GIT} describe --dirty --long --match [0-9]*)
        execute_process(
            COMMAND ${GIT_COMMAND}
            WORKING_DIRECTORY ${DIRECTORY}
            OUTPUT_VARIABLE GIT_TAG_VERSION
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE RESULT
            ERROR_QUIET)
        if(${RESULT} EQUAL 0)
            set(_output ${GIT_TAG_VERSION})
        else()
            execute_process(
                COMMAND ${GIT_COMMAND} --always
                WORKING_DIRECTORY ${DIRECTORY}
                OUTPUT_VARIABLE GIT_TAG_VERSION
                OUTPUT_STRIP_TRAILING_WHITESPACE
                RESULT_VARIABLE RESULT
                ERROR_QUIET)
            if(${RESULT} EQUAL 0)
                set(_output ${GIT_TAG_VERSION})
            endif()
        endif()
    else()
        set(_output "")
    endif()
    rocm_set_parent(${OUTPUT_VERSION} ${_output})
endfunction()

function(rocm_setup_version)
    set(options NO_GIT_TAG_VERSION)
    set(oneValueArgs VERSION PARENT)
    set(multiValueArgs)

    cmake_parse_arguments(PARSE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(PARSE_VERSION)
        # Compensate for missing patch version
        if(PARSE_VERSION MATCHES "^[0-9]+\\.[0-9]+$")
            set(PARSE_VERSION ${PARSE_VERSION}.0)
        endif()

        rocm_set_parent(PROJECT_VERSION ${PARSE_VERSION})
        rocm_set_parent(${PROJECT_NAME}_VERSION ${PROJECT_VERSION})
        rocm_version_regex_parse("^([0-9]+).*" _version_MAJOR "${PROJECT_VERSION}")
        rocm_version_regex_parse("^[0-9]+\\.([0-9]+).*" _version_MINOR "${PROJECT_VERSION}")
        rocm_version_regex_parse("^[0-9]+\\.[0-9]+\\.([0-9]+).*" _version_PATCH "${PROJECT_VERSION}")
        foreach(level MAJOR MINOR PATCH)
            rocm_set_parent(${PROJECT_NAME}_VERSION_${level} ${_version_${level}})
            rocm_set_parent(PROJECT_VERSION_${level} ${_version_${level}})
        endforeach()
        set(ROCM_GIT_TAG_HASH "")
        if(NOT PARSE_NO_GIT_TAG_VERSION)
            rocm_get_git_commit_tag(ROCM_GIT_TAG_HASH)
        endif()
        rocm_set_parent(PROJECT_VERSION_TWEAK ${ROCM_GIT_TAG_HASH})
        rocm_set_parent(${PROJECT_NAME}_VERSION_TWEAK ${ROCM_GIT_TAG_HASH})
    endif()
endfunction()

function(rocm_get_patch_version OUTPUT)
    set(_patch "")
    if(DEFINED ENV{ROCM_LIBPATCH_VERSION})
        set(_patch $ENV{ROCM_LIBPATCH_VERSION})
    endif()
    rocm_set_parent(${OUTPUT} ${_patch})
endfunction()

function(rocm_set_soversion LIBRARY_TARGET SOVERSION)
    if(NOT WIN32 AND NOT APPLE)
        rocm_version_regex_parse("^([0-9]+).*" LIB_VERSION_MAJOR "${SOVERSION}")
        rocm_version_regex_parse("^[0-9]+\\.(.*)" LIB_VERSION_MINOR "${SOVERSION}")

        set(LIB_VERSION_STRING "${LIB_VERSION_MAJOR}.${LIB_VERSION_MINOR}")
        rocm_get_patch_version(LIB_VERSION_PATCH)
        if(NOT ${LIB_VERSION_PATCH} EQUAL "")
            set(LIB_VERSION_STRING "${LIB_VERSION_STRING}.${LIB_VERSION_PATCH}")
        endif()

        set_target_properties(${LIBRARY_TARGET} PROPERTIES SOVERSION ${LIB_VERSION_MAJOR})
        set_target_properties(${LIBRARY_TARGET} PROPERTIES VERSION ${LIB_VERSION_STRING})
    endif()
endfunction()
