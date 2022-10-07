# ######################################################################################################################
# Copyright (C) 2017 Advanced Micro Devices, Inc.
# ######################################################################################################################

include(CMakeParseArguments)
include(ProcessorCount)
include(ROCMAnalyzers)

find_program(
    CPPCHECK_EXE
    NAMES cppcheck
    PATHS /opt/rocm/bin)

function(rocm_find_cppcheck_version VAR)
    execute_process(COMMAND ${CPPCHECK_EXE} --version OUTPUT_VARIABLE VERSION_OUTPUT)
    separate_arguments(VERSION_OUTPUT_LIST UNIX_COMMAND "${VERSION_OUTPUT}")
    list(LENGTH VERSION_OUTPUT_LIST VERSION_OUTPUT_LIST_LEN)
    if(VERSION_OUTPUT_LIST_LEN GREATER 1)
        list(GET VERSION_OUTPUT_LIST 1 VERSION)
        set(${VAR}
            ${VERSION}
            PARENT_SCOPE)
    else()
        set(${VAR}
            "0.0"
            PARENT_SCOPE)
    endif()

endfunction()

if(NOT CPPCHECK_EXE)
    message(STATUS "Cppcheck not found")
    set(CPPCHECK_VERSION "0.0")
else()
    rocm_find_cppcheck_version(CPPCHECK_VERSION)
    message(STATUS "Cppcheck found: ${CPPCHECK_VERSION}")
endif()

processorcount(CPPCHECK_JOBS)

set(CPPCHECK_BUILD_DIR
    ${CMAKE_BINARY_DIR}/cppcheck-build
    CACHE STRING "")
file(MAKE_DIRECTORY ${CPPCHECK_BUILD_DIR})
set_property(
    DIRECTORY
    APPEND
    PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${CPPCHECK_BUILD_DIR})

macro(rocm_enable_cppcheck)
    set(options FORCE INCONCLUSIVE)
    set(oneValueArgs RULE_FILE)
    set(multiValueArgs CHECKS SUPPRESS DEFINE UNDEFINE INCLUDE SOURCES ADDONS)

    cmake_parse_arguments(PARSE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    string(REPLACE ";" "," CPPCHECK_CHECKS "${PARSE_CHECKS}")
    string(REPLACE ";" "\n" CPPCHECK_SUPPRESS "${PARSE_SUPPRESS};*:/usr/*")
    file(WRITE ${CMAKE_BINARY_DIR}/cppcheck-supressions "${CPPCHECK_SUPPRESS}")
    set(CPPCHECK_DEFINES)
    foreach(DEF ${PARSE_DEFINE})
        list(APPEND CPPCHECK_DEFINES "-D${DEF}")
    endforeach()

    set(CPPCHECK_UNDEFINES)
    foreach(DEF ${PARSE_UNDEFINE})
        list(APPEND CPPCHECK_UNDEFINES "-U${DEF}")
    endforeach()

    set(CPPCHECK_INCLUDES)
    foreach(INC ${PARSE_INCLUDE})
        list(APPEND CPPCHECK_INCLUDES "-I${INC}")
    endforeach()

    # set(CPPCHECK_FORCE)
    set(CPPCHECK_FORCE "--project=${CMAKE_BINARY_DIR}/compile_commands.json")
    if(PARSE_FORCE)
        set(CPPCHECK_FORCE --force)
    endif()

    set(CPPCHECK_INCONCLUSIVE "")
    if(PARSE_INCONCLUSIVE)
        set(CPPCHECK_INCONCLUSIVE --inconclusive)
    endif()

    if(${CPPCHECK_VERSION} VERSION_LESS "1.80")
        set(CPPCHECK_BUILD_DIR_FLAG)
    else()
        set(CPPCHECK_BUILD_DIR_FLAG "--cppcheck-build-dir=${CPPCHECK_BUILD_DIR}")
    endif()

    if(${CPPCHECK_VERSION} VERSION_LESS "1.80")
        set(CPPCHECK_PLATFORM_FLAG)
    else()
        set(CPPCHECK_PLATFORM_FLAG "--platform=native")
    endif()

    set(CPPCHECK_RULE_FILE_ARG)
    if(PARSE_RULE_FILE)
        set(CPPCHECK_RULE_FILE_ARG "--rule-file=${PARSE_RULE_FILE}")
    endif()

    set(CPPCHECK_ADDONS_ARG)
    foreach(ADDON ${PARSE_ADDONS})
        list(APPEND CPPCHECK_ADDONS_ARG "--addon=${ADDON}")
    endforeach()

    set(CPPCHECK_TEMPLATE_ARG)
    if(ROCM_ENABLE_GH_ANNOTATIONS)
        # cmake-lint: disable=C0301
        set(CPPCHECK_TEMPLATE_ARG
            "--template=::warning file={file},line={line},col={column}::{severity}: {inconclusive:inconclusive: }{message} [{id}]"
            "--template-location={file}:{line}:{column}: note: {info}\\\n{code}")
    endif()

    set(CPPCHECK_COMMAND
        ${CPPCHECK_EXE} -q
        # -v --report-progress
        ${CPPCHECK_FORCE} ${CPPCHECK_INCONCLUSIVE} ${CPPCHECK_BUILD_DIR_FLAG} ${CPPCHECK_PLATFORM_FLAG}
        ${CPPCHECK_RULE_FILE_ARG} ${CPPCHECK_TEMPLATE_ARG} ${CPPCHECK_ADDONS_ARG} --inline-suppr --error-exitcode=1 -j
        ${CPPCHECK_JOBS} ${CPPCHECK_DEFINES} ${CPPCHECK_UNDEFINES} ${CPPCHECK_INCLUDES}
        "--relative-paths=${CMAKE_SOURCE_DIR}" --enable=${CPPCHECK_CHECKS}
        --suppressions-list=${CMAKE_BINARY_DIR}/cppcheck-supressions)

    add_custom_target(
        cppcheck
        COMMAND ${CPPCHECK_COMMAND} ${PARSE_SOURCES}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "cppcheck: Running cppcheck...")
    if(CPPCHECK_EXE)
        rocm_mark_as_analyzer(cppcheck)
    endif()
endmacro()
