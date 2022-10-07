# ######################################################################################################################
# Copyright (C) 2017 Advanced Micro Devices, Inc.
# ######################################################################################################################
include(CMakeParseArguments)
include(ROCMAnalyzers)

get_filename_component(CLANG_TIDY_EXE_HINT "${CMAKE_CXX_COMPILER}" PATH)

find_program(
    CLANG_TIDY_EXE
    NAMES clang-tidy
          clang-tidy-9.0
          clang-tidy-8.0
          clang-tidy-7.0
          clang-tidy-6.0
          clang-tidy-5.0
          clang-tidy-4.0
          clang-tidy-3.9
          clang-tidy-3.8
          clang-tidy-3.7
          clang-tidy-3.6
          clang-tidy-3.5
    HINTS ${CLANG_TIDY_EXE_HINT}
    PATH_SUFFIXES compiler/bin bin
    PATHS /opt/rocm/llvm/bin /opt/rocm/hcc /usr/local/opt/llvm/bin)

execute_process(COMMAND ${CMAKE_CXX_COMPILER} --version OUTPUT_VARIABLE CLANG_TIDY_COMPILER_VERSION_OUTPUT)
function(rocm_find_clang_tidy_version VAR)
    execute_process(COMMAND ${CLANG_TIDY_EXE} -version OUTPUT_VARIABLE VERSION_OUTPUT)
    separate_arguments(VERSION_OUTPUT_LIST UNIX_COMMAND "${VERSION_OUTPUT}")
    list(FIND VERSION_OUTPUT_LIST "version" VERSION_INDEX)
    if(VERSION_INDEX GREATER 0)
        math(EXPR VERSION_INDEX "${VERSION_INDEX} + 1")
        list(GET VERSION_OUTPUT_LIST ${VERSION_INDEX} VERSION)
        set(${VAR}
            ${VERSION}
            PARENT_SCOPE)
    else()
        set(${VAR}
            "0.0"
            PARENT_SCOPE)
    endif()

endfunction()

if(NOT CLANG_TIDY_EXE)
    message(STATUS "Clang tidy not found")
    set(CLANG_TIDY_VERSION "0.0")
else()
    rocm_find_clang_tidy_version(CLANG_TIDY_VERSION)
    message(STATUS "Clang tidy found: ${CLANG_TIDY_VERSION}")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CLANG_TIDY_CACHE
    "${CMAKE_BINARY_DIR}/tidy-cache"
    CACHE STRING "")

if(CMAKE_GENERATOR MATCHES "Make")
    set(CLANG_TIDY_CACHE_SIZE
        10
        CACHE STRING "")
else()
    set(CLANG_TIDY_CACHE_SIZE
        0
        CACHE STRING "")
endif()
set(CLANG_TIDY_FIXIT_DIR ${CMAKE_BINARY_DIR}/fixits)
file(MAKE_DIRECTORY ${CLANG_TIDY_FIXIT_DIR})
set_property(
    DIRECTORY
    APPEND
    PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${CLANG_TIDY_FIXIT_DIR})

set(CLANG_TIDY_DEPEND_ON_TARGET
    On
    CACHE BOOL "")

set(ROCM_ENABLE_GH_ANNOTATIONS
    Off
    CACHE BOOL "Enable github annotations in output")

set(CLANG_TIDY_USE_COLOR
    On
    CACHE BOOL "Enable color diagnostics in output")

macro(rocm_enable_clang_tidy)
    set(options ALL ANALYZE_TEMPORARY_DTORS ENABLE_ALPHA_CHECKS)
    set(oneValueArgs HEADER_FILTER)
    set(multiValueArgs CHECKS ERRORS EXTRA_ARGS CLANG_ARGS)

    cmake_parse_arguments(PARSE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    string(REPLACE ";" "," CLANG_TIDY_CHECKS "${PARSE_CHECKS}")
    string(REPLACE ";" "," CLANG_TIDY_ERRORS "${PARSE_ERRORS}")

    message(STATUS "Clang tidy checks: ${CLANG_TIDY_CHECKS}")

    set(CLANG_TIDY_ALL)
    if(PARSE_ALL)
        set(CLANG_TIDY_ALL ALL)
    endif()

    set(CLANG_TIDY_EXTRA_ARGS)
    foreach(ARG ${PARSE_EXTRA_ARGS})
        list(APPEND CLANG_TIDY_EXTRA_ARGS "-extra-arg=${ARG}")
    endforeach()

    foreach(ARG ${PARSE_CLANG_ARGS})
        list(APPEND CLANG_TIDY_EXTRA_ARGS -extra-arg=-Xclang "-extra-arg=${ARG}")
    endforeach()

    set(CLANG_TIDY_USE_COLOR_ARGS)
    if(${CLANG_TIDY_VERSION} VERSION_GREATER "11.0.0"
       AND CLANG_TIDY_USE_COLOR
       AND NOT ROCM_ENABLE_GH_ANNOTATIONS)
        set(CLANG_TIDY_USE_COLOR_ARGS "--use-color")
    endif()
    set(CLANG_TIDY_ENABLE_ALPHA_CHECKS_ARGS)
    if(PARSE_ENABLE_ALPHA_CHECKS)
        set(CLANG_TIDY_ENABLE_ALPHA_CHECKS_ARGS --allow-enabling-analyzer-alpha-checkers)
    endif()

    if(${CLANG_TIDY_VERSION} VERSION_LESS "3.9.0")
        set(CLANG_TIDY_ERRORS_ARG "")
    else()
        set(CLANG_TIDY_ERRORS_ARG "-warnings-as-errors=${CLANG_TIDY_ERRORS}")
    endif()

    if(${CLANG_TIDY_VERSION} VERSION_LESS "4.0.0" OR WIN32)
        set(CLANG_TIDY_QUIET_ARG "")
    else()
        set(CLANG_TIDY_QUIET_ARG "-quiet")
    endif()

    if(EXISTS ${CMAKE_SOURCE_DIR}/.clang-tidy)
        set(CLANG_TIDY_CONFIG_ARG "-config-file=${CMAKE_SOURCE_DIR}/.clang-tidy")
    else()
        set(CLANG_TIDY_CONFIG_ARG)
    endif()

    if(PARSE_HEADER_FILTER)
        string(REPLACE "$" "$$" CLANG_TIDY_HEADER_FILTER "${PARSE_HEADER_FILTER}")
    else()
        set(CLANG_TIDY_HEADER_FILTER ".*")
    endif()

    set(CLANG_TIDY_COMMAND
        ${CLANG_TIDY_EXE} ${CLANG_TIDY_USE_COLOR_ARGS} ${CLANG_TIDY_CONFIG_ARG} ${CLANG_TIDY_QUIET_ARG}
        ${CLANG_TIDY_ENABLE_ALPHA_CHECKS_ARGS} -p "${CMAKE_BINARY_DIR}" "-checks=${CLANG_TIDY_CHECKS}"
        "${CLANG_TIDY_ERRORS_ARG}" ${CLANG_TIDY_EXTRA_ARGS} "-header-filter=${CLANG_TIDY_HEADER_FILTER}")
    execute_process(COMMAND ${CLANG_TIDY_COMMAND} -dump-config OUTPUT_VARIABLE CLANG_TIDY_CONFIG)
    file(WRITE ${CMAKE_BINARY_DIR}/clang-tidy.yml ${CLANG_TIDY_CONFIG})
    add_custom_target(tidy ${CLANG_TIDY_ALL})
    if(CLANG_TIDY_EXE)
        rocm_mark_as_analyzer(tidy)
    endif()
    add_custom_target(tidy-base)
    add_custom_target(tidy-make-fixit-dir COMMAND ${CMAKE_COMMAND} -E make_directory ${CLANG_TIDY_FIXIT_DIR})
    add_custom_target(tidy-rm-fixit-dir COMMAND ${CMAKE_COMMAND} -E remove_directory ${CLANG_TIDY_FIXIT_DIR})
    add_dependencies(tidy-make-fixit-dir tidy-rm-fixit-dir)
    add_dependencies(tidy-base tidy-make-fixit-dir)
    if(CLANG_TIDY_CACHE_SIZE GREATER 0)
        add_custom_target(tidy-create-cache-dir COMMAND ${CMAKE_COMMAND} -E make_directory ${CLANG_TIDY_CACHE})
        add_dependencies(tidy-base tidy-create-cache-dir)
    endif()
endmacro()

function(rocm_clang_tidy_check TARGET)
    get_target_property(SOURCES ${TARGET} SOURCES)
    # TODO: Use generator expressions instead COMMAND ${CLANG_TIDY_COMMAND} $<TARGET_PROPERTY:${TARGET},SOURCES> COMMAND
    # ${CLANG_TIDY_COMMAND} $<JOIN:$<TARGET_PROPERTY:${TARGET},SOURCES>, >
    add_custom_target(tidy-target-${TARGET})
    foreach(SOURCE ${SOURCES})
        if(NOT "${SOURCE}" MATCHES "(h|hpp|hxx)$")
            string(MAKE_C_IDENTIFIER "${SOURCE}" tidy_file)
            set(tidy_target tidy-target-${TARGET}-${tidy_file})
            if(CLANG_TIDY_CACHE_SIZE GREATER 0)
                get_filename_component(SRC_ABS ${SOURCE} ABSOLUTE)
                string(FIND ${SRC_ABS} ${CMAKE_CURRENT_BINARY_DIR} BINARY_IDX)
                if(BINARY_IDX EQUAL -1)
                    set(ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
                else()
                    set(ROOT_DIR ${CMAKE_CURRENT_BINARY_DIR})
                endif()
                get_filename_component(SRC_PATH ${SRC_ABS} DIRECTORY)
                file(RELATIVE_PATH REL_PATH ${ROOT_DIR} ${SRC_PATH})
                get_filename_component(BASE_SOURCE_NAME ${SOURCE} NAME_WE)
                if(REL_PATH)
                    set(BASE_SOURCE ${REL_PATH}/${BASE_SOURCE_NAME})
                else()
                    set(BASE_SOURCE ${BASE_SOURCE_NAME})
                endif()
                file(
                    WRITE ${CMAKE_CURRENT_BINARY_DIR}/${tidy_target}.cmake
                    "
                    set(CLANG_TIDY_COMMAND_LIST \"${CLANG_TIDY_COMMAND}\")
                    set(GH_ANNOTATIONS ${ROCM_ENABLE_GH_ANNOTATIONS})
                    execute_process(
                        COMMAND ${CMAKE_COMMAND} --build ${CMAKE_CURRENT_BINARY_DIR} --target ${BASE_SOURCE}.i
                        ERROR_QUIET
                        OUTPUT_VARIABLE PP_OUT
                        RESULT_VARIABLE RESULT1)
                    if(NOT RESULT1 EQUAL 0)
                        message(WARNING \"Could not preprocess ${SOURCE} -> ${BASE_SOURCE}.i\")
                        execute_process(
                            COMMAND
                                \${CLANG_TIDY_COMMAND_LIST}
                                ${SOURCE}
                                \"-export-fixes=${CLANG_TIDY_FIXIT_DIR}/${TARGET}-${tidy_file}.yaml\"
                            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                            RESULT_VARIABLE RESULT2)
                        if(NOT RESULT2 EQUAL 0)
                            message(FATAL_ERROR \"Clang tidy failed. \")
                        endif()
                        return()
                    endif()
                    string(REPLACE \"Preprocessing CXX source to \" \"\" PP_FILE \"\${PP_OUT}\")
                    string(STRIP \"\${PP_FILE}\" PP_FILE)
                    file(MD5 ${CMAKE_CURRENT_BINARY_DIR}/\${PP_FILE} PP_HASH)
                    execute_process(
                        COMMAND \${CLANG_TIDY_COMMAND_LIST} ${SOURCE} --dump-config
                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                        OUTPUT_VARIABLE TIDY_CONFIG)
                    string(MD5 CONFIG_HASH \"
                        \${TIDY_CONFIG}
                        ${CLANG_TIDY_EXTRA_ARGS}
                        ${CLANG_TIDY_COMPILER_VERSION_OUTPUT}\")
                    set(HASH \${PP_HASH}-\${CONFIG_HASH})
                    set(HASHES \${HASH})
                    set(HASH_FILE ${CLANG_TIDY_CACHE}/${TARGET}-${tidy_file})
                    set(RUN_TIDY On)
                    if(EXISTS \${HASH_FILE})
                        file(STRINGS \${HASH_FILE} CACHED_HASHES)
                        list(FIND CACHED_HASHES \${HASH} HASH_IDX)
                        if(NOT HASH_IDX EQUAL -1)
                            set(RUN_TIDY Off)
                            list(REMOVE_AT CACHED_HASHES HASH_IDX)
                        endif()
                        list(LENGTH CACHED_HASHES NHASHES)
                        math(EXPR NHASHES \"\${NHASHES} - 1\")
                        if(NHASHES GREATER_EQUAL ${CLANG_TIDY_CACHE_SIZE})
                            foreach(IDX RANGE ${CLANG_TIDY_CACHE_SIZE} \${NHASHES})
                                list(REMOVE_AT CACHED_HASHES \${IDX})
                            endforeach()
                        endif()
                        list(APPEND HASHES \${CACHED_HASHES})
                    endif()
                    if(RUN_TIDY)
                        execute_process(
                            COMMAND
                                \${CLANG_TIDY_COMMAND_LIST} ${SOURCE}
                                \"-export-fixes=${CLANG_TIDY_FIXIT_DIR}/${TARGET}-${tidy_file}.yaml\"
                            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                            RESULT_VARIABLE RESULT3
                            OUTPUT_VARIABLE TIDY_OUTPUT
                            ERROR_VARIABLE TIDY_OUTPUT)
                        if(GH_ANNOTATIONS)
                            string(REGEX REPLACE
                                \"(/[^:\\t\\r\\n]+):([0-9]+):([0-9]+): (error|warning): ([^]]+])\"
                                \"::warning file=\\\\1,line=\\\\2,col=\\\\3::\\\\5\"
                                TIDY_OUTPUT
                                \"\${TIDY_OUTPUT}\")
                            string(REPLACE \"${CMAKE_SOURCE_DIR}/\" \"\" TIDY_OUTPUT \"\${TIDY_OUTPUT}\")
                        endif()
                        message(\"\${TIDY_OUTPUT}\")
                        if(RESULT3 EQUAL 0)
                            string(REPLACE \";\" \"\\n\" HASH_LINES \"\${HASHES}\")
                            file(WRITE \${HASH_FILE} \"\${HASH_LINES}\")
                        else()
                            message(FATAL_ERROR \"Clang tidy failed. \")
                        endif()
                    endif()
                ")
                add_custom_target(
                    ${tidy_target}
                    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/${tidy_target}.cmake
                    COMMENT "clang-tidy: Running clang-tidy on target ${SOURCE}...")
            else()
                add_custom_target(
                    ${tidy_target}
                    COMMAND ${CLANG_TIDY_COMMAND} ${SOURCE}
                            "-export-fixes=${CLANG_TIDY_FIXIT_DIR}/${TARGET}-${tidy_file}.yaml"
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                    COMMENT "clang-tidy: Running clang-tidy on target ${SOURCE}...")
            endif()
            if(CLANG_TIDY_DEPEND_ON_TARGET)
                add_dependencies(${tidy_target} ${TARGET})
            endif()
            add_dependencies(${tidy_target} tidy-base)
            add_dependencies(tidy-target-${TARGET} ${tidy_target})
            add_dependencies(tidy ${tidy_target})
        endif()
    endforeach()
endfunction()
