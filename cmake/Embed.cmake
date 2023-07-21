#####################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
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
find_program(EMBED_LD ld)
find_program(EMBED_OBJCOPY objcopy)

include(CMakeDependentOption)
cmake_dependent_option(EMBED_USE_LD "Use ld to embed data files" ON "NOT WIN32" OFF)

function(wrap_string)
    set(options)
    set(oneValueArgs VARIABLE AT_COLUMN)
    set(multiValueArgs)
    cmake_parse_arguments(PARSE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    cmake_parse_arguments(WRAP_STRING "${options}" "${oneValueArgs}" "" ${ARGN})

    string(LENGTH ${${PARSE_VARIABLE}} string_length)
    math(EXPR offset "0")

    while(string_length GREATER 0)

        if(string_length GREATER ${PARSE_AT_COLUMN})
            math(EXPR length "${PARSE_AT_COLUMN}")
        else()
            math(EXPR length "${string_length}")
        endif()

        string(SUBSTRING ${${PARSE_VARIABLE}} ${offset} ${length} line)
        set(lines "${lines}\n${line}")

        math(EXPR string_length "${string_length} - ${length}")
        math(EXPR offset "${offset} + ${length}")
    endwhile()

    set(${PARSE_VARIABLE} "${lines}" PARENT_SCOPE)
endfunction()

function(generate_embed_source EMBED_NAME)
    set(options)
    set(oneValueArgs SRC HEADER)
    set(multiValueArgs OBJECTS SYMBOLS)

    cmake_parse_arguments(PARSE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(EXTERNS)
    set(INIT_KERNELS)

    list(LENGTH PARSE_SYMBOLS SYMBOLS_LEN)
    list(LENGTH PARSE_OBJECTS OBJECTS_LEN)
    if(NOT ${SYMBOLS_LEN} EQUAL ${OBJECTS_LEN})
        message(FATAL_ERROR "Symbols and objects dont match: ${SYMBOLS_LEN} != ${OBJECTS_LEN}")
    endif()
    math(EXPR LEN "${SYMBOLS_LEN} - 1")

    foreach(idx RANGE ${LEN})
        list(GET PARSE_SYMBOLS ${idx} SYMBOL)
        list(GET PARSE_OBJECTS ${idx} OBJECT)
        set(START_SYMBOL "_binary_${SYMBOL}_start")
        set(LENGTH_SYMBOL "_binary_${SYMBOL}_length")
        if(EMBED_USE_LD)
            string(APPEND EXTERNS "
                extern const char ${START_SYMBOL}[];
                extern const size_t _binary_${SYMBOL}_size;
                const auto ${LENGTH_SYMBOL} = reinterpret_cast<size_t>(&_binary_${SYMBOL}_size);
            ")
        else()
            string(APPEND EXTERNS "
                extern const char ${START_SYMBOL}[];
                extern const size_t ${LENGTH_SYMBOL};
            ")
        endif()

        # TODO: Should use NAME_WLE
        get_filename_component(BASE_NAME "${OBJECT}" NAME)
        string(REGEX REPLACE ".[A-Za-z0-9_]+$" "" BASE_NAME ${BASE_NAME})

        string(APPEND INIT_KERNELS "
            { \"migraphx/kernels/${BASE_NAME}\", { ${START_SYMBOL}, ${LENGTH_SYMBOL} } },
        ")
    endforeach()

    file(WRITE "${PARSE_HEADER}" "
#include <utility>
#include <string_view>
#include <vector>
std::vector<std::pair<std::string_view, std::string_view>> ${EMBED_NAME}();
")

    file(WRITE "${PARSE_SRC}" "
#include <${EMBED_NAME}.hpp>
${EXTERNS}
std::vector<std::pair<std::string_view, std::string_view>> ${EMBED_NAME}()
{
    static std::vector<std::pair<std::string_view, std::string_view>> result = {${INIT_KERNELS}};
    return result;
}
")
endfunction()

function(embed_file OUTPUT_FILE OUTPUT_SYMBOL FILE)
    set(WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    # Glob is used to compute the relative path
    file(GLOB FILES RELATIVE ${WORKING_DIRECTORY} ${FILE})
    foreach(REL_FILE ${FILES})
        string(MAKE_C_IDENTIFIER "${REL_FILE}" SYMBOL)
        get_filename_component(OUTPUT_FILE_DIR "${REL_FILE}" DIRECTORY)
        file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${OUTPUT_FILE_DIR}")
        if(EMBED_USE_LD)
            set(OUT_FILE "${CMAKE_CURRENT_BINARY_DIR}/${REL_FILE}.o")
        else()
            set(OUT_FILE "${CMAKE_CURRENT_BINARY_DIR}/${REL_FILE}.cpp")
        endif()
        set(${OUTPUT_SYMBOL} ${SYMBOL} PARENT_SCOPE)
        set(${OUTPUT_FILE} "${OUT_FILE}" PARENT_SCOPE)
        if(EMBED_USE_LD)
            add_custom_command(
                OUTPUT "${OUT_FILE}"
                COMMAND ${EMBED_LD} -r -o "${OUT_FILE}" -z noexecstack --format=binary "${REL_FILE}" 
                COMMAND ${EMBED_OBJCOPY} --rename-section .data=.rodata,alloc,load,readonly,data,contents "${OUT_FILE}"
                WORKING_DIRECTORY ${WORKING_DIRECTORY}
                DEPENDS ${FILE}
                VERBATIM
            )
        else()
            set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${FILE})
            # reads source file contents as hex string
            file(READ ${FILE} HEX_STRING HEX)
            # wraps the hex string into multiple lines
            wrap_string(VARIABLE HEX_STRING AT_COLUMN 80)
            # adds '0x' prefix and comma suffix before and after every byte respectively
            string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1, " ARRAY_VALUES ${HEX_STRING})
            # removes trailing comma
            string(REGEX REPLACE ", $" "" ARRAY_VALUES ${ARRAY_VALUES})
            file(WRITE "${OUT_FILE}" "
                extern const char _binary_${SYMBOL}_start[] = { ${ARRAY_VALUES} };
                extern const size_t _binary_${SYMBOL}_length = sizeof(_binary_${SYMBOL}_start);
            \n")
        endif()
    endforeach()
endfunction()

function(add_embed_library EMBED_NAME)
    set(EMBED_DIR ${CMAKE_CURRENT_BINARY_DIR}/embed/${EMBED_NAME})
    file(MAKE_DIRECTORY ${EMBED_DIR})
    set(SRC_FILE "${EMBED_DIR}/${EMBED_NAME}.cpp")
    set(HEADER_FILE "${EMBED_DIR}/include/${EMBED_NAME}.hpp")
    message(STATUS "Embedding kernel files: ${ARGN}")
    foreach(FILE ${ARGN})
        embed_file(OUTPUT_FILE OUTPUT_SYMBOL ${FILE})
        list(APPEND OUTPUT_FILES ${OUTPUT_FILE})
        list(APPEND SYMBOLS ${OUTPUT_SYMBOL})
    endforeach()
    message(STATUS "Generating embedding library ${EMBED_NAME}")
    generate_embed_source(${EMBED_NAME} SRC ${SRC_FILE} HEADER ${HEADER_FILE} OBJECTS ${OUTPUT_FILES} SYMBOLS ${SYMBOLS})
    add_library(${EMBED_NAME} STATIC ${OUTPUT_FILES} "${SRC_FILE}")
    target_include_directories(${EMBED_NAME} PUBLIC "${EMBED_DIR}/include")
    target_compile_options(${EMBED_NAME} PRIVATE -Wno-reserved-identifier -Wno-extern-initializer -Wno-missing-variable-declarations)
    set_target_properties(${EMBED_NAME} PROPERTIES POSITION_INDEPENDENT_CODE On)
endfunction()
