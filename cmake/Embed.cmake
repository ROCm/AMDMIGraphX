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
    cmake_parse_arguments(PARSE "" "VARIABLE;AT_COLUMN" "" ${ARGN})

    string(LENGTH "${${PARSE_VARIABLE}}" string_length)
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

function(generate_embed_source EMBED_NAME SRC_FILE HEADER_FILE BASE_DIRECTORY)
    cmake_parse_arguments(PARSE "" "" "SYMBOLS;FILES" ${ARGN})
    foreach(SYMBOL FILE IN ZIP_LISTS PARSE_SYMBOLS PARSE_FILES)
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
        cmake_path(RELATIVE_PATH FILE BASE_DIRECTORY ${BASE_DIRECTORY} OUTPUT_VARIABLE BASE_NAME)
        string(APPEND INIT_KERNELS "
            { \"${BASE_NAME}\", { ${START_SYMBOL}, ${LENGTH_SYMBOL}} },")
    endforeach()

    file(WRITE "${HEADER_FILE}" "
#include <string_view>
#include <unordered_map>
#include <utility>
std::unordered_map<std::string_view, std::string_view> ${EMBED_NAME}();
")

    file(WRITE "${SRC_FILE}" "
#include <${EMBED_NAME}.hpp>
${EXTERNS}
std::unordered_map<std::string_view, std::string_view> ${EMBED_NAME}()
{
    static std::unordered_map<std::string_view, std::string_view> result = {${INIT_KERNELS}};
    return result;
}
")
endfunction()

function(embed_file FILE BASE_DIRECTORY)
    message(STATUS "    ${FILE}")
    cmake_path(RELATIVE_PATH FILE BASE_DIRECTORY ${BASE_DIRECTORY} OUTPUT_VARIABLE REL_FILE)
    string(MAKE_C_IDENTIFIER "${REL_FILE}" OUTPUT_SYMBOL)
    get_filename_component(OUTPUT_FILE_DIR "${REL_FILE}" DIRECTORY)
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${OUTPUT_FILE_DIR}")
    if(EMBED_USE_LD)
        set(OUTPUT_FILE "${CMAKE_CURRENT_BINARY_DIR}/${REL_FILE}.o")
        add_custom_command(
            OUTPUT ${OUTPUT_FILE}
            COMMAND ${EMBED_LD} -r -o "${OUTPUT_FILE}" -z noexecstack --format=binary "${REL_FILE}"
            COMMAND ${EMBED_OBJCOPY} --rename-section .data=.rodata,alloc,load,readonly,data,contents "${OUTPUT_FILE}"
            WORKING_DIRECTORY ${BASE_DIRECTORY}
            DEPENDS ${FILE}
            VERBATIM)
    else()
        set(OUTPUT_FILE "${CMAKE_CURRENT_BINARY_DIR}/${REL_FILE}.cpp")
        # reads source file contents as hex string
        file(READ ${FILE} HEX_STRING HEX)
        # wraps the hex string into multiple lines
        wrap_string(VARIABLE HEX_STRING AT_COLUMN 80)
        # adds '0x' prefix and comma suffix before and after every byte respectively
        string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1, " ARRAY_VALUES ${HEX_STRING})
        # removes trailing comma
        string(REGEX REPLACE ", $" "" ARRAY_VALUES ${ARRAY_VALUES})
        file(WRITE "${OUTPUT_FILE}" "
#include <cstddef>
extern const char _binary_${OUTPUT_SYMBOL}_start[] = { ${ARRAY_VALUES} };
extern const size_t _binary_${OUTPUT_SYMBOL}_length = sizeof(_binary_${OUTPUT_SYMBOL}_start);
")
    endif()
    set(OUTPUT_FILE ${OUTPUT_FILE} PARENT_SCOPE)
    set(OUTPUT_SYMBOL ${OUTPUT_SYMBOL} PARENT_SCOPE)
endfunction()

function(add_embed_library EMBED_NAME)
    cmake_parse_arguments(PARSE "" "BASE_DIRECTORY" "" ${ARGN})
    set(EMBED_DIR ${CMAKE_CURRENT_BINARY_DIR}/embed/${EMBED_NAME})
    file(MAKE_DIRECTORY ${EMBED_DIR})
    set(SRC_FILE "${EMBED_DIR}/${EMBED_NAME}.cpp")
    set(HEADER_FILE "${EMBED_DIR}/include/${EMBED_NAME}.hpp")
    message(STATUS "Embedding kernel files:")
    foreach(FILE ${PARSE_UNPARSED_ARGUMENTS})
        embed_file(${FILE} ${PARSE_BASE_DIRECTORY})
        list(APPEND OUTPUT_FILES ${OUTPUT_FILE})
        list(APPEND SYMBOLS ${OUTPUT_SYMBOL})
    endforeach()
    message(STATUS "Generating embedding library '${EMBED_NAME}'")
    generate_embed_source(${EMBED_NAME} ${SRC_FILE} ${HEADER_FILE} "${PARSE_BASE_DIRECTORY}" SYMBOLS ${SYMBOLS} FILES ${PARSE_UNPARSED_ARGUMENTS})
    add_library(object_${EMBED_NAME} OBJECT ${SRC_FILE} ${HEADER_FILE} $<$<NOT:$<BOOL:${EMBED_USE_LD}>>:${OUTPUT_FILES}>)
    target_include_directories(object_${EMBED_NAME} PUBLIC ${EMBED_DIR}/include)
    target_compile_options(object_${EMBED_NAME} PRIVATE
            -Wno-reserved-identifier -Wno-extern-initializer -Wno-missing-variable-declarations)
    set_target_properties(object_${EMBED_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    add_library(${EMBED_NAME} INTERFACE $<TARGET_OBJECTS:object_${EMBED_NAME}> ${OUTPUT_FILES})
    target_link_libraries(${EMBED_NAME} INTERFACE $<BUILD_INTERFACE:
            $<TARGET_OBJECTS:object_${EMBED_NAME}>;$<$<BOOL:${EMBED_USE_LD}>:${OUTPUT_FILES}>>)
    target_include_directories(${EMBED_NAME} INTERFACE $<BUILD_INTERFACE:${EMBED_DIR}/include>)
endfunction()
