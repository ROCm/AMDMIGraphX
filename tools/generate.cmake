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

find_program(CLANG_FORMAT clang-format PATHS /opt/rocm/llvm $ENV{HIP_PATH})
if(NOT CLANG_FORMAT)
    message(WARNING "clang-format not found - skipping 'generate' target!")
    return()
endif()

cmake_path(NATIVE_PATH CLANG_FORMAT NORMALIZE __clang_format)
if(WIN32)
    set(__clang_format "C:${__clang_format}")
endif()

find_package(Python 3 COMPONENTS Interpreter)
if(NOT Python_EXECUTABLE)
    message(WARNING "Python 3 interpreter not found - skipping 'generate' target!")
    return()
endif()

cmake_path(SET SRC_DIR NORMALIZE ${CMAKE_CURRENT_LIST_DIR}/../src)

file(GLOB __files ${CMAKE_CURRENT_LIST_DIR}/include/*.hpp)
foreach(__file ${__files})
    cmake_path(NATIVE_PATH __file NORMALIZE __input_native_path)
    cmake_path(GET __file FILENAME __file_name)
    cmake_path(SET __output_path "${SRC_DIR}/include/migraphx/${__file_name}")
    cmake_path(NATIVE_PATH __output_path __output_native_path)
    message("Generating ${__output_native_path}")
    execute_process(
            COMMAND ${Python_EXECUTABLE} te.py ${__input_native_path} | ${__clang_format} -style=file
            OUTPUT_FILE ${__output_native_path}
            ERROR_VARIABLE __error_code
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
    if(__error_code)
        message(WARNING "${__error_code}")
    endif()
endforeach()

function(generate_api __input_path __output_path)
    cmake_path(NATIVE_PATH __output_path NORMALIZE __output_native_path)
    cmake_path(NATIVE_PATH __input_path NORMALIZE __input_native_path)
    cmake_path(SET __migraphx_py_path "${SRC_DIR}/api/migraphx.py")
    cmake_path(NATIVE_PATH __migraphx_py_path __migraphx_py_native_path)
    cmake_path(SET __api_py_path "${CMAKE_CURRENT_LIST_DIR}/api.py")
    cmake_path(NATIVE_PATH __api_py_path __api_py_native_path)
    message("Generating ${__output_native_path}")
    execute_process(
            COMMAND ${Python_EXECUTABLE} ${__api_py_native_path} ${__migraphx_py_native_path} ${__input_native_path} | ${__clang_format} -style=file
            OUTPUT_FILE ${__output_native_path}
            ERROR_VARIABLE __error_code
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
    if(__error_code)
        message(WARNING "${__error_code}")
    endif()
endfunction()

generate_api("${CMAKE_CURRENT_LIST_DIR}/api/migraphx.h" "${SRC_DIR}/api/include/migraphx/migraphx.h")
generate_api("${CMAKE_CURRENT_LIST_DIR}/api/api.cpp" "${SRC_DIR}/api/api.cpp")
