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

#[=======================================================================[
Add Windows version resource to a target (DLL or EXE)

Usage:
  add_version_resource(
    TARGET target_name
    DESCRIPTION "Brief description of the component"
    [FILENAME output_filename]  # Optional, defaults to target output name
  )

Example:
  add_version_resource(
    TARGET migraphx
    DESCRIPTION "MIGraphX Core Library"
  )

This function generates a .rc file with version information and adds it
to the target's sources on Windows. On non-Windows platforms, it does nothing.
#]=======================================================================]

function(add_version_resource)
    # Only process on Windows
    if(NOT WIN32)
        return()
    endif()

    # Parse arguments
    set(options "")
    set(oneValueArgs TARGET DESCRIPTION FILENAME)
    set(multiValueArgs "")
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # Validate required arguments
    if(NOT ARG_TARGET)
        message(FATAL_ERROR "add_version_resource: TARGET argument is required")
    endif()

    if(NOT ARG_DESCRIPTION)
        message(FATAL_ERROR "add_version_resource: DESCRIPTION argument is required for ${ARG_TARGET}")
    endif()

    # Get target properties
    get_target_property(TARGET_TYPE ${ARG_TARGET} TYPE)

    # Set component-specific variables
    set(COMPONENT_NAME ${ARG_TARGET})
    set(COMPONENT_DESCRIPTION "${ARG_DESCRIPTION}")

    # Determine output filename
    if(ARG_FILENAME)
        set(COMPONENT_FILENAME "${ARG_FILENAME}")
    else()
        # Get the actual output name of the target
        get_target_property(OUTPUT_NAME ${ARG_TARGET} OUTPUT_NAME)
        if(OUTPUT_NAME)
            set(COMPONENT_FILENAME "${OUTPUT_NAME}")
        else()
            set(COMPONENT_FILENAME "${ARG_TARGET}")
        endif()

        # Add appropriate extension
        if(TARGET_TYPE STREQUAL "SHARED_LIBRARY")
            set(COMPONENT_FILENAME "${COMPONENT_FILENAME}.dll")
        elseif(TARGET_TYPE STREQUAL "EXECUTABLE")
            set(COMPONENT_FILENAME "${COMPONENT_FILENAME}.exe")
        endif()
    endif()

    # Set DLL_BUILD flag for resource compiler
    if(TARGET_TYPE STREQUAL "SHARED_LIBRARY")
        set(DLL_BUILD_FLAG "-DDLL_BUILD")
    else()
        set(DLL_BUILD_FLAG "")
    endif()

    # Configure the version resource file
    set(RC_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${ARG_TARGET}_version.rc")
    configure_file(
        "${PROJECT_SOURCE_DIR}/src/migraphx_version.rc.in"
        "${RC_OUTPUT}"
        @ONLY
    )

    # Add the resource file to the target
    target_sources(${ARG_TARGET} PRIVATE ${RC_OUTPUT})

    # Set resource compiler flags if needed
    if(DLL_BUILD_FLAG)
        set_source_files_properties(${RC_OUTPUT} PROPERTIES
            COMPILE_FLAGS ${DLL_BUILD_FLAG}
        )
    endif()

    message(STATUS "Added version resource to ${ARG_TARGET}: ${COMPONENT_DESCRIPTION}")
endfunction()
