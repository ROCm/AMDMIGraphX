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

# Stub for rocm_add_version_resource until rocm-cmake ships one.
# Embeds a Win32 VERSIONINFO resource into the target on Windows; no-op elsewhere.
#
# Usage:
#   rocm_add_version_resource(<target> <product_name> <description>)

function(rocm_add_version_resource TARGET_NAME PRODUCT_NAME DESCRIPTION)
    if(NOT WIN32)
        return()
    endif()

    get_target_property(TARGET_TYPE ${TARGET_NAME} TYPE)

    get_target_property(OUTPUT_NAME ${TARGET_NAME} OUTPUT_NAME)
    if(NOT OUTPUT_NAME)
        set(OUTPUT_NAME "${TARGET_NAME}")
    endif()

    if(TARGET_TYPE STREQUAL "SHARED_LIBRARY")
        set(ORIGINAL_FILENAME "${OUTPUT_NAME}.dll")
        set(DLL_BUILD_DEF "#define DLL_BUILD")
    elseif(TARGET_TYPE STREQUAL "EXECUTABLE")
        set(ORIGINAL_FILENAME "${OUTPUT_NAME}.exe")
        set(DLL_BUILD_DEF "")
    else()
        set(ORIGINAL_FILENAME "${OUTPUT_NAME}")
        set(DLL_BUILD_DEF "")
    endif()

    set(RC_FILE "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}_version.rc")

    file(WRITE "${RC_FILE}"
"#include <winver.h>

${DLL_BUILD_DEF}

#define VER_FILEVERSION             ${PROJECT_VERSION_MAJOR},${PROJECT_VERSION_MINOR},${PROJECT_VERSION_PATCH},0
#define VER_FILEVERSION_STR         \"${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}.0\\0\"

#define VER_PRODUCTVERSION          ${PROJECT_VERSION_MAJOR},${PROJECT_VERSION_MINOR},${PROJECT_VERSION_PATCH},0
#define VER_PRODUCTVERSION_STR      \"${PROJECT_VERSION}\\0\"

#define VER_COMPANYNAME_STR         \"Advanced Micro Devices, Inc.\\0\"
#define VER_LEGALCOPYRIGHT_STR      \"Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.\\0\"
#define VER_PRODUCTNAME_STR         \"${PRODUCT_NAME} ${ORIGINAL_FILENAME}\\0\"
#define VER_FILEDESCRIPTION_STR     \"${DESCRIPTION}\\0\"
#define VER_INTERNALNAME_STR        \"${TARGET_NAME}\\0\"
#define VER_ORIGINALFILENAME_STR    \"${ORIGINAL_FILENAME}\\0\"

#ifdef _DEBUG
#define VER_DEBUG                   VS_FF_DEBUG
#else
#define VER_DEBUG                   0
#endif

VS_VERSION_INFO VERSIONINFO
FILEVERSION     VER_FILEVERSION
PRODUCTVERSION  VER_PRODUCTVERSION
FILEFLAGSMASK   VS_FFI_FILEFLAGSMASK
FILEFLAGS       VER_DEBUG
FILEOS          VOS_NT_WINDOWS32
#ifdef DLL_BUILD
FILETYPE        VFT_DLL
#else
FILETYPE        VFT_APP
#endif
FILESUBTYPE     VFT2_UNKNOWN
BEGIN
    BLOCK \"StringFileInfo\"
    BEGIN
        BLOCK \"040904B0\"
        BEGIN
            VALUE \"CompanyName\",      VER_COMPANYNAME_STR
            VALUE \"FileDescription\",  VER_FILEDESCRIPTION_STR
            VALUE \"FileVersion\",      VER_FILEVERSION_STR
            VALUE \"InternalName\",     VER_INTERNALNAME_STR
            VALUE \"LegalCopyright\",   VER_LEGALCOPYRIGHT_STR
            VALUE \"OriginalFilename\", VER_ORIGINALFILENAME_STR
            VALUE \"ProductName\",      VER_PRODUCTNAME_STR
            VALUE \"ProductVersion\",   VER_PRODUCTVERSION_STR
        END
    END
    BLOCK \"VarFileInfo\"
    BEGIN
        VALUE \"Translation\", 0x409, 1200
    END
END
")

    target_sources(${TARGET_NAME} PRIVATE "${RC_FILE}")

    message(STATUS "Added version resource to ${TARGET_NAME}: ${DESCRIPTION}")
endfunction()
