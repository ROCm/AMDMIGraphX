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
project(migraphx-doc)
find_package(ROCM REQUIRED)

include(ROCMDoxygenDoc)

set(DOXYGEN_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/doxygen)
rocm_add_doxygen_doc(
    OUTPUT_DIRECTORY ${DOXYGEN_OUTPUT}
    INPUT 
        ${CMAKE_SOURCE_DIR}/src
    INCLUDE_PATH
        ${CMAKE_SOURCE_DIR}/src/include
        ${CMAKE_SOURCE_DIR}/src/targets/cpu/include
        ${CMAKE_SOURCE_DIR}/src/targets/gpu/include
    STRIP_FROM_INC_PATH
        ${CMAKE_SOURCE_DIR}/src/include
        ${CMAKE_SOURCE_DIR}/src/targets/cpu/include
        ${CMAKE_SOURCE_DIR}/src/targets/gpu/include
    EXCLUDE_PATTERNS
        ${CMAKE_SOURCE_DIR}/src/targets/gpu/kernels
        ${CMAKE_SOURCE_DIR}/src/targets/gpu/device
    SEARCH_INCLUDES YES
    MACRO_EXPANSION YES
    RECURSIVE YES
    GENERATE_XML YES
    GENERATE_LATEX YES
    USE_PDFLATEX YES
    CALL_GRAPH YES
    CALLER_GRAPH YES
    BUILTIN_STL_SUPPORT YES
    PROJECT_NAME MIGraphX
    SORT_MEMBERS_CTORS_1ST YES
    SOURCE_BROWSER YES
    GENERATE_TREEVIEW YES
    REFERENCED_BY_RELATION YES
    REFERENCES_RELATION YES
    REFERENCES_LINK_SOURCE YES
    EXTRACT_ALL YES
    ENUM_VALUES_PER_LINE 1
    FULL_PATH_NAMES YES
    WARN_LOGFILE "${DOXYGEN_OUTPUT}/DoxygenWarningLog.txt"
    PREDEFINED DOXYGEN
)

include(ROCMSphinxDoc)
rocm_add_sphinx_doc(src 
    BUILDER html 
    OUTPUT_DIR html
    VARS 
        breathe_projects.proj=${DOXYGEN_OUTPUT}/xml
        breathe_default_project=proj
    DEPENDS doxygen
)

find_package(LATEX)
if(LATEX_FOUND)
    rocm_add_sphinx_doc(src 
        BUILDER latex
        OUTPUT_DIR pdf
        VARS 
            breathe_projects.proj=${DOXYGEN_OUTPUT}/xml
            breathe_default_project=proj
        DEPENDS doxygen
    )
else()
    message("Latex builder not found. Latex builder is required only for building the PDF documentation for MIGraphX and is not necessary for building the library, or any other components. To build PDF documentation run make in ${CMAKE_CURRENT_SOURCE_DIR}/pdf, once a latex builder is installed.")
endif()

