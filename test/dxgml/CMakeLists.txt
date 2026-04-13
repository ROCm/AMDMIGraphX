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

function(add_dxgml_test TEST_NAME)
    rocm_add_test_executable(${TEST_NAME} ${ARGN})
    rocm_clang_tidy_check(${TEST_NAME})
    target_link_libraries(${TEST_NAME} amdxgml migraphx_ref dxgml_files)
    target_include_directories(${TEST_NAME} PUBLIC ../include include)
endfunction()

# Embed only the small test-fixture .mlir files used by the C++ parse unit tests.
# Large model files (resources.mlir, LLM weights, etc.) are excluded — they are
# read directly from disk by the driver tests in run_dxgml_tests.bat.
set(DXGML_FIXTURE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/mlir/ConvRelu.CompilationInput.mlir
    ${CMAKE_CURRENT_SOURCE_DIR}/mlir/Gelu.CompilationInput.mlir
    ${CMAKE_CURRENT_SOURCE_DIR}/mlir/ReluErf.CompilationInput.mlir
    ${CMAKE_CURRENT_SOURCE_DIR}/mlir/StandaloneCluster.CompilationInput.mlir
)
add_embed_library(dxgml_files ${DXGML_FIXTURE_FILES} RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/mlir)

file(GLOB DXGML_PARSE_TESTS CONFIGURE_DEPENDS parse/*.cpp)
foreach(TEST_SRC ${DXGML_PARSE_TESTS})
    get_filename_component(BASE_NAME ${TEST_SRC} NAME_WE)
    add_dxgml_test(test_dxgml_${BASE_NAME} ${TEST_SRC})
endforeach()
