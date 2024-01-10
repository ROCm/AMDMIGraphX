#####################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#
#####################################################################################
rocm_test_link_libraries(tiny_rocm_cxx)
rocm_test_include_directories(include)

function(add_tests PREFIX GLOB)
    file(GLOB TESTS CONFIGURE_DEPENDS ${GLOB})
    foreach(TEST ${TESTS})
        get_filename_component(BASE_NAME ${TEST} NAME_WE)
        rocm_add_test_executable(test_${PREFIX}_${BASE_NAME} ${TEST})
        rocm_clang_tidy_check(test_${PREFIX}_${BASE_NAME})
    endforeach()
endfunction()

add_tests(main *.cpp)
add_tests(type_traits type_traits/*.cpp)

