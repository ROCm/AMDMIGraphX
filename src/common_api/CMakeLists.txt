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
#####################################################################################


add_executable(MNISTSample mnist_sample.cpp)
rocm_clang_tidy_check(MNISTSample)
target_link_libraries(MNISTSample migraphx_onnx migraphx_ref migraphx_c migraphx migraphx_all_targets)
target_include_directories(MNISTSample PUBLIC ../include include)

add_executable(Network network.cpp)
rocm_clang_tidy_check(Network)
target_link_libraries(Network migraphx_onnx migraphx_ref migraphx_c migraphx migraphx_all_targets)
target_include_directories(Network PUBLIC ../include include)

add_executable(RNNSample rnn_sample.cpp)
rocm_clang_tidy_check(RNNSample)
target_link_libraries(RNNSample migraphx_onnx migraphx_ref migraphx_c migraphx migraphx_all_targets)
target_include_directories(RNNSample PUBLIC ../include include)
