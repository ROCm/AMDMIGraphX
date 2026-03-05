/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include <onnx_test.hpp>

template <class T>
struct TD;

TEST_CASE(debug_symbols_onnx_names)
{
    migraphx::program p;
    auto* mm    = p.get_main_module();
    auto l0     = mm->add_parameter("x", {migraphx::shape::float_type, {1, 1, 3, 3}});
    auto l1     = mm->add_parameter("w", {migraphx::shape::float_type, {1, 1, 3, 3}});
    auto conv_t = mm->add_instruction(migraphx::make_op("convolution_backwards"), l0, l1);
    mm->add_debug_symbols(conv_t, {"conv1"});
    mm->add_return({conv_t});

    migraphx::onnx_options options;
    options.use_debug_symbols = true;

    auto prog = read_onnx("conv_transpose_test.onnx", options);
    EXPECT(p == prog);
}

TEST_CASE(debug_symbols_migx_names)
{
    migraphx::program p;
    auto* mm      = p.get_main_module();
    auto l0       = mm->add_parameter("0", {migraphx::shape::float_type, {1, 3, 32, 32}});
    auto l1       = mm->add_parameter("1", {migraphx::shape::float_type, {1, 3, 5, 5}});
    auto l2       = mm->add_parameter("2", {migraphx::shape::float_type, {1}});
    uint64_t axis = 1;
    auto l3       = mm->add_instruction(migraphx::make_op("convolution"), l0, l1);
    mm->add_debug_symbols(l3, {"migx_uid:Conv_3"});
    auto l4 = mm->add_instruction(
        migraphx::make_op("broadcast", {{"axis", axis}, {"out_lens", l3->get_shape().lens()}}), l2);
    mm->add_debug_symbols(l4, {"migx_uid:Conv_3"});
    auto l5 = mm->add_instruction(migraphx::make_op("add"), l3, l4);
    mm->add_debug_symbols(l5, {"migx_uid:Conv_3"});
    mm->add_return({l5});

    migraphx::onnx_options options;
    options.use_debug_symbols = true;
    auto prog                 = read_onnx("conv_bias_test.onnx", options);
    EXPECT(p == prog);
}
