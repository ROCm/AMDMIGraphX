/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
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

TEST_CASE(conv_dynamic_bias_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x0 =
        mm->add_parameter("0", {migraphx::shape::float_type, {{1, 6}, {3, 3}, {32, 32}, {32, 32}}});
    auto x1 = mm->add_parameter("1", {migraphx::shape::float_type, {1, 3, 5, 5}});
    auto x2 = mm->add_parameter("2", {migraphx::shape::float_type, {1}});
    auto x3 = mm->add_instruction(migraphx::make_op("convolution"), x0, x1);
    auto x4 = mm->add_instruction(migraphx::make_op("broadcast", {{"axis", 1}}), x2, x3);
    auto x5 = mm->add_instruction(migraphx::make_op("add"), x3, x4);
    mm->add_return({x5});

    migraphx::onnx_options options;
    options.default_dyn_dim_value = {1, 6};
    auto prog                     = read_onnx("conv_dynamic_bias_test.onnx", options);
    EXPECT(p == prog);
}
