/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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

TEST_CASE(biasadd_test)
{
    migraphx::program p;
    auto* mm  = p.get_main_module();
    auto x    = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {2, 3, 4}});
    auto bias = mm->add_parameter("bias", migraphx::shape{migraphx::shape::float_type, {4}});
    auto skip = mm->add_parameter("skip", migraphx::shape{migraphx::shape::float_type, {2, 3, 4}});

    auto x_plus_bias = add_common_op(*mm, migraphx::make_op("add"), {x, bias});
    auto ret         = add_common_op(*mm, migraphx::make_op("add"), {x_plus_bias, skip});
    mm->add_return({ret});

    auto prog = read_onnx("biasadd_test.onnx");

    EXPECT(p == prog);
}
