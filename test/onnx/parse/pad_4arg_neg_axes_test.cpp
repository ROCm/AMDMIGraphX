/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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

TEST_CASE(pad_4arg_neg_axes_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 3, 4, 5}});
    // axes=[-3,-1]
    mm->add_literal({migraphx::shape{migraphx::shape::int32_type, {2}}, {-3, -1}});
    // constant_value=1
    mm->add_literal({migraphx::shape{migraphx::shape::float_type}, {1.0f}});
    // pads=[1,3,2,4]
    mm->add_literal({migraphx::shape{migraphx::shape::int32_type, {4}}, {1, 3, 2, 4}});
    auto r = mm->add_instruction(
        migraphx::make_op("pad", {{"pads", {0, 1, 0, 3, 0, 2, 0, 4}}, {"value", 1.0f}}), l0);
    mm->add_return({r});

    auto prog = read_onnx("pad_4arg_neg_axes_test.onnx");

    EXPECT(p == prog);
}
