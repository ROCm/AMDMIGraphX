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

TEST_CASE(reducel1_variable_axes_test)
{
    using namespace migraphx;

    program p;
    auto* mm   = p.get_main_module();
    auto x     = mm->add_parameter("x", shape{shape::float_type, {3, 4, 5, 6}});
    auto axes  = mm->add_parameter("axes", shape{shape::int64_type, {1}});
    auto abs_x = mm->add_instruction(make_op("abs"), x);
    mm->add_instruction(make_op("reduce_sum", {{"axes", {}}}), abs_x, axes);

    auto prog = optimize_onnx("reducel1_variable_axes_test.onnx");
    EXPECT(p == prog);
}
