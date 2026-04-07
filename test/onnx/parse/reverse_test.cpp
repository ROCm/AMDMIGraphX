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

TEST_CASE(reverse_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    constexpr int batch_axis = 0;
    constexpr int time_axis  = 1;

    migraphx::shape sx{migraphx::shape::float_type, {2, 4}};
    auto input = mm->add_parameter("x", sx);

    auto add_slice = [&mm, &input](int b, int t_start, int t_end) {
        return mm->add_instruction(
            migraphx::make_op("slice",
                              {{"axes", {batch_axis, time_axis}},
                               {"starts", {b, t_start}},
                               {"ends", {b + 1, t_end}}}),
            input);
    };

    auto s0 = add_slice(0, 0, 4);
    s0      = mm->add_instruction(migraphx::make_op("reverse", {{"axes", {time_axis}}}), s0);
    auto s1 = add_slice(1, 0, 4);
    s1      = mm->add_instruction(migraphx::make_op("reverse", {{"axes", {time_axis}}}), s1);
    auto ret = mm->add_instruction(migraphx::make_op("concat", {{"axis", batch_axis}}), s0, s1);
    mm->add_return({ret});

    auto prog = read_onnx("reverse_test.onnx");
    EXPECT(p == prog);
}
