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

TEST_CASE(pad_reflect_multiaxis_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 3}});
    mm->add_literal({migraphx::shape{migraphx::shape::int32_type, {4}}, {0, 2, 2, 0}});
    auto l1 = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {0, 1}}, {"starts", {0, 1}}, {"ends", {2, 2}}}), l0);
    auto l2 = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {0, 1}}, {"starts", {0, 2}}, {"ends", {2, 3}}}), l0);
    auto l3 = mm->add_instruction(migraphx::make_op("concat", {{"axis", 1}}), l2, l1, l0);
    auto l4 = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {0, 1}}, {"starts", {0, 0}}, {"ends", {1, 5}}}), l3);
    auto l5 = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {0, 1}}, {"starts", {1, 0}}, {"ends", {2, 5}}}), l3);
    auto r = mm->add_instruction(migraphx::make_op("concat", {{"axis", 0}}), l3, l4, l5);
    mm->add_return({r});

    auto prog = read_onnx("pad_reflect_multiaxis_test.onnx");

    EXPECT(p == prog);
}
