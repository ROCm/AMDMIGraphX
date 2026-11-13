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

TEST_CASE(range_bounded_limit_test)
{
    using migraphx::sym::var;
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto half =
        mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::float_type}, {0.5f}});
    mm->add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {0}});
    auto start =
        mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::int64_type}, {0}});
    auto delta =
        mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::int64_type}, {1}});
    auto x = mm->add_parameter("x", {migraphx::shape::float_type, {8}});

    half = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {8}}}), half);
    auto above = mm->add_instruction(migraphx::make_op("greater"), x, half);
    above      = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::bool_type}}), above);
    above = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::int64_type}}), above);
    auto count = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {0}}}), above);
    count      = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), count);
    // The count is at most 8, so the length is a symbol over [0, 8]
    const migraphx::shape::dynamic_dimension length{var("main_Range_11", {0, 8})};
    auto ret = mm->add_instruction(
        migraphx::make_op("dynamic_range",
                          {{"max_output", 8}, {"output_dim", migraphx::to_value(length)}}),
        start,
        count,
        delta);
    mm->add_return({ret});

    EXPECT(p == read_onnx("range_bounded_limit_test.onnx"));
}
