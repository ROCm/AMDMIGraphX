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

#include <op_builder_test_utils.hpp>

TEST_CASE(celu_happy_path_op_builder_test)
{
    migraphx::module mm;

    const float alpha       = 0.8;
    const migraphx::shape s = {migraphx::shape::float_type, {3}};

    auto x = mm.add_parameter("x", s);

    const auto& input_lens = s.lens();
    const auto& input_type = s.type();
    auto zero_lit =
        mm.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", input_lens}}),
                           mm.add_literal(migraphx::literal{migraphx::shape{input_type}, {0.}}));
    auto one_lit =
        mm.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", input_lens}}),
                           mm.add_literal(migraphx::literal{migraphx::shape{input_type}, {1.}}));
    auto alpha_lit =
        mm.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", input_lens}}),
                           mm.add_literal(migraphx::literal{migraphx::shape{input_type}, {alpha}}));
    auto linear_part = mm.add_instruction(migraphx::make_op("max"), zero_lit, x);
    auto divi        = mm.add_instruction(migraphx::make_op("div"), x, alpha_lit);
    auto expo        = mm.add_instruction(migraphx::make_op("exp"), divi);
    auto sub         = mm.add_instruction(migraphx::make_op("sub"), expo, one_lit);
    auto mul         = mm.add_instruction(migraphx::make_op("mul"), alpha_lit, sub);
    auto exp_part    = mm.add_instruction(migraphx::make_op("min"), zero_lit, mul);
    mm.add_instruction(migraphx::make_op("add"), linear_part, exp_part);

    EXPECT(mm == make_op_module("celu", {{"alpha", alpha}}, mm.get_parameters()));
}

TEST_CASE(celu_zero_alpha_op_builder_test)
{
    migraphx::module mm;

    const float alpha = 0.0f;

    EXPECT(
        test::throws<migraphx::exception>([&] { make_op_module("celu", {{"alpha", alpha}}, {}); },
                                          "alpha is zero (division by zero)"));
}

TEST_CASE(celu_wrong_shape_type_op_builder_test)
{
    migraphx::module mm;
    const float alpha       = 0.8;
    const migraphx::shape s = {migraphx::shape::int8_type, {3}};

    mm.add_parameter("x", s);

    EXPECT(test::throws<migraphx::exception>(
        [&] { make_op_module("celu", {{"alpha", alpha}}, mm.get_parameters()); },
        "input tensor not float type"));
}
