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

#include <migraphx/common.hpp>
#include <op_builder_test_utils.hpp>

TEST_CASE(gelu_quick_happy_path_op_builder_test)
{
    migraphx::module mm;
    const float alpha_val = 0.5f;

    auto type      = migraphx::shape::float_type;
    auto lens      = {3, 3};
    auto shape     = migraphx::shape{type, lens};
    auto x         = mm.add_parameter("x", shape);
    auto alpha     = mm.add_literal(migraphx::literal{migraphx::shape{type}, {alpha_val}});
    auto mul_alpha = add_common_op(mm, migraphx::make_op("mul"), {alpha, x});
    auto sigmoid   = mm.add_instruction(migraphx::make_op("sigmoid"), {mul_alpha});
    add_common_op(mm, migraphx::make_op("mul"), {x, sigmoid});

    EXPECT(mm == make_op_module("gelu_quick", {{"alpha", alpha_val}}, mm.get_parameters()));
}

TEST_CASE(gelu_erf_happy_path_op_builder_test)
{
    EXPECT(false);
}

TEST_CASE(gelu_tanh_fast_happy_path_op_builder_test)
{
    EXPECT(false);
}

TEST_CASE(gelu_tanh_not_fast_happy_path_op_builder_test)
{
    EXPECT(false);
}

TEST_CASE(gelu_split_happy_path_op_builder_path)
{
    EXPECT(false);
}

TEST_CASE(gelu_split_invalid_dimension_op_builder_path)
{
    EXPECT(false);
}
