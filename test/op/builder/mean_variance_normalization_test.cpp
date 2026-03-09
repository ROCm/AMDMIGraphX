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

TEST_CASE(mean_variance_normalization_invalid_input_dim_op_builder_test)
{
    migraphx::module mm;
    mm.add_parameter("x", {migraphx::shape::float_type, {3}});

    EXPECT(test::throws<migraphx::exception>(
        [&] { make_op_module("mean_variance_normalization", {}, mm.get_parameters()); },
        "mvn op_builder: Length of axes attribute needs to be equal to input tensor rank - 1"));
}

TEST_CASE(mean_variance_normalization_happy_path_op_builder_test)
{
    migraphx::module mm;

    const auto axes = {2, 2, 2};
    auto x          = mm.add_parameter("x", {migraphx::shape::float_type, {2, 2, 2, 2}});

    auto x_mean         = mm.add_instruction(migraphx::make_op("reduce_mean", {{"axes", axes}}), x);
    auto x_mean_squared = add_common_op(mm, migraphx::make_op("mul"), {x_mean, x_mean});
    auto x_squared      = add_common_op(mm, migraphx::make_op("mul"), {x, x});
    auto x_squared_mean =
        mm.add_instruction(migraphx::make_op("reduce_mean", {{"axes", axes}}), x_squared);
    auto mean_sub = add_common_op(mm, migraphx::make_op("sub"), {x_squared_mean, x_mean_squared});
    auto std      = add_common_op(mm, migraphx::make_op("sqrt"), {mean_sub});
    auto dividend = add_common_op(mm, migraphx::make_op("sub"), {x, x_mean});
    auto epsilon  = mm.add_literal(1e-9f);
    auto divisor  = add_common_op(mm, migraphx::make_op("add"), {std, epsilon});
    add_common_op(mm, migraphx::make_op("div"), {dividend, divisor});

    EXPECT(mm ==
           make_op_module("mean_variance_normalization", {{"axes", axes}}, mm.get_parameters()));
}
