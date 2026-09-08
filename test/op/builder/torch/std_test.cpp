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

#include <op_builder_test_utils.hpp>
#include <migraphx/make_op.hpp>

// tm::std == sqrt(sum((x - mean)^2) / (N - correction)) reduced over the axes,
// squeezing them out unless keepdim. Here N == 4 and correction == 1, so N - 1 == 3.
TEST_CASE(torch_kit_std_op_builder_test)
{
    const auto f = migraphx::shape::float_type;
    migraphx::value options{{"axes", {1}}, {"keepdim", false}, {"correction", 1.0f}};

    migraphx::module mm;
    auto x     = mm.add_parameter("x", {f, {2, 4}});
    auto mean  = mm.add_instruction(migraphx::make_op("reduce_mean", {{"axes", {1}}}), x);
    auto sub   = add_common_op(mm, migraphx::make_op("sub"), {x, mean});
    auto sq    = add_common_op(mm, migraphx::make_op("mul"), {sub, sub});
    auto sum   = mm.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), sq);
    auto denom = mm.add_literal(migraphx::literal{migraphx::shape{f}, {3.0f}});
    auto var   = add_common_op(mm, migraphx::make_op("div"), {sum, denom});
    auto out   = mm.add_instruction(migraphx::make_op("sqrt"), var);
    mm.add_instruction(migraphx::make_op("squeeze", {{"axes", {1}}}), out);

    EXPECT(mm == make_op_module("tm::std", options, mm.get_parameters()));
}
