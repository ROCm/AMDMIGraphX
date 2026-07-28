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

// tm::selu == gamma * (max(0, x) + min(0, alpha * (exp(x) - 1))) with the SELU
// constants; literals are created in the builder's order so the modules match.
TEST_CASE(torch_kit_selu_op_builder_test)
{
    const auto f = migraphx::shape::float_type;
    migraphx::module mm;
    auto x        = mm.add_parameter("x", {f, {2, 3}});
    auto zero     = mm.add_literal(migraphx::literal{migraphx::shape{f}, {0.0f}});
    auto one      = mm.add_literal(migraphx::literal{migraphx::shape{f}, {1.0f}});
    auto alpha    = mm.add_literal(migraphx::literal{migraphx::shape{f}, {1.6732632423543772f}});
    auto gamma    = mm.add_literal(migraphx::literal{migraphx::shape{f}, {1.0507009873554805f}});
    auto linear   = add_common_op(mm, migraphx::make_op("max"), {zero, x});
    auto exp_x    = mm.add_instruction(migraphx::make_op("exp"), x);
    auto exp_sub  = add_common_op(mm, migraphx::make_op("sub"), {exp_x, one});
    auto exp_mul  = add_common_op(mm, migraphx::make_op("mul"), {alpha, exp_sub});
    auto exp_part = add_common_op(mm, migraphx::make_op("min"), {zero, exp_mul});
    auto sum      = add_common_op(mm, migraphx::make_op("add"), {linear, exp_part});
    add_common_op(mm, migraphx::make_op("mul"), {gamma, sum});

    EXPECT(mm == make_op_module("tm::selu", mm.get_parameters()));
}
