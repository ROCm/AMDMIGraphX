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

// tm::hardsigmoid == clip(alpha * x + beta, 0, 1) with alpha = 1/6, beta = 1/2.
TEST_CASE(torch_kit_hardsigmoid_op_builder_test)
{
    const auto f = migraphx::shape::float_type;
    migraphx::module mm;
    auto x       = mm.add_parameter("x", {f, {2, 3}});
    auto alpha   = mm.add_literal(migraphx::literal{migraphx::shape{f}, {1.0f / 6.0f}});
    auto beta    = mm.add_literal(migraphx::literal{migraphx::shape{f}, {0.5f}});
    auto lo      = mm.add_literal(migraphx::literal{migraphx::shape{f}, {0.0f}});
    auto hi      = mm.add_literal(migraphx::literal{migraphx::shape{f}, {1.0f}});
    auto scaled  = add_common_op(mm, migraphx::make_op("mul"), {alpha, x});
    auto shifted = add_common_op(mm, migraphx::make_op("add"), {beta, scaled});
    add_common_op(mm, migraphx::make_op("clip"), {shifted, lo, hi});

    EXPECT(mm == make_op_module("tm::hardsigmoid", mm.get_parameters()));
}
