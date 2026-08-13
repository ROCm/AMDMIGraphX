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
#include <migraphx/bf16.hpp>
#include <migraphx/make_op.hpp>
#include <limits>

// tm::nan_to_num replaces NaN with `nan`, +inf with `posinf`, -inf with `neginf`;
// the inf sign is recovered by comparing the input against 0. where broadcasts its
// operands but does not promote the boolean condition.
TEST_CASE(torch_kit_nan_to_num_op_builder_test)
{
    const auto f = migraphx::shape::float_type;
    migraphx::value options{{"nan", 0.0f}, {"posinf", 1e4f}, {"neginf", -1e4f}};

    migraphx::module mm;
    auto x          = mm.add_parameter("x", {f, {2, 3}});
    auto nan_lit    = mm.add_literal(migraphx::literal{migraphx::shape{f}, {0.0f}});
    auto zero       = mm.add_literal(migraphx::literal{migraphx::shape{f}, {0.0f}});
    auto posinf_lit = mm.add_literal(migraphx::literal{migraphx::shape{f}, {1e4f}});
    auto neginf_lit = mm.add_literal(migraphx::literal{migraphx::shape{f}, {-1e4f}});

    auto is_nan = mm.add_instruction(migraphx::make_op("isnan"), x);
    auto result =
        add_common_op(mm, migraphx::make_op("where"), {is_nan, nan_lit, x}, {.common_type = false});
    auto is_inf   = mm.add_instruction(migraphx::make_op("isinf"), x);
    auto less     = add_common_op(mm, migraphx::make_op("less"), {x, zero});
    auto greater  = add_common_op(mm, migraphx::make_op("greater"), {x, zero});
    auto neg_mask = add_common_op(mm, migraphx::make_op("logical_and"), {less, is_inf});
    auto pos_mask = add_common_op(mm, migraphx::make_op("logical_and"), {greater, is_inf});
    result        = add_common_op(
        mm, migraphx::make_op("where"), {neg_mask, neginf_lit, result}, {.common_type = false});
    add_common_op(
        mm, migraphx::make_op("where"), {pos_mask, posinf_lit, result}, {.common_type = false});

    EXPECT(mm == make_op_module("tm::nan_to_num", options, mm.get_parameters()));
}

// The posinf/neginf defaults are FLT_MAX/FLT_LOWEST, standing for the largest finite value of the
// tensor's type. Converted directly they round past bf16's range to infinity, leaving the op
// substituting an infinity for an infinity, so they must saturate to bf16's finite limits.
TEST_CASE(torch_kit_nan_to_num_bf16_default_saturates_test)
{
    const auto bf = migraphx::shape::bf16_type;

    migraphx::module mm;
    auto x       = mm.add_parameter("x", {bf, {2, 3}});
    auto nan_lit = mm.add_literal(migraphx::literal{migraphx::shape{bf}, {migraphx::bf16{0.0f}}});
    auto zero    = mm.add_literal(migraphx::literal{migraphx::shape{bf}, {0.0f}});
    auto posinf_lit = mm.add_literal(
        migraphx::literal{migraphx::shape{bf}, {std::numeric_limits<migraphx::bf16>::max()}});
    auto neginf_lit = mm.add_literal(
        migraphx::literal{migraphx::shape{bf}, {std::numeric_limits<migraphx::bf16>::lowest()}});

    auto is_nan = mm.add_instruction(migraphx::make_op("isnan"), x);
    auto result =
        add_common_op(mm, migraphx::make_op("where"), {is_nan, nan_lit, x}, {.common_type = false});
    auto is_inf   = mm.add_instruction(migraphx::make_op("isinf"), x);
    auto less     = add_common_op(mm, migraphx::make_op("less"), {x, zero});
    auto greater  = add_common_op(mm, migraphx::make_op("greater"), {x, zero});
    auto neg_mask = add_common_op(mm, migraphx::make_op("logical_and"), {less, is_inf});
    auto pos_mask = add_common_op(mm, migraphx::make_op("logical_and"), {greater, is_inf});
    result        = add_common_op(
        mm, migraphx::make_op("where"), {neg_mask, neginf_lit, result}, {.common_type = false});
    add_common_op(
        mm, migraphx::make_op("where"), {pos_mask, posinf_lit, result}, {.common_type = false});

    EXPECT(mm == make_op_module("tm::nan_to_num", mm.get_parameters()));
}
