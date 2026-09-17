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

// tm::clip lowers to clip/min/max/identity based on which optional bounds are given
// (an undefined arg means "absent").

TEST_CASE(torch_kit_clip_min_and_max_op_builder_test)
{
    const auto f = migraphx::shape::float_type;
    migraphx::module mm;
    auto x  = mm.add_parameter("x", {f, {2, 3}});
    auto lo = mm.add_parameter("lo", {f, {2, 3}});
    auto hi = mm.add_parameter("hi", {f, {2, 3}});
    add_common_op(mm, migraphx::make_op("clip"), {x, lo, hi});

    EXPECT(mm == make_op_module("tm::clip", mm.get_parameters()));
}

TEST_CASE(torch_kit_clip_min_only_op_builder_test)
{
    // max is undefined -> lowers to max(x, lo).
    const auto f = migraphx::shape::float_type;
    migraphx::module mm;
    auto x  = mm.add_parameter("x", {f, {2, 3}});
    auto lo = mm.add_parameter("lo", {f, {2, 3}});
    mm.add_instruction(migraphx::make_op("undefined"));
    add_common_op(mm, migraphx::make_op("max"), {x, lo});

    migraphx::module mm_op_built;
    auto x_op  = mm_op_built.add_parameter("x", {f, {2, 3}});
    auto lo_op = mm_op_built.add_parameter("lo", {f, {2, 3}});
    auto hi_op = mm_op_built.add_instruction(migraphx::make_op("undefined"));
    migraphx::op::builder::add("tm::clip", mm_op_built, {x_op, lo_op, hi_op});
    EXPECT(mm == mm_op_built);
}

TEST_CASE(torch_kit_clip_max_only_op_builder_test)
{
    // min is undefined -> lowers to min(x, hi).
    const auto f = migraphx::shape::float_type;
    migraphx::module mm;
    auto x = mm.add_parameter("x", {f, {2, 3}});
    mm.add_instruction(migraphx::make_op("undefined"));
    auto hi = mm.add_parameter("hi", {f, {2, 3}});
    add_common_op(mm, migraphx::make_op("min"), {x, hi});

    migraphx::module mm_op_built;
    auto x_op  = mm_op_built.add_parameter("x", {f, {2, 3}});
    auto lo_op = mm_op_built.add_instruction(migraphx::make_op("undefined"));
    auto hi_op = mm_op_built.add_parameter("hi", {f, {2, 3}});
    migraphx::op::builder::add("tm::clip", mm_op_built, {x_op, lo_op, hi_op});
    EXPECT(mm == mm_op_built);
}

TEST_CASE(torch_kit_clip_none_op_builder_test)
{
    // Neither bound supplied -> identity(x).
    const auto f = migraphx::shape::float_type;
    migraphx::module mm;
    auto x = mm.add_parameter("x", {f, {2, 3}});
    mm.add_instruction(migraphx::make_op("undefined"));
    mm.add_instruction(migraphx::make_op("undefined"));
    mm.add_instruction(migraphx::make_op("identity"), x);

    migraphx::module mm_op_built;
    auto x_op  = mm_op_built.add_parameter("x", {f, {2, 3}});
    auto lo_op = mm_op_built.add_instruction(migraphx::make_op("undefined"));
    auto hi_op = mm_op_built.add_instruction(migraphx::make_op("undefined"));
    migraphx::op::builder::add("tm::clip", mm_op_built, {x_op, lo_op, hi_op});
    EXPECT(mm == mm_op_built);
}
