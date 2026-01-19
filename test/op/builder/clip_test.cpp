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

TEST_CASE(clip_max_and_min_op_builder_test)
{
    migraphx::module mm;
    auto arg0 = mm.add_parameter("0", {migraphx::shape::float_type, {2, 4, 5}});
    auto arg1 = mm.add_parameter("1", {migraphx::shape::float_type, {2, 4, 5}});
    auto arg2 = mm.add_parameter("2", {migraphx::shape::float_type, {2, 4, 5}});
    add_common_op(mm, migraphx::make_op("clip"), {arg0, arg1, arg2});

    EXPECT(mm == make_op_module("clip", migraphx::value("", {}, false), mm.get_parameters()));
}

TEST_CASE(clip_only_max_op_builder_test)
{
    migraphx::module mm;
    auto arg0 = mm.add_parameter("0", {migraphx::shape::float_type, {2, 4, 5}});
    mm.add_instruction(migraphx::make_op("undefined"));
    auto arg2 = mm.add_parameter("2", {migraphx::shape::float_type, {2, 4, 5}});
    add_common_op(mm, migraphx::make_op("min"), {arg0, arg2});

    migraphx::module mm_op_built;
    auto arg0_op = mm_op_built.add_parameter("0", {migraphx::shape::float_type, {2, 4, 5}});
    auto arg1_op = mm_op_built.add_instruction(migraphx::make_op("undefined"));
    auto arg2_op = mm_op_built.add_parameter("2", {migraphx::shape::float_type, {2, 4, 5}});

    migraphx::op::builder::add("clip", mm_op_built, {arg0_op, arg1_op, arg2_op});
    EXPECT(mm == mm_op_built);
}

TEST_CASE(clip_only_min_op_builder_test)
{
    migraphx::module mm;
    auto arg0 = mm.add_parameter("0", {migraphx::shape::float_type, {2, 4, 5}});
    auto arg1 = mm.add_parameter("1", {migraphx::shape::float_type, {2, 4, 5}});
    add_common_op(mm, migraphx::make_op("max"), {arg0, arg1});

    EXPECT(mm == make_op_module("clip", migraphx::value("", {}, false), mm.get_parameters()));
}

TEST_CASE(clip_identity_op_builder_test)
{
    migraphx::module mm;
    auto arg0 = mm.add_parameter("0", {migraphx::shape::float_type, {2, 4, 5}});
    add_common_op(mm, migraphx::make_op("identity"), {arg0});

    EXPECT(mm == make_op_module("clip", migraphx::value("", {}, false), mm.get_parameters()));
}
