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
#include <migraphx/common.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/module.hpp>
#include <migraphx/serialize.hpp>
#include <migraphx/sym.hpp>

#include <test.hpp>

using dd = migraphx::shape::dynamic_dimension;
using migraphx::sym::lit;
using migraphx::sym::var;

TEST_CASE(add_common_op_symbolic_no_common_input)
{
    auto n = var("n", {1, 8});
    auto m = var("m", {1, 16});
    migraphx::shape a_shape{migraphx::shape::float_type, {dd{n}, dd{lit(1)}}};
    migraphx::shape b_shape{migraphx::shape::float_type, {dd{lit(1)}, dd{m}}};
    std::vector<dd> output_dims{dd{n}, dd{m}};

    migraphx::module actual;
    auto actual_a = actual.add_parameter("a", a_shape);
    auto actual_b = actual.add_parameter("b", b_shape);
    auto actual_result =
        migraphx::add_common_op(actual, migraphx::make_op("add"), {actual_a, actual_b});
    actual.add_return({actual_result});

    migraphx::module expected;
    auto expected_a = expected.add_parameter("a", a_shape);
    auto expected_b = expected.add_parameter("b", b_shape);
    auto common     = expected.add_instruction(
        migraphx::make_op("multibroadcast", {{"out_dyn_dims", migraphx::to_value(output_dims)}}),
        expected_a,
        expected_b);
    auto broadcast_b = expected.add_instruction(
        migraphx::make_op("multibroadcast", {{"out_dyn_dims", migraphx::to_value(output_dims)}}),
        expected_b,
        common);
    auto expected_result = expected.add_instruction(migraphx::make_op("add"), common, broadcast_b);
    expected.add_return({expected_result});

    EXPECT(actual == expected);
}

TEST_CASE(add_common_op_symbolic_common_input)
{
    auto n = var("n", {1, 8});
    migraphx::shape a_shape{migraphx::shape::float_type, {dd{n}, dd{lit(4)}}};
    migraphx::shape b_shape{migraphx::shape::float_type, {4}};
    std::vector<dd> output_dims{dd{n}, dd{lit(4)}};

    migraphx::module actual;
    auto actual_a = actual.add_parameter("a", a_shape);
    auto actual_b = actual.add_parameter("b", b_shape);
    auto actual_result =
        migraphx::add_common_op(actual, migraphx::make_op("add"), {actual_a, actual_b});
    actual.add_return({actual_result});

    migraphx::module expected;
    auto expected_a  = expected.add_parameter("a", a_shape);
    auto expected_b  = expected.add_parameter("b", b_shape);
    auto broadcast_b = expected.add_instruction(
        migraphx::make_op("multibroadcast", {{"out_dyn_dims", migraphx::to_value(output_dims)}}),
        expected_b,
        expected_a);
    auto expected_result =
        expected.add_instruction(migraphx::make_op("add"), expected_a, broadcast_b);
    expected.add_return({expected_result});

    EXPECT(actual == expected);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
