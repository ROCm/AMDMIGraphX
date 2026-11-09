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

#include <cstdint>
#include <limits>
#include <vector>
#include <op_builder_test_utils.hpp>
#include <migraphx/make_op.hpp>

// tm::vector_norm reduces abs(x) over axes with the ord-specific formula, then
// squeezes the reduced axes unless keepdim. General p-norm: sum(abs(x)^ord)^(1/ord).
TEST_CASE(torch_kit_vector_norm_p_op_builder_test)
{
    const auto f              = migraphx::shape::float_type;
    std::vector<int64_t> axes = {1};

    migraphx::module mm;
    auto x       = mm.add_parameter("x", {f, {2, 3}});
    auto abs_x   = mm.add_instruction(migraphx::make_op("abs"), x);
    auto ord_lit = mm.add_literal(migraphx::literal{migraphx::shape{f}, {2.0f}});
    auto pow_x   = add_common_op(mm, migraphx::make_op("pow"), {abs_x, ord_lit});
    auto sum_pow = mm.add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), pow_x);
    auto recip   = mm.add_instruction(migraphx::make_op("recip"), ord_lit);
    auto out     = add_common_op(mm, migraphx::make_op("pow"), {sum_pow, recip});
    mm.add_instruction(migraphx::make_op("squeeze", {{"axes", axes}}), out);

    EXPECT(mm == make_op_module("tm::vector_norm",
                                {{"ord", 2.0f}, {"axes", axes}, {"keepdim", false}},
                                mm.get_parameters()));
}

// ord = +inf -> max(abs(x)); keepdim = true leaves the reduced axis in place.
TEST_CASE(torch_kit_vector_norm_inf_op_builder_test)
{
    const auto f              = migraphx::shape::float_type;
    std::vector<int64_t> axes = {1};

    migraphx::module mm;
    auto x     = mm.add_parameter("x", {f, {2, 3}});
    auto abs_x = mm.add_instruction(migraphx::make_op("abs"), x);
    mm.add_instruction(migraphx::make_op("reduce_max", {{"axes", axes}}), abs_x);

    EXPECT(mm ==
           make_op_module(
               "tm::vector_norm",
               {{"ord", std::numeric_limits<float>::infinity()}, {"axes", axes}, {"keepdim", true}},
               mm.get_parameters()));
}

// ord = 0 -> count of nonzero elements: sum(abs(x) > 0).
TEST_CASE(torch_kit_vector_norm_zero_op_builder_test)
{
    const auto f              = migraphx::shape::float_type;
    std::vector<int64_t> axes = {1};

    migraphx::module mm;
    auto x       = mm.add_parameter("x", {f, {2, 3}});
    auto abs_x   = mm.add_instruction(migraphx::make_op("abs"), x);
    auto zero    = mm.add_literal(migraphx::literal{migraphx::shape{f}, {0.0f}});
    auto nonzero = add_common_op(mm, migraphx::make_op("greater"), {abs_x, zero});
    auto counts  = mm.add_instruction(migraphx::make_op("convert", {{"target_type", f}}), nonzero);
    auto out     = mm.add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), counts);
    mm.add_instruction(migraphx::make_op("squeeze", {{"axes", axes}}), out);

    EXPECT(mm == make_op_module("tm::vector_norm",
                                {{"ord", 0.0f}, {"axes", axes}, {"keepdim", false}},
                                mm.get_parameters()));
}
