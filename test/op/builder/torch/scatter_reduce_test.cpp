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

#include <limits>
#include <string>
#include <op_builder_test_utils.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/literal.hpp>

// tm::scatter_reduce maps the torch reduction onto the matching scatter op. When
// include_self is false the target positions are first overwritten with the
// reduction identity (via scatter_none) so they drop out of the reduction.
static void check_scatter_reduce(const std::string& reduce,
                                 const std::string& scatter_op,
                                 float identity,
                                 bool include_self)
{
    const auto f = migraphx::shape::float_type;
    const auto i = migraphx::shape::int32_type;

    migraphx::module mm;
    auto inp  = mm.add_parameter("inp", {f, {4, 4}});
    auto idx  = mm.add_parameter("idx", {i, {2, 4}});
    auto src  = mm.add_parameter("src", {f, {2, 4}});
    auto data = inp;
    if(not include_self)
    {
        auto id = mm.add_literal(migraphx::literal{migraphx::shape{f, {1}}, {identity}});
        id   = mm.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 4}}}), id);
        data = mm.add_instruction(migraphx::make_op("scatter_none", {{"axis", 0}}), inp, idx, id);
    }
    mm.add_instruction(migraphx::make_op(scatter_op, {{"axis", 0}}), data, idx, src);

    migraphx::value options{{"dim", 0}, {"reduce", reduce}, {"include_self", include_self}};
    EXPECT(mm == make_op_module("tm::scatter_reduce", options, mm.get_parameters()));
}

TEST_CASE(torch_kit_scatter_reduce_sum_include_self)
{
    check_scatter_reduce("sum", "scatter_add", 0.0f, true);
}

TEST_CASE(torch_kit_scatter_reduce_sum) { check_scatter_reduce("sum", "scatter_add", 0.0f, false); }

TEST_CASE(torch_kit_scatter_reduce_prod)
{
    check_scatter_reduce("prod", "scatter_mul", 1.0f, false);
}

TEST_CASE(torch_kit_scatter_reduce_amax)
{
    check_scatter_reduce("amax", "scatter_max", std::numeric_limits<float>::lowest(), false);
}

TEST_CASE(torch_kit_scatter_reduce_amin)
{
    check_scatter_reduce("amin", "scatter_min", std::numeric_limits<float>::max(), false);
}

TEST_CASE(torch_kit_scatter_reduce_unsupported_reduce)
{
    EXPECT(test::throws<migraphx::exception>([&] {
        make_op_module(
            "tm::scatter_reduce", {{"dim", 0}, {"reduce", "bogus"}, {"include_self", true}}, {});
    }));
}
