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
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/program.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/simplify_symbolic_dimensions.hpp>

#include "test.hpp"

static void run_pass(migraphx::module& m)
{
    migraphx::run_passes(m, {migraphx::simplify_symbolic_dimensions{}});
}

TEST_CASE(simplify_symbolic_dimensions_binary_parameters)
{
    using migraphx::sym::lit;
    using migraphx::sym::min;
    using migraphx::sym::var;

    const auto nms_count     = var("nms_count", {0, 258});
    const auto topk_count    = min(lit(100), nms_count);
    const auto nonzero_count = var("nonzero_count", {0, 100});

    migraphx::module m;
    auto topk = m.add_parameter("topk",
                                {migraphx::shape::int64_type,
                                 std::vector<migraphx::shape::dynamic_dimension>{{topk_count}}});
    auto nonzero =
        m.add_parameter("nonzero",
                        {migraphx::shape::int64_type,
                         std::vector<migraphx::shape::dynamic_dimension>{{nonzero_count}}});
    auto add = m.add_instruction(migraphx::make_op("add"), topk, nonzero);
    m.add_return({add});

    run_pass(m);

    const migraphx::shape expected{migraphx::shape::int64_type,
                                   std::vector<migraphx::shape::dynamic_dimension>{{topk_count}}};
    EXPECT(m.get_parameter_shape("topk") == expected);
    EXPECT(m.get_parameter_shape("nonzero") == expected);
    EXPECT(m.get_output_shapes().front() == expected);

    auto once = m;
    run_pass(m);
    EXPECT(m == once);
}

TEST_CASE(simplify_symbolic_dimensions_shape_attributes)
{
    using migraphx::sym::lit;
    using migraphx::sym::min;
    using migraphx::sym::var;

    const auto nms_count     = var("nms_count", {0, 258});
    const auto topk_count    = min(lit(100), nms_count);
    const auto nonzero_count = var("nonzero_count", {0, 100});

    migraphx::module m;
    auto topk    = m.add_parameter("topk",
                                   {migraphx::shape::int64_type,
                                    std::vector<migraphx::shape::dynamic_dimension>{{topk_count}}});
    auto indices = m.add_parameter("indices", {migraphx::shape::int64_type, {1, 100}});
    auto count   = m.add_parameter("count", {migraphx::shape::int64_type, {1}});
    auto zero    = m.add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {0}});
    auto sliced  = m.add_instruction(
        migraphx::make_op("dyn_slice",
                           {{"axes", {1}},
                            {"starts", {0}},
                            {"ends", migraphx::value::array{migraphx::to_value(nonzero_count)}},
                            {"always_leq", true}}),
        indices,
        zero,
        count);
    auto squeezed  = m.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), sliced);
    auto one       = m.add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {1}});
    auto broadcast = m.add_instruction(
        migraphx::make_op("multibroadcast",
                          {{"out_dyn_dims",
                            migraphx::to_value(std::vector<migraphx::shape::dynamic_dimension>{
                                {nonzero_count}})}}),
        one,
        squeezed);
    auto add = m.add_instruction(migraphx::make_op("add"), topk, broadcast);
    m.add_return({add});

    run_pass(m);

    auto slices = migraphx::find_all(migraphx::iterator_for(m),
                                     [](auto ins) { return ins->name() == "dyn_slice"; });
    EXPECT(slices.size() == 1);
    if(slices.size() == 1)
    {
        auto ends = slices.front()->get_operator().to_value().at("ends");
        EXPECT(migraphx::from_value<std::vector<migraphx::sym::expr>>(ends).front() == topk_count);
    }

    auto broadcasts = migraphx::find_all(migraphx::iterator_for(m),
                                         [](auto ins) { return ins->name() == "multibroadcast"; });
    EXPECT(broadcasts.size() == 1);
    if(broadcasts.size() == 1)
    {
        auto dims = migraphx::from_value<std::vector<migraphx::shape::dynamic_dimension>>(
            broadcasts.front()->get_operator().to_value().at("out_dyn_dims"));
        EXPECT(dims.front().sym_expr == topk_count);
    }
    EXPECT(m.get_output_shapes().front().dyn_dims().front().sym_expr == topk_count);
}

TEST_CASE(symbolic_binary_explicit_broadcast)
{
    using migraphx::sym::lit;
    using migraphx::sym::var;

    const auto count = var("count", {1, 100});
    migraphx::module m;
    auto scalar = m.add_parameter(
        "scalar",
        {migraphx::shape::float_type, std::vector<migraphx::shape::dynamic_dimension>{{lit(1)}}});
    auto values = m.add_parameter(
        "values",
        {migraphx::shape::float_type, std::vector<migraphx::shape::dynamic_dimension>{{count}}});
    auto add = migraphx::add_common_op(m, migraphx::make_op("add"), {scalar, values});
    m.add_return({add});

    auto broadcasts = migraphx::find_all(migraphx::iterator_for(m),
                                         [](auto ins) { return ins->name() == "multibroadcast"; });
    EXPECT(broadcasts.size() == 1);
    EXPECT(add->inputs().front() == broadcasts.front());
    EXPECT(add->inputs().front()->get_shape().dyn_dims() ==
           add->inputs().back()->get_shape().dyn_dims());
}

TEST_CASE(symbolic_binary_different_expressions_do_not_broadcast)
{
    using migraphx::sym::var;

    const auto n = var("n", {0, 100});
    const auto m = var("m", {0, 100});
    migraphx::module mod;
    auto x = mod.add_parameter(
        "x",
        {migraphx::shape::float_type,
         std::vector<migraphx::shape::dynamic_dimension>{{n}}});
    auto y = mod.add_parameter(
        "y",
        {migraphx::shape::float_type,
         std::vector<migraphx::shape::dynamic_dimension>{{m}}});
    auto add = migraphx::add_common_op(mod, migraphx::make_op("add"), {x, y});
    mod.add_return({add});

    auto broadcasts = migraphx::find_all(
        migraphx::iterator_for(mod), [](auto ins) { return ins->name() == "multibroadcast"; });
    EXPECT(broadcasts.empty());
    EXPECT(add->inputs() == std::vector<migraphx::instruction_ref>{x, y});
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
