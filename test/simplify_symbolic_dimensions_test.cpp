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
                           {"ends", migraphx::value::array{migraphx::to_value(nonzero_count)}}}),
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

int main(int argc, const char* argv[]) { test::run(argc, argv); }
