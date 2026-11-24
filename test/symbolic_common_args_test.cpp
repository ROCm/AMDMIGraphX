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
#include <migraphx/sym.hpp>
#include <test.hpp>
#include <vector>

TEST_CASE(symbolic_binary_explicit_broadcast)
{
    using migraphx::sym::lit;
    using migraphx::sym::var;

    const auto count = var("count", {1, 100});
    std::vector<migraphx::shape::dynamic_dimension> values_dims{{count}};
    migraphx::module m1;
    {
        auto scalar = m1.add_parameter("scalar",
                                       {migraphx::shape::float_type,
                                        std::vector<migraphx::shape::dynamic_dimension>{{lit(1)}}});
        auto values = m1.add_parameter("values", {migraphx::shape::float_type, values_dims});
        auto add    = migraphx::add_common_op(m1, migraphx::make_op("add"), {scalar, values});
        m1.add_return({add});
    }
    migraphx::module m2;
    {
        auto scalar = m2.add_parameter("scalar",
                                       {migraphx::shape::float_type,
                                        std::vector<migraphx::shape::dynamic_dimension>{{lit(1)}}});
        auto values = m2.add_parameter("values", {migraphx::shape::float_type, values_dims});
        auto broadcast = m2.add_instruction(
            migraphx::make_op("multibroadcast",
                              {{"out_dyn_dims", migraphx::to_value(values_dims)}}),
            scalar,
            values);
        auto add = m2.add_instruction(migraphx::make_op("add"), broadcast, values);
        m2.add_return({add});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(symbolic_binary_disjoint_expressions_do_not_broadcast)
{
    using migraphx::sym::var;

    const auto n = var("n", {0, 10});
    const auto k = var("k", {20, 30});
    migraphx::module m1;
    {
        auto x = m1.add_parameter(
            "x",
            {migraphx::shape::float_type, std::vector<migraphx::shape::dynamic_dimension>{{n}}});
        auto y = m1.add_parameter(
            "y",
            {migraphx::shape::float_type, std::vector<migraphx::shape::dynamic_dimension>{{k}}});
        auto add = migraphx::add_common_op(m1, migraphx::make_op("add"), {x, y});
        m1.add_return({add});
    }
    migraphx::module m2;
    {
        auto x = m2.add_parameter(
            "x",
            {migraphx::shape::float_type, std::vector<migraphx::shape::dynamic_dimension>{{n}}});
        auto y = m2.add_parameter(
            "y",
            {migraphx::shape::float_type, std::vector<migraphx::shape::dynamic_dimension>{{k}}});
        auto add = m2.add_instruction(migraphx::make_op("add"), x, y);
        m2.add_return({add});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(symbolic_binary_literal_expression_mismatch)
{
    using migraphx::sym::lit;
    using migraphx::sym::var;

    const auto n = var("n", {4, 8});
    migraphx::module m;
    auto x = m.add_parameter(
        "x",
        {migraphx::shape::float_type, std::vector<migraphx::shape::dynamic_dimension>{{lit(4)}}});
    auto y = m.add_parameter(
        "y", {migraphx::shape::float_type, std::vector<migraphx::shape::dynamic_dimension>{{n}}});
    EXPECT(test::throws([&] { migraphx::add_common_op(m, migraphx::make_op("add"), {x, y}); }));
}

TEST_CASE(symbolic_range_common_dims_mismatch)
{
    using migraphx::sym::var;

    std::vector<migraphx::shape::dynamic_dimension> symbolic_dims{{var("n", {2, 8})}};
    std::vector<migraphx::shape::dynamic_dimension> range_dims{{2, 8}};
    EXPECT(
        test::throws([&] { migraphx::compute_broadcasted_dyn_dims(symbolic_dims, range_dims); }));
    EXPECT(
        test::throws([&] { migraphx::compute_broadcasted_dyn_dims(range_dims, symbolic_dims); }));
}

// A mask of length m added to scores of length n + 1 keeps m in either operand order; only the
// axis of length 1 is broadcast
TEST_CASE(symbolic_binary_own_symbol_either_order)
{
    using migraphx::sym::lit;
    using migraphx::sym::var;

    const auto m = var("m", {2, 9});
    const auto n = var("n", {1, 8});
    std::vector<migraphx::shape::dynamic_dimension> mask_dims{{lit(1)}, {lit(1)}, {m}};
    std::vector<migraphx::shape::dynamic_dimension> scores_dims{{lit(1)}, {lit(12)}, {n + lit(1)}};
    std::vector<migraphx::shape::dynamic_dimension> bcast_dims{{lit(1)}, {lit(12)}, {m}};
    for(bool mask_first : {false, true})
    {
        migraphx::module m1;
        {
            auto mask   = m1.add_parameter("mask", {migraphx::shape::float_type, mask_dims});
            auto scores = m1.add_parameter("scores", {migraphx::shape::float_type, scores_dims});
            std::vector<migraphx::instruction_ref> args{scores, mask};
            if(mask_first)
                std::swap(args.front(), args.back());
            m1.add_return({migraphx::add_common_op(m1, migraphx::make_op("add"), args)});
        }
        migraphx::module m2;
        {
            auto mask   = m2.add_parameter("mask", {migraphx::shape::float_type, mask_dims});
            auto scores = m2.add_parameter("scores", {migraphx::shape::float_type, scores_dims});
            auto bcast  = m2.add_instruction(
                migraphx::make_op("multibroadcast",
                                  {{"out_dyn_dims", migraphx::to_value(bcast_dims)}}),
                mask);
            std::vector<migraphx::instruction_ref> args{scores, bcast};
            if(mask_first)
                std::swap(args.front(), args.back());
            m2.add_return({m2.add_instruction(migraphx::make_op("add"), args)});
        }
        EXPECT(m1 == m2);
    }
}

// Each input broadcasts the other's axis of length 1, and keeps its own symbol on the last axis
TEST_CASE(symbolic_binary_mutual_broadcast_own_symbols)
{
    using migraphx::sym::lit;
    using migraphx::sym::var;

    const auto b = var("b", {1, 4});
    const auto m = var("m", {2, 9});
    const auto n = var("n", {2, 9});
    std::vector<migraphx::shape::dynamic_dimension> x_dims{{b}, {lit(1)}, {m}};
    std::vector<migraphx::shape::dynamic_dimension> y_dims{{lit(1)}, {lit(12)}, {n}};
    std::vector<migraphx::shape::dynamic_dimension> x_out{{b}, {lit(12)}, {m}};
    std::vector<migraphx::shape::dynamic_dimension> y_out{{b}, {lit(12)}, {n}};
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::float_type, x_dims});
        auto y = m1.add_parameter("y", {migraphx::shape::float_type, y_dims});
        m1.add_return({migraphx::add_common_op(m1, migraphx::make_op("add"), {x, y})});
    }
    migraphx::module m2;
    {
        auto x  = m2.add_parameter("x", {migraphx::shape::float_type, x_dims});
        auto y  = m2.add_parameter("y", {migraphx::shape::float_type, y_dims});
        auto bx = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_dyn_dims", migraphx::to_value(x_out)}}), x);
        auto by = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_dyn_dims", migraphx::to_value(y_out)}}), y);
        m2.add_return({m2.add_instruction(migraphx::make_op("add"), bx, by)});
    }
    EXPECT(m1 == m2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
