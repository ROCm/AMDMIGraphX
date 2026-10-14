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
#include <migraphx/make_op.hpp>
#include <migraphx/module.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/sym.hpp>
#include <test.hpp>
#include <vector>

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
    if(broadcasts.size() == 1)
    {
        EXPECT(add->inputs().front() == broadcasts.front());
        EXPECT(add->inputs().front()->get_shape().dyn_dims() ==
               add->inputs().back()->get_shape().dyn_dims());
    }
}

TEST_CASE(symbolic_binary_disjoint_expressions_do_not_broadcast)
{
    using migraphx::sym::var;

    const auto n = var("n", {0, 10});
    const auto k = var("k", {20, 30});
    migraphx::module m;
    auto x = m.add_parameter(
        "x", {migraphx::shape::float_type, std::vector<migraphx::shape::dynamic_dimension>{{n}}});
    auto y = m.add_parameter(
        "y", {migraphx::shape::float_type, std::vector<migraphx::shape::dynamic_dimension>{{k}}});
    auto add = migraphx::add_common_op(m, migraphx::make_op("add"), {x, y});
    m.add_return({add});

    auto broadcasts = migraphx::find_all(migraphx::iterator_for(m),
                                         [](auto ins) { return ins->name() == "multibroadcast"; });
    EXPECT(broadcasts.empty());
    EXPECT(add->inputs() == std::vector<migraphx::instruction_ref>{x, y});
}

TEST_CASE(symbolic_binary_literal_expression_mismatch)
{
    using migraphx::sym::lit;
    using migraphx::sym::var;

    const auto n = var("n", {4, 8});
    migraphx::module m;
    auto x = m.add_parameter(
        "x",
        {migraphx::shape::float_type,
         std::vector<migraphx::shape::dynamic_dimension>{{lit(4)}}});
    auto y = m.add_parameter(
        "y", {migraphx::shape::float_type, std::vector<migraphx::shape::dynamic_dimension>{{n}}});
    EXPECT(test::throws(
        [&] { migraphx::add_common_op(m, migraphx::make_op("add"), {x, y}); }));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
