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
#include <migraphx/symbolic_shape_helpers.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/module.hpp>
#include <migraphx/sym.hpp>
#include <test.hpp>

#include <vector>

TEST_CASE(find_expression_sources_from_data)
{
    using dd = migraphx::shape::dynamic_dimension;
    using migraphx::sym::lit;
    using migraphx::sym::var;

    const auto n = var("n", {1, 8});
    migraphx::module m;
    auto data =
        m.add_parameter("data", {migraphx::shape::float_type, {dd{n + lit(1)}, dd{lit(4)}}});

    auto sources = migraphx::find_expression_sources(m, data, {n + lit(1), lit(4)});
    EXPECT(sources.has_value());
    if(sources.has_value())
        EXPECT(*sources == std::vector<migraphx::instruction_ref>{data});
}

TEST_CASE(find_expression_sources_from_root)
{
    using dd = migraphx::shape::dynamic_dimension;
    using migraphx::sym::lit;
    using migraphx::sym::var;

    const auto n = var("n", {1, 8});
    migraphx::module m;
    auto root = m.add_parameter("root", {migraphx::shape::float_type, {dd{n}, dd{lit(4)}}});
    auto data = m.add_instruction(migraphx::make_op("flatten", {{"axis", 0}}), root);

    auto sources = migraphx::find_expression_sources(m, data, {n, lit(4)});
    EXPECT(sources.has_value());
    if(sources.has_value())
        EXPECT(*sources == std::vector<migraphx::instruction_ref>{data, root});
}

TEST_CASE(find_expression_sources_from_multiple_roots)
{
    using dd = migraphx::shape::dynamic_dimension;
    using migraphx::sym::var;

    const auto m_dim = var("m", {1, 8});
    const auto n_dim = var("n", {1, 8});
    migraphx::module m;
    auto m_root = m.add_parameter("m_root", {migraphx::shape::float_type, {dd{m_dim}}});
    auto n_root = m.add_parameter("n_root", {migraphx::shape::float_type, {dd{n_dim}}});
    auto data   = m.add_parameter("data", {migraphx::shape::float_type, {1}});

    auto sources = migraphx::find_expression_sources(m, data, {m_dim + n_dim});
    EXPECT(sources.has_value());
    if(sources.has_value())
        EXPECT(*sources == std::vector<migraphx::instruction_ref>{data, n_root, m_root});
}

TEST_CASE(find_expression_sources_missing_root)
{
    using dd = migraphx::shape::dynamic_dimension;
    using migraphx::sym::lit;
    using migraphx::sym::var;

    const auto n = var("n", {1, 8});
    migraphx::module m;
    auto data = m.add_parameter("data", {migraphx::shape::float_type, {dd{n * lit(4)}}});

    auto sources = migraphx::find_expression_sources(m, data, {n});
    EXPECT(not sources.has_value());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
