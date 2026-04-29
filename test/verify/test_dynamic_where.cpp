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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "verify_program.hpp"
#include <migraphx/make_op.hpp>
#include <migraphx/program.hpp>
#include <migraphx/sym.hpp>

template <std::size_t... TestDims>
struct test_dynamic_where : verify_program<test_dynamic_where<TestDims...>>
{
    migraphx::program create_program() const
    {
        migraphx::shape sb{migraphx::shape::bool_type, {{2, 4}, {8, 16}}};
        migraphx::shape sx{migraphx::shape::float_type, {{2, 4}, {8, 16}}};
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto b   = mm->add_parameter("b", sb);
        auto x   = mm->add_parameter("x", sx);
        auto y   = mm->add_parameter("y", sx);
        auto w   = mm->add_instruction(migraphx::make_op("where"), b, x, y);
        mm->add_return({w});
        return p;
    }

    std::unordered_map<std::string, migraphx::shape> get_test_dims() const
    {
        return {{"b", migraphx::shape{migraphx::shape::bool_type, {TestDims...}}},
                {"x", migraphx::shape{migraphx::shape::float_type, {TestDims...}}},
                {"y", migraphx::shape{migraphx::shape::float_type, {TestDims...}}}};
    }
};

template struct test_dynamic_where<4, 16>;
template struct test_dynamic_where<2, 8>;
template struct test_dynamic_where<3, 10>;

/// Dynamic x/y with broadcast-compatible ranges; predicate matches merged output.
struct test_dynamic_where_broadcast : verify_program<test_dynamic_where_broadcast>
{
    migraphx::program create_program() const
    {
        migraphx::shape sb{migraphx::shape::bool_type, {{2, 3}, {3, 3}}};
        migraphx::shape sx{migraphx::shape::float_type, {{2, 3}, {3, 3}}};
        migraphx::shape sy{migraphx::shape::float_type, {{2, 3}, {2, 3}}};
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto b   = mm->add_parameter("b", sb);
        auto x   = mm->add_parameter("x", sx);
        auto y   = mm->add_parameter("y", sy);
        auto w   = mm->add_instruction(migraphx::make_op("where"), b, x, y);
        mm->add_return({w});
        return p;
    }

    std::unordered_map<std::string, migraphx::shape> get_test_dims() const
    {
        return {{"b", migraphx::shape{migraphx::shape::bool_type, {3, 3}}},
                {"x", migraphx::shape{migraphx::shape::float_type, {3, 3}}},
                {"y", migraphx::shape{migraphx::shape::float_type, {3, 3}}}};
    }
};

/// Shared symbolic first dimension (GPU dynamic/symbolic path).
struct test_symbolic_where_gpu : verify_program<test_symbolic_where_gpu>
{
    migraphx::program create_program() const
    {
        using migraphx::sym::var;
        auto n = var("n", {1, 4});
        using dd = migraphx::shape::dynamic_dimension;

        migraphx::shape sb{migraphx::shape::bool_type, {dd{n}, dd{2, 8}}};
        migraphx::shape sx{migraphx::shape::float_type, {dd{n}, dd{2, 8}}};
        migraphx::shape sy{migraphx::shape::float_type, {dd{n}, dd{2, 8}}};
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto b   = mm->add_parameter("b", sb);
        auto x   = mm->add_parameter("x", sx);
        auto y   = mm->add_parameter("y", sy);
        auto w   = mm->add_instruction(migraphx::make_op("where"), b, x, y);
        mm->add_return({w});
        return p;
    }

    std::unordered_map<std::string, migraphx::shape> get_test_dims() const
    {
        return {{"b", migraphx::shape{migraphx::shape::bool_type, {3, 10}}},
                {"x", migraphx::shape{migraphx::shape::float_type, {3, 10}}},
                {"y", migraphx::shape{migraphx::shape::float_type, {3, 10}}}};
    }
};
