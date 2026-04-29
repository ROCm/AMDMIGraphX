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
struct test_dynamic_greater : verify_program<test_dynamic_greater<TestDims...>>
{
    migraphx::program create_program() const
    {
        migraphx::shape s{migraphx::shape::float_type, {{2, 4}, {8, 16}}};
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto x   = mm->add_parameter("x", s);
        auto y   = mm->add_parameter("y", s);
        auto g   = mm->add_instruction(migraphx::make_op("greater"), x, y);
        mm->add_return({g});
        return p;
    }

    std::unordered_map<std::string, migraphx::shape> get_test_dims() const
    {
        return {{"x", migraphx::shape{migraphx::shape::float_type, {TestDims...}}},
                {"y", migraphx::shape{migraphx::shape::float_type, {TestDims...}}}};
    }
};

template struct test_dynamic_greater<4, 16>;
template struct test_dynamic_greater<2, 8>;
template struct test_dynamic_greater<3, 10>;

struct test_dynamic_greater_broadcast : verify_program<test_dynamic_greater_broadcast>
{
    migraphx::program create_program() const
    {
        migraphx::shape s0{migraphx::shape::float_type, {{2, 3}, {3, 3}}};
        migraphx::shape s1{migraphx::shape::float_type, {{2, 3}, {2, 3}}};
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto x   = mm->add_parameter("x", s0);
        auto y   = mm->add_parameter("y", s1);
        auto g   = mm->add_instruction(migraphx::make_op("greater"), x, y);
        mm->add_return({g});
        return p;
    }

    std::unordered_map<std::string, migraphx::shape> get_test_dims() const
    {
        return {{"x", migraphx::shape{migraphx::shape::float_type, {3, 3}}},
                {"y", migraphx::shape{migraphx::shape::float_type, {3, 3}}}};
    }
};

struct test_symbolic_greater_gpu : verify_program<test_symbolic_greater_gpu>
{
    migraphx::program create_program() const
    {
        using migraphx::sym::var;
        auto n = var("n", {1, 4});
        using dd = migraphx::shape::dynamic_dimension;

        migraphx::shape sx{migraphx::shape::float_type, {dd{n}, dd{2, 8}}};
        migraphx::shape sy{migraphx::shape::float_type, {dd{n}, dd{2, 8}}};
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto x   = mm->add_parameter("x", sx);
        auto y   = mm->add_parameter("y", sy);
        auto g   = mm->add_instruction(migraphx::make_op("greater"), x, y);
        mm->add_return({g});
        return p;
    }

    std::unordered_map<std::string, migraphx::shape> get_test_dims() const
    {
        return {{"x", migraphx::shape{migraphx::shape::float_type, {3, 10}}},
                {"y", migraphx::shape{migraphx::shape::float_type, {3, 10}}}};
    }
};
