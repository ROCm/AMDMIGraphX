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

#include "verify_program.hpp"
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/program.hpp>
#include <migraphx/sym.hpp>

struct test_split_sym_dim_bounded_prefix_dyn_slice
    : verify_program<test_split_sym_dim_bounded_prefix_dyn_slice>
{
    migraphx::program create_program() const
    {
        using dd = migraphx::shape::dynamic_dimension;
        using migraphx::sym::lit;
        using migraphx::sym::var;

        auto n = var("n", {1, 4}, {2});
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto data =
            mm->add_parameter("data", migraphx::shape{migraphx::shape::float_type, {dd{n}}});
        auto start = mm->add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {0}});
        auto count = mm->add_instruction(
            migraphx::make_op("dimensions_of", {{"start", 0}, {"end", 1}}), data);
        auto cap = mm->add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {1000}});
        auto extents = mm->add_instruction(migraphx::make_op("concat", {{"axis", 0}}), cap, count);
        auto end = mm->add_instruction(migraphx::make_op("reduce_min", {{"axes", {0}}}), extents);
        auto output = mm->add_instruction(
            migraphx::make_op(
                "dyn_slice",
                {{"axes", {0}},
                 {"starts", migraphx::to_value(std::vector<migraphx::sym::expr>{lit(0)})},
                 {"ends",
                  migraphx::to_value(
                      std::vector<migraphx::sym::expr>{migraphx::sym::min(lit(1000), n)})}}),
            data,
            start,
            end);
        mm->add_return({mm->add_instruction(migraphx::make_op("relu"), output)});
        return p;
    }

    std::unordered_map<std::string, migraphx::shape> get_test_dims() const
    {
        return {{"data", migraphx::shape{migraphx::shape::float_type, {3}}}};
    }
};
