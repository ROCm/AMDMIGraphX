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
#include <cstdint>
#include <vector>

struct test_split_sym_dim_integral_prefix_scan
    : verify_program<test_split_sym_dim_integral_prefix_scan>
{
    migraphx::program create_program() const
    {
        using dd = migraphx::shape::dynamic_dimension;
        using migraphx::sym::lit;
        using migraphx::sym::var;

        auto n = var("n", {1, 4}, {2});
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto shape_source =
            mm->add_parameter("shape_source", {migraphx::shape::uint8_type, {dd{n}}});
        auto data =
            mm->add_literal(migraphx::literal{{migraphx::shape::uint8_type, {4}}, {255, 0, 0, 0}});
        auto start = mm->add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {0}});
        auto end   = mm->add_instruction(
            migraphx::make_op("dimensions_of", {{"start", 0}, {"end", 1}}), shape_source);
        auto input = mm->add_instruction(
            migraphx::make_op(
                "dyn_slice",
                {{"axes", {0}},
                 {"starts", migraphx::to_value(std::vector<migraphx::sym::expr>{lit(0)})},
                 {"ends", migraphx::to_value(std::vector<migraphx::sym::expr>{n})}}),
            data,
            start,
            end);
        auto values =
            mm->add_instruction(migraphx::make_op("prefix_scan_sum", {{"axis", 0}}), input);
        auto output = mm->add_instruction(migraphx::make_op("reduce_min", {{"axes", {0}}}), values);
        mm->add_return({output});
        return p;
    }

    std::unordered_map<std::string, migraphx::shape> get_test_dims() const
    {
        return {{"shape_source", migraphx::shape{migraphx::shape::uint8_type, {3}}}};
    }
};
