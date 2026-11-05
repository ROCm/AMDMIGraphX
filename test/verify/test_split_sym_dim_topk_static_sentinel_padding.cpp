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
#include <migraphx/make_op.hpp>
#include <migraphx/program.hpp>
#include <migraphx/sym.hpp>

struct test_split_sym_dim_topk_static_sentinel_padding
    : verify_program<test_split_sym_dim_topk_static_sentinel_padding>
{
    migraphx::program create_program() const
    {
        using dd = migraphx::shape::dynamic_dimension;
        using migraphx::sym::var;

        auto n = var("topk_input_count", {1, 4});
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto data =
            mm->add_parameter("data", migraphx::shape{migraphx::shape::float_type, {dd{n}}});
        auto indices =
            mm->add_parameter("indices", migraphx::shape{migraphx::shape::int64_type, {dd{n}}});
        auto padded_indices =
            mm->add_instruction(migraphx::make_op("fixed_pad", {{"value", 0.0f}}), indices);
        auto largest = mm->add_instruction(
            migraphx::make_op("topk", {{"k", 4}, {"axis", 0}, {"largest", true}}),
            data,
            padded_indices);
        auto largest_values =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), largest);
        auto largest_indices =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), largest);
        auto smallest = mm->add_instruction(
            migraphx::make_op("topk", {{"k", 4}, {"axis", 0}, {"largest", false}}), data);
        auto smallest_values =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), smallest);
        mm->add_return({largest_values, largest_indices, smallest_values});
        return p;
    }

    std::unordered_map<std::string, migraphx::shape> get_test_dims() const
    {
        return {{"data", migraphx::shape{migraphx::shape::float_type, {2}}},
                {"indices", migraphx::shape{migraphx::shape::int64_type, {2}}}};
    }
};
