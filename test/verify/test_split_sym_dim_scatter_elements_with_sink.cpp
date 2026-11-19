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

struct test_split_sym_dim_scatter_elements_with_sink
    : verify_program<test_split_sym_dim_scatter_elements_with_sink>
{
    migraphx::program create_program() const
    {
        using dd = migraphx::shape::dynamic_dimension;
        using migraphx::sym::lit;
        using migraphx::sym::var;

        auto data_count   = var("scatter_data_count", {1, 4}, {2});
        auto update_count = var("scatter_update_count", {0, 4}, {1});
        auto index_count  = var("scatter_index_count", {0, 4}, {2});
        migraphx::program p;
        auto* mm  = p.get_main_module();
        auto data = mm->add_parameter(
            "data", migraphx::shape{migraphx::shape::float_type, {dd{data_count}, dd{lit(2)}}});
        auto condition =
            mm->add_parameter("condition", migraphx::shape{migraphx::shape::bool_type, {4}});
        auto update_values = mm->add_parameter(
            "update_values", migraphx::shape{migraphx::shape::float_type, {4, 2}});
        auto starts = mm->add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {0}});

        auto update_nonzero = mm->add_instruction(migraphx::make_op("nonzero"), condition);
        auto update_indices = mm->add_instruction(
            migraphx::make_op("get_tuple_elem", {{"index", 0}}), update_nonzero);
        auto update_size = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}),
                                               update_nonzero);
        update_indices   = mm->add_instruction(
            migraphx::make_op("dyn_slice",
                                {{"axes", {1}},
                                 {"starts", {0}},
                                 {"ends", migraphx::value::array{migraphx::to_value(update_count)}}}),
            update_indices,
            starts,
            update_size);
        update_indices =
            mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), update_indices);
        auto updates = mm->add_instruction(
            migraphx::make_op("gather", {{"axis", 0}}), update_values, update_indices);

        auto index_nonzero = mm->add_instruction(migraphx::make_op("nonzero"), condition);
        auto indices =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), index_nonzero);
        auto index_size =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), index_nonzero);
        indices = mm->add_instruction(
            migraphx::make_op("dyn_slice",
                              {{"axes", {1}},
                               {"starts", {0}},
                               {"ends", migraphx::value::array{migraphx::to_value(index_count)}}}),
            indices,
            starts,
            index_size);
        indices =
            mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), indices);
        auto dims = mm->add_instruction(
            migraphx::make_op("dimensions_of", {{"start", 0}, {"end", 2}}), updates);
        indices = mm->add_instruction(
            migraphx::make_op("broadcast_with_dims",
                              {{"out_dyn_dims",
                                migraphx::to_value(std::vector<dd>{dd{index_count}, dd{lit(2)}})}}),
            indices,
            dims);
        indices = mm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::int32_type}}), indices);
        indices        = mm->add_instruction(migraphx::make_op("contiguous"), indices);
        auto scattered = mm->add_instruction(
            migraphx::make_op("scatter_none", {{"axis", 0}}), data, indices, updates);
        auto output = mm->add_instruction(migraphx::make_op("relu"), scattered);
        mm->add_return({output});
        return p;
    }

    std::unordered_map<std::string, migraphx::shape> get_test_dims() const
    {
        return {{"data", migraphx::shape{migraphx::shape::float_type, {4, 2}}}};
    }
};
