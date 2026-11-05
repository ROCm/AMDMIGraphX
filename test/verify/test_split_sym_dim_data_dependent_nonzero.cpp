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

struct test_split_sym_dim_data_dependent_nonzero
    : verify_program<test_split_sym_dim_data_dependent_nonzero>
{
    migraphx::program create_program() const
    {
        using migraphx::sym::var;
        auto count = var("nonzero_count", {0, 4});

        migraphx::program p;
        auto* mm  = p.get_main_module();
        auto data = mm->add_parameter("data", migraphx::shape{migraphx::shape::bool_type, {2, 2}});
        auto nonzero = mm->add_instruction(migraphx::make_op("nonzero"), data);
        auto indices =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), nonzero);
        indices = mm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::int32_type}}), indices);
        auto num_nonzero =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), nonzero);
        auto starts   = mm->add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {0}});
        auto selected = mm->add_instruction(
            migraphx::make_op("dyn_slice",
                              {{"axes", {1}},
                               {"starts", {0}},
                               {"ends", migraphx::value::array{migraphx::to_value(count)}}}),
            indices,
            starts,
            num_nonzero);
        auto converted = mm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), selected);
        auto output = mm->add_instruction(migraphx::make_op("relu"), converted);

        auto always_true   = mm->add_instruction(migraphx::make_op("equal"), data, data);
        auto always_false  = mm->add_instruction(migraphx::make_op("not"), always_true);
        auto empty_nonzero = mm->add_instruction(migraphx::make_op("nonzero"), always_false);
        auto empty_indices =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), empty_nonzero);
        empty_indices = mm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::int32_type}}),
            empty_indices);
        auto empty_count =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), empty_nonzero);
        auto empty_selected = mm->add_instruction(
            migraphx::make_op(
                "dyn_slice",
                {{"axes", {1}},
                 {"starts", {0}},
                 {"ends",
                  migraphx::value::array{migraphx::to_value(var("empty_nonzero_count", {0, 4}))}}}),
            empty_indices,
            starts,
            empty_count);
        auto empty_converted = mm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}),
            empty_selected);
        auto empty_output = mm->add_instruction(migraphx::make_op("relu"), empty_converted);
        mm->add_return({output, empty_output});
        return p;
    }
};
