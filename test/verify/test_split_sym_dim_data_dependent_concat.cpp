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
#include <migraphx/serialize.hpp>
#include <migraphx/sym.hpp>

struct test_split_sym_dim_data_dependent_concat
    : verify_program<test_split_sym_dim_data_dependent_concat>
{
    migraphx::program create_program() const
    {
        using migraphx::sym::var;
        auto a            = var("a", {0, 2});
        auto b            = var("b", {0, 2});
        auto concat_count = var("concat_count", {0, 4});

        migraphx::program p;
        auto* mm        = p.get_main_module();
        auto condition0 = mm->add_parameter("condition0", {migraphx::shape::bool_type, {2}});
        auto condition1 = mm->add_parameter("condition1", {migraphx::shape::bool_type, {2}});
        auto data0      = mm->add_parameter("data0", {migraphx::shape::float_type, {2, 2}});
        auto data1      = mm->add_parameter("data1", {migraphx::shape::float_type, {2, 2}});
        auto starts = mm->add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {0}});

        auto nonzero0 = mm->add_instruction(migraphx::make_op("nonzero"), condition0);
        auto buffer0 =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), nonzero0);
        auto count0 =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), nonzero0);
        auto indices0 = mm->add_instruction(
            migraphx::make_op("dyn_slice",
                              {{"axes", {1}},
                               {"starts", {0}},
                               {"ends", migraphx::value::array{migraphx::to_value(a)}}}),
            buffer0,
            starts,
            count0);
        indices0 = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), indices0);
        auto gathered0 =
            mm->add_instruction(migraphx::make_op("gather", {{"axis", 0}}), data0, indices0);

        auto nonzero1 = mm->add_instruction(migraphx::make_op("nonzero"), condition1);
        auto buffer1 =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), nonzero1);
        auto count1 =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), nonzero1);
        auto indices1 = mm->add_instruction(
            migraphx::make_op("dyn_slice",
                              {{"axes", {1}},
                               {"starts", {0}},
                               {"ends", migraphx::value::array{migraphx::to_value(b)}}}),
            buffer1,
            starts,
            count1);
        indices1 = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), indices1);
        auto gathered1 =
            mm->add_instruction(migraphx::make_op("gather", {{"axis", 0}}), data1, indices1);

        auto counts = mm->add_instruction(
            migraphx::make_op(
                "eval_expr_from_shape",
                {{"expressions", migraphx::to_value(std::vector<migraphx::sym::expr>{a, b})}}),
            gathered0,
            gathered1);
        auto concat = mm->add_instruction(
            migraphx::make_op("dyn_concat", {{"axis", 0}}), counts, gathered0, gathered1);
        auto concat_buffer =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), concat);
        auto total =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), concat);
        auto output = mm->add_instruction(
            migraphx::make_op("dyn_slice",
                              {{"axes", {0}},
                               {"starts", {0}},
                               {"ends", migraphx::value::array{migraphx::to_value(concat_count)}}}),
            concat_buffer,
            starts,
            total);
        mm->add_return({output});
        return p;
    }
};
