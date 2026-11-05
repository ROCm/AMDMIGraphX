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

struct test_split_sym_dim_shared_replacement : verify_program<test_split_sym_dim_shared_replacement>
{
    migraphx::program create_program() const
    {
        using migraphx::sym::var;
        auto count = var("memo_count", {0, 4});

        migraphx::program p;
        auto* mm = p.get_main_module();
        auto condition =
            mm->add_parameter("condition", migraphx::shape{migraphx::shape::bool_type, {4}});
        auto values0 =
            mm->add_parameter("values0", migraphx::shape{migraphx::shape::float_type, {4, 2}});
        auto values1 =
            mm->add_parameter("values1", migraphx::shape{migraphx::shape::float_type, {4, 2}});
        auto starts  = mm->add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {0}});
        auto nonzero = mm->add_instruction(migraphx::make_op("nonzero"), condition);
        auto indices =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), nonzero);
        auto num_nonzero =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), nonzero);
        auto selected = mm->add_instruction(
            migraphx::make_op("dyn_slice",
                              {{"axes", {1}},
                               {"starts", {0}},
                               {"ends", migraphx::value::array{migraphx::to_value(count)}}}),
            indices,
            starts,
            num_nonzero);
        auto shared = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), selected);
        auto branch0 =
            mm->add_instruction(migraphx::make_op("gather", {{"axis", 0}}), values0, shared);
        branch0 = mm->add_instruction(migraphx::make_op("relu"), branch0);
        auto branch1 =
            mm->add_instruction(migraphx::make_op("gather", {{"axis", 0}}), values1, shared);
        branch1 = mm->add_instruction(migraphx::make_op("sigmoid"), branch1);
        auto joined =
            mm->add_instruction(migraphx::make_op("concat", {{"axis", 1}}), branch0, branch1);
        mm->add_return({branch0, branch1, joined});
        return p;
    }
};
