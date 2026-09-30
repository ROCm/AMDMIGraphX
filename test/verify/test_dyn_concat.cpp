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

struct test_dyn_concat : verify_program<test_dyn_concat>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm     = p.get_main_module();
        auto x       = mm->add_parameter("x", {migraphx::shape::float_type, {2, 3, 2}});
        auto y       = mm->add_parameter("y", {migraphx::shape::float_type, {2, 2, 2}});
        auto x_count = mm->add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {2}});
        auto y_count = mm->add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {1}});
        auto result  = mm->add_instruction(
            migraphx::make_op("dyn_concat", {{"axis", 1}}), x, y, x_count, y_count);
        auto buffer =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), result);
        auto count =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), result);
        mm->add_return({buffer, count});
        return p;
    }
};
