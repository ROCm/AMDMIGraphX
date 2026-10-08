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
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

// The index is a device literal on the gpu, so the select reads it back with
// hip::load_scalar before dispatching to the second submodule.
struct test_select_module_index : verify_program<test_select_module_index>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape data_s{migraphx::shape::float_type, {2, 4}};

        auto* sub0 = p.create_module("sub_0");
        auto x0    = sub0->add_parameter("data", data_s);
        sub0->add_return({sub0->add_instruction(migraphx::make_op("neg"), x0)});
        auto* sub1 = p.create_module("sub_1");
        auto x1    = sub1->add_parameter("data", data_s);
        sub1->add_return({sub1->add_instruction(migraphx::make_op("abs"), x1)});

        auto index = mm->add_literal(
            migraphx::literal{migraphx::shape{migraphx::shape::int64_type, {1}}, {1}});
        auto data = mm->add_parameter("data", data_s);
        auto smi  = mm->add_instruction(
            migraphx::make_op("select_module_index"), {index, data}, {sub0, sub1});
        auto ret = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), smi);
        mm->add_return({ret});
        return p;
    }
};
