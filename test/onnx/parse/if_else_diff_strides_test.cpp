/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include <onnx_test.hpp>

TEST_CASE(if_else_diff_strides_test)
{
    migraphx::program expected;
    auto* mm = expected.get_main_module();
    migraphx::shape sc{migraphx::shape::bool_type, {1}};
    migraphx::shape s1{migraphx::shape::float_type, {2, 3}};
    migraphx::shape s2{migraphx::shape::float_type, {3, 2}};

    std::vector<float> ones_data(s1.elements(), 1.0f);

    auto ones   = mm->add_literal(s1, ones_data);
    auto x    = mm->add_parameter("x", s2);
    auto y    = mm->add_parameter("y", s1);
    auto cond = mm->add_parameter("cond", sc);

    auto xt = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), x);

    auto* then_mod = expected.create_module("If_5_if");
    {
        auto id        = then_mod->add_instruction(migraphx::make_op("identity"), xt);
        auto contiguous = then_mod->add_instruction(migraphx::make_op("contiguous"), id);
        then_mod->add_return({contiguous});
    }

    auto* else_mod = expected.create_module("If_5_else");
    {
        auto add        = else_mod->add_instruction(migraphx::make_op("add"), y, ones);
        else_mod->add_return({add});
    }

    auto ret = mm->add_instruction(migraphx::make_op("if"), {cond}, {then_mod, else_mod});
    auto r   = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), ret);
    mm->add_return({r});

    auto prog = read_onnx("if_else_diff_strides_test.onnx");
    EXPECT(expected.sort() == prog.sort());
}
