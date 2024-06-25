/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
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

TEST_CASE(if_literal_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape cond_s{migraphx::shape::bool_type};
    auto cond = mm->add_parameter("cond", cond_s);

    migraphx::shape s{migraphx::shape::float_type, {5}};

    auto* then_mod           = p.create_module("If_1_if");
    std::vector<float> data1 = {1, 2, 3, 4, 5};
    auto l1                  = then_mod->add_literal(migraphx::literal(s, data1));
    then_mod->add_literal({});
    then_mod->add_return({l1});

    auto* else_mod           = p.create_module("If_1_else");
    std::vector<float> data2 = {5, 4, 3, 2, 1};
    auto l2                  = else_mod->add_literal(migraphx::literal(s, data2));
    else_mod->add_literal({});
    else_mod->add_return({l2});

    auto ret = mm->add_instruction(migraphx::make_op("if"), {cond}, {then_mod, else_mod});
    auto r   = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), ret);
    mm->add_return({r});

    auto prog = read_onnx("if_literal_test.onnx");
    EXPECT(p == prog);
}
