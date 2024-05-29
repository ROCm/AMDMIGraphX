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

TEST_CASE(isinf_neg_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    auto t1     = mm->add_parameter("t1", s);
    auto is_inf = mm->add_instruction(migraphx::make_op("isinf"), t1);
    auto zero_l = mm->add_literal(migraphx::literal{migraphx::shape::float_type, {0}});
    auto mb_zero =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), zero_l);

    auto is_neg = mm->add_instruction(migraphx::make_op("less"), t1, mb_zero);
    if(is_neg->get_shape().type() != migraphx::shape::bool_type)
    {
        is_neg = mm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::bool_type}}), is_neg);
    }
    auto ret = mm->add_instruction(migraphx::make_op("logical_and"), is_inf, is_neg);
    mm->add_return({ret});

    auto prog = read_onnx("isinf_neg_test.onnx");
    EXPECT(p == prog);
}
