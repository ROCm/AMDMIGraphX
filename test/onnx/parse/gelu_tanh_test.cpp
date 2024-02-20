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

TEST_CASE(gelu_tanh_test)
{
    migraphx::program p;
    auto shape      = migraphx::shape{migraphx::shape::float_type, {3}};
    auto* mm        = p.get_main_module();
    auto x          = mm->add_parameter("x", shape);
    auto fit_const  = mm->add_literal(migraphx::literal{shape, {0.044708251953125}});
    auto sqrt_2_rpi = mm->add_literal(migraphx::literal{shape, {0.7978515625}});
    auto one        = mm->add_literal(migraphx::literal{shape, {1.0f}});
    auto one_half   = mm->add_literal(migraphx::literal{shape, {0.5f}});
    auto three      = mm->add_literal(migraphx::literal{shape, {3.0f}});
    auto pow0       = mm->add_instruction(migraphx::make_op("pow"), {x, three});
    auto mul0       = mm->add_instruction(migraphx::make_op("mul"), {pow0, fit_const});
    auto add0       = mm->add_instruction(migraphx::make_op("add"), {mul0, x});
    auto mul1       = mm->add_instruction(migraphx::make_op("mul"), {add0, sqrt_2_rpi});
    auto tanh0      = mm->add_instruction(migraphx::make_op("tanh"), mul1);
    auto add1       = mm->add_instruction(migraphx::make_op("add"), {tanh0, one});
    auto mul2       = mm->add_instruction(migraphx::make_op("mul"), {x, one_half});
    mm->add_instruction(migraphx::make_op("mul"), {add1, mul2});

    auto prog = optimize_onnx("gelu_tanh_test.onnx");

    EXPECT(p == prog);
}
