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

TEST_CASE(gelu_fast_bias_test)
{
    migraphx::program p;
    auto type  = migraphx::shape::half_type;
    auto lens  = {3, 3};
    auto shape = migraphx::shape{type, lens};
    auto* mm   = p.get_main_module();
    auto x     = mm->add_parameter("x", shape);
    auto bias     = mm->add_parameter("y", shape);
    x = add_common_op(*mm, migraphx::make_op("add"), {x, bias});
    auto const1     = mm->add_literal(migraphx::literal{migraphx::shape{type}, {0.797885}});
    auto const2     = mm->add_literal(migraphx::literal{migraphx::shape{type}, {0.035677}});
    auto one          = mm->add_literal(migraphx::literal{migraphx::shape{type}, {1.0f}});
    auto half         = mm->add_literal(migraphx::literal{migraphx::shape{type}, {0.5f}});
    auto three        = mm->add_literal(migraphx::literal{migraphx::shape{type}, {3.0f}});
    // 0.035677XXX
    auto three_mbcast = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", lens}}), three);
    auto pow0             = mm->add_instruction(migraphx::make_op("pow"), {x, three_mbcast});
    auto const2_mbcast = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", lens}}), const2);
    auto mul0 = mm->add_instruction(migraphx::make_op("mul"), {pow0, const2_mbcast});

    // 0.797885X+0.035677XXX
    auto const1_mbcast = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", lens}}), const1);
    auto mul1 = mm->add_instruction(migraphx::make_op("mul"), {const1_mbcast, x});
    auto add1 = mm->add_instruction(migraphx::make_op("add"), {mul0, mul1});

    // 1+tanh(0.797885X+0.035677XXX)
    auto tanh0 = mm->add_instruction(migraphx::make_op("tanh"), add1);
    auto one_mbcast =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", lens}}), one);
    auto add2 = mm->add_instruction(migraphx::make_op("add"), {tanh0, one_mbcast});

    // 0.5X(1+tanh(0.797885X+0.035677XXX))
    auto half_mbcast =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", lens}}), half);
    auto mul2 = mm->add_instruction(migraphx::make_op("mul"), {x, half_mbcast});
    mm->add_instruction(migraphx::make_op("mul"), {add2, mul2});

    auto prog = optimize_onnx("gelu_fast_bias_test.onnx");

    EXPECT(p == prog);
}
