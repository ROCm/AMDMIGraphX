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

TEST_CASE(gelu_default_test)
{
    migraphx::program p;
    auto type  = migraphx::shape::float_type;
    auto lens  = {3, 3};
    auto shape = migraphx::shape{type, lens};
    auto* mm   = p.get_main_module();
    auto x     = mm->add_parameter("x", shape);
    auto half  = mm->add_literal(migraphx::literal{migraphx::shape{type}, {0.5f}});
    auto one   = mm->add_literal(migraphx::literal{migraphx::shape{type}, {1.0f}});
    auto sqrt2 =
        mm->add_literal(migraphx::literal{migraphx::shape{type}, {static_cast<float>(M_SQRT2)}});
    auto half_mbcast =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", lens}}), half);
    auto mul_half = mm->add_instruction(migraphx::make_op("mul"), x, half_mbcast);
    auto sqrt2_mbcast =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", lens}}), sqrt2);
    auto div = mm->add_instruction(migraphx::make_op("div"), x, sqrt2_mbcast);
    auto erf = mm->add_instruction(migraphx::make_op("erf"), div);
    auto one_mbcast =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", lens}}), one);
    auto add_one = mm->add_instruction(migraphx::make_op("add"), erf, one_mbcast);
    mm->add_instruction(migraphx::make_op("mul"), mul_half, add_one);

    auto prog = optimize_onnx("gelu_default_test.onnx");

    EXPECT(p == prog);
}
