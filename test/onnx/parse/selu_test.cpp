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

#include <onnx_test.hpp>

TEST_CASE(selu_test)
{
    migraphx::program p;
    auto* mm                      = p.get_main_module();
    std::vector<std::size_t> lens = {2, 3};
    migraphx::shape s{migraphx::shape::double_type, lens};
    auto x = mm->add_parameter("x", s);

    migraphx::shape ls{migraphx::shape::double_type};
    auto zero  = mm->add_literal({ls, {0.0f}});
    auto one   = mm->add_literal({ls, {1.0f}});
    auto alpha = mm->add_literal({ls, {0.3f}});
    auto gamma = mm->add_literal({ls, {0.5f}});

    auto mb_zero =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", lens}}), zero);
    auto linear = mm->add_instruction(migraphx::make_op("max"), mb_zero, x);
    auto exp_x  = mm->add_instruction(migraphx::make_op("exp"), x);
    auto mb_one =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", lens}}), one);
    auto exp_sub = mm->add_instruction(migraphx::make_op("sub"), exp_x, mb_one);
    auto mb_alpha =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", lens}}), alpha);
    auto exp_mul = mm->add_instruction(migraphx::make_op("mul"), mb_alpha, exp_sub);
    auto mb_zero2 =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", lens}}), zero);
    auto exp_part = mm->add_instruction(migraphx::make_op("min"), mb_zero2, exp_mul);
    auto sum      = mm->add_instruction(migraphx::make_op("add"), linear, exp_part);
    auto mb_gamma =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", lens}}), gamma);
    auto r = mm->add_instruction(migraphx::make_op("mul"), mb_gamma, sum);
    mm->add_return({r});

    auto prog = read_onnx("selu_test.onnx");

    EXPECT(p == prog);
}
