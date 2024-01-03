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
#include <migraphx/rewrite_gelu.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/program.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/op/convolution.hpp>
#include <migraphx/op/reshape.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/ranges.hpp>
#include <test.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/common.hpp>

#include <migraphx/serialize.hpp>

#include <migraphx/verify.hpp>

TEST_CASE(bias_gelu)
{
    migraphx::shape s1{migraphx::shape::half_type, {2, 4, 8}};
    migraphx::shape s2{migraphx::shape::half_type};
    migraphx::module m1;
    {
        auto a    = m1.add_parameter("a", s1);
        auto b    = m1.add_parameter("b", s1);
        auto add1 = m1.add_instruction(migraphx::make_op("add"), a, b);
        auto l1   = m1.add_literal(migraphx::literal{s2, {1.4140625f}});
        auto div  = add_common_op(m1, migraphx::make_op("div"), {add1, l1});
        auto erf  = m1.add_instruction(migraphx::make_op("erf"), div);
        auto l2   = m1.add_literal(migraphx::literal{s2, {1.0f}});
        auto add2 = add_common_op(m1, migraphx::make_op("add"), {erf, l2});
        auto mul  = m1.add_instruction(migraphx::make_op("mul"), add1, add2);
        auto l3   = m1.add_literal(migraphx::literal{s2, {0.5f}});
        mul       = add_common_op(m1, migraphx::make_op("mul"), {mul, l3});
        m1.add_return({mul});
    }
    migraphx::rewrite_gelu pass;
    pass.apply(m1);
    migraphx::dead_code_elimination dce;
    dce.apply(m1);

    migraphx::module m2;
    {
        auto a   = m2.add_parameter("a", s1);
        auto b   = m2.add_parameter("b", s1);
        auto add = m2.add_instruction(migraphx::make_op("add"), a, b);
        auto l1  = m2.add_literal(migraphx::literal{s2, {1.702f}});
        auto mul = add_common_op(m2, migraphx::make_op("mul"), {add, l1});
        auto sig = m2.add_instruction(migraphx::make_op("neg"), mul);
        sig      = m2.add_instruction(migraphx::make_op("exp"), sig);
        auto l2  = m2.add_literal(migraphx::literal{s2, {1.0f}});
        sig      = add_common_op(m2, migraphx::make_op("add"), {sig, l2});
        sig      = m2.add_instruction(migraphx::make_op("div"), add, sig);
        m2.add_return({sig});
    }

    EXPECT(m1 == m2);
}

TEST_CASE(non_bias_gelu)
{
    migraphx::shape s1{migraphx::shape::half_type, {2, 4, 8}};
    migraphx::shape s2{migraphx::shape::half_type};
    migraphx::module m1;
    {
        auto a    = m1.add_parameter("a", s1);
        auto b    = m1.add_parameter("b", s1);
        auto sub  = m1.add_instruction(migraphx::make_op("sub"), a, b);
        auto l1   = m1.add_literal(migraphx::literal{s2, {1.4140625f}});
        auto div  = add_common_op(m1, migraphx::make_op("div"), {sub, l1});
        auto erf  = m1.add_instruction(migraphx::make_op("erf"), div);
        auto l2   = m1.add_literal(migraphx::literal{s2, {1.0f}});
        auto add2 = add_common_op(m1, migraphx::make_op("add"), {erf, l2});
        auto mul  = m1.add_instruction(migraphx::make_op("mul"), sub, add2);
        auto l3   = m1.add_literal(migraphx::literal{s2, {0.5f}});
        mul       = add_common_op(m1, migraphx::make_op("mul"), {mul, l3});
        m1.add_return({mul});
    }
    migraphx::rewrite_gelu pass;
    pass.apply(m1);
    migraphx::dead_code_elimination dce;
    dce.apply(m1);

    migraphx::module m2;
    {
        auto a   = m2.add_parameter("a", s1);
        auto b   = m2.add_parameter("b", s1);
        auto sub = m2.add_instruction(migraphx::make_op("sub"), a, b);
        auto l1  = m2.add_literal(migraphx::literal{s2, {1.702f}});
        auto mul = add_common_op(m2, migraphx::make_op("mul"), {sub, l1});
        auto sig = m2.add_instruction(migraphx::make_op("neg"), mul);
        sig      = m2.add_instruction(migraphx::make_op("exp"), sig);
        auto l2  = m2.add_literal(migraphx::literal{s2, {1.0f}});
        sig      = add_common_op(m2, migraphx::make_op("add"), {sig, l2});
        sig      = m2.add_instruction(migraphx::make_op("div"), sub, sig);
        m2.add_return({sig});
    }

    EXPECT(m1 == m2);
}

TEST_CASE(tanh_gelu_distilgpt2_fp16)
{
    // Uses constant values seen in the distilgpt2_fp16 model, note how they're not exactly right
    migraphx::shape s1{migraphx::shape::half_type, {2, 4, 8}};
    migraphx::shape s2{migraphx::shape::half_type};
    migraphx::module m1;
    {
        auto x          = m1.add_parameter("x", s1);
        auto fit_const  = m1.add_literal(migraphx::literal{s2, {0.044708251953125}});
        auto sqrt_2_rpi = m1.add_literal(migraphx::literal{s2, {0.7978515625}});
        auto one        = m1.add_literal(migraphx::literal{s2, {1.0f}});
        auto one_half   = m1.add_literal(migraphx::literal{s2, {0.5f}});
        auto three      = m1.add_literal(migraphx::literal{s2, {3.0f}});
        auto pow0       = add_common_op(m1, migraphx::make_op("pow"), {x, three});
        auto mul0       = add_common_op(m1, migraphx::make_op("mul"), {pow0, fit_const});
        auto add0       = m1.add_instruction(migraphx::make_op("add"), {mul0, x});
        auto mul1       = add_common_op(m1, migraphx::make_op("mul"), {add0, sqrt_2_rpi});
        auto tanh0      = m1.add_instruction(migraphx::make_op("tanh"), mul1);
        auto add1       = add_common_op(m1, migraphx::make_op("add"), {tanh0, one});
        auto mul2       = add_common_op(m1, migraphx::make_op("mul"), {x, one_half});
        auto y          = m1.add_instruction(migraphx::make_op("mul"), {add1, mul2});
        m1.add_return({y});
    }
    migraphx::rewrite_gelu pass;
    pass.apply(m1);
    migraphx::dead_code_elimination dce;
    dce.apply(m1);

    /* erf() version
    migraphx::module m2;
    {
        auto x        = m2.add_parameter("x", s1);
        auto sqrt1_2  = m2.add_literal(migraphx::literal{s2, {M_SQRT1_2}});
        auto one      = m2.add_literal(migraphx::literal{s2, {1.0f}});
        auto one_half = m2.add_literal(migraphx::literal{s2, {0.5f}});
        auto a        = add_common_op(m2, migraphx::make_op("mul"), {x, sqrt1_2});
        auto erf      = m2.add_instruction(migraphx::make_op("erf"), a);
        auto add_erf  = add_common_op(m2, migraphx::make_op("add"), {one, erf});
        auto b        = add_common_op(m2, migraphx::make_op("mul"), {one_half, add_erf});
        auto y        = m2.add_instruction(migraphx::make_op("mul"), x, b);
        m2.add_return({y});
    }
    */
    migraphx::module m2;
    {
        auto x          = m2.add_parameter("x", s1);
        auto sqrt_2_rpi = m2.add_literal(migraphx::literal{
            migraphx::shape{x->get_shape().type()},
            {0.7978845608028653558798921198687637369517172623298693153318516593}});
        auto fit_const =
            m2.add_literal(migraphx::literal{migraphx::shape{x->get_shape().type()}, {0.044715f}});
        auto one =
            m2.add_literal(migraphx::literal{migraphx::shape{x->get_shape().type()}, {1.0f}});
        auto xb    = add_common_op(m2, migraphx::make_op("mul"), {x, sqrt_2_rpi});
        auto a     = add_common_op(m2, migraphx::make_op("mul"), {xb, fit_const});
        auto b     = m2.add_instruction(migraphx::make_op("mul"), a, x);
        auto c     = m2.add_instruction(migraphx::make_op("mul"), b, x);
        auto u     = m2.add_instruction(migraphx::make_op("add"), c, xb);
        auto neg_u = m2.add_instruction(migraphx::make_op("neg"), u);
        auto d     = m2.add_instruction(migraphx::make_op("sub"), neg_u, u);
        auto emu   = m2.add_instruction(migraphx::make_op("exp"), d);
        auto e     = add_common_op(m2, migraphx::make_op("add"), {one, emu});
        auto cdf   = add_common_op(m2, migraphx::make_op("div"), {one, e});
        auto y     = m2.add_instruction(migraphx::make_op("mul"), x, cdf);
        m2.add_return({y});
    }

    EXPECT(m1 == m2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
