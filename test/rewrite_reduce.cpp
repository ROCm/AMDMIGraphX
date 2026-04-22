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
#include <migraphx/rewrite_reduce.hpp>
#include <migraphx/algorithm.hpp>
#include <migraphx/common.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/module.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/program.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <test.hpp>

static void run_pass(migraphx::module& m)
{
    migraphx::run_passes(m, {migraphx::rewrite_reduce{}, migraphx::dead_code_elimination{}});
}

TEST_CASE(softmax)
{
    migraphx::shape s{migraphx::shape::float_type, {10, 1000}};
    migraphx::module m;
    auto x       = m.add_parameter("x", s);
    auto softmax = m.add_instruction(migraphx::make_op("softmax", {{"axis", 1}}), x);
    m.add_return({softmax});
    run_pass(m);
    EXPECT(none_of(migraphx::iterator_for(m), [](auto ins) { return ins->name() == "softmax"; }));

    auto reduces = find_all(migraphx::iterator_for(m),
                            [&](auto ins) { return migraphx::contains(ins->name(), "reduce"); });
    EXPECT(all_of(reduces, [](auto ins) {
        auto axes = ins->get_operator().to_value()["axes"].template to_vector<int64_t>();
        return axes.size() == 1 and axes[0] == 1;
    }));
}

TEST_CASE(softmax_upcast)
{
    migraphx::shape s{migraphx::shape::half_type, {10, 1000}};
    migraphx::module m;
    auto x       = m.add_parameter("x", s);
    auto softmax = m.add_instruction(migraphx::make_op("softmax", {{"axis", 1}}), x);
    m.add_return({softmax});
    run_pass(m);
    EXPECT(none_of(migraphx::iterator_for(m), [](auto ins) { return ins->name() == "softmax"; }));

    auto reduces = find_all(migraphx::iterator_for(m),
                            [&](auto ins) { return migraphx::contains(ins->name(), "reduce"); });
    EXPECT(all_of(reduces, [](auto ins) {
        auto axes  = ins->get_operator().to_value()["axes"].template to_vector<int64_t>();
        auto dtype = ins->get_shape().type();
        return axes.size() == 1 and axes[0] == 1 and dtype == migraphx::shape::float_type;
    }));
}

// Skinny dot [M=1, K] @ [K, N] gets rewritten to mul + reduce_sum.
TEST_CASE(dot_skinny_rewrite)
{
    migraphx::shape a_shape{migraphx::shape::float_type, {1, 128}};
    migraphx::shape b_shape{migraphx::shape::float_type, {128, 4}};
    migraphx::module m1;
    {
        auto a   = m1.add_parameter("a", a_shape);
        auto b   = m1.add_parameter("b", b_shape);
        auto dot = m1.add_instruction(migraphx::make_op("dot"), a, b);
        m1.add_return({dot});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto a = m2.add_parameter("a", a_shape);
        auto b = m2.add_parameter("b", b_shape);
        auto a_unsq =
            m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), a);
        auto b_unsq =
            m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), b);
        auto b_trans = m2.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {1, 2, 0}}}), b_unsq);
        auto a_bc = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 4, 128}}}), a_unsq);
        auto mul = m2.add_instruction(migraphx::make_op("mul"), a_bc, b_trans);
        auto red = m2.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), mul);
        auto sq  = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {2}}}), red);
        m2.add_return({sq});
    }
    EXPECT(m1.sort() == m2.sort());
}

// Skinny dot with M=2 also gets rewritten.
TEST_CASE(dot_skinny_m2_rewrite)
{
    migraphx::shape a_shape{migraphx::shape::float_type, {2, 128}};
    migraphx::shape b_shape{migraphx::shape::float_type, {128, 4}};
    migraphx::module m1;
    {
        auto a   = m1.add_parameter("a", a_shape);
        auto b   = m1.add_parameter("b", b_shape);
        auto dot = m1.add_instruction(migraphx::make_op("dot"), a, b);
        m1.add_return({dot});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto a = m2.add_parameter("a", a_shape);
        auto b = m2.add_parameter("b", b_shape);
        auto a_unsq =
            m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), a);
        auto a_bc = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 4, 128}}}), a_unsq);
        auto b_trans = m2.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {1, 0}}}), b);
        auto b_bc = m2.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {2, 4, 128}}}), b_trans);
        auto mul = m2.add_instruction(migraphx::make_op("mul"), a_bc, b_bc);
        auto red = m2.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), mul);
        auto sq  = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {2}}}), red);
        m2.add_return({sq});
    }
    EXPECT(m1.sort() == m2.sort());
}

// rows > 2 exceeds the skinny threshold so the dot is left alone.
TEST_CASE(dot_wide_no_rewrite)
{
    migraphx::shape a_shape{migraphx::shape::float_type, {3, 128}};
    migraphx::shape b_shape{migraphx::shape::float_type, {128, 4}};
    migraphx::module m1;
    {
        auto a   = m1.add_parameter("a", a_shape);
        auto b   = m1.add_parameter("b", b_shape);
        auto dot = m1.add_instruction(migraphx::make_op("dot"), a, b);
        m1.add_return({dot});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto a   = m2.add_parameter("a", a_shape);
        auto b   = m2.add_parameter("b", b_shape);
        auto dot = m2.add_instruction(migraphx::make_op("dot"), a, b);
        m2.add_return({dot});
    }
    EXPECT(m1.sort() == m2.sort());
}

// Batched skinny dot [B, M=1, K] @ [B, K, N] gets rewritten.
TEST_CASE(dot_batched_skinny_rewrite)
{
    migraphx::shape a_shape{migraphx::shape::float_type, {4, 1, 128}};
    migraphx::shape b_shape{migraphx::shape::float_type, {4, 128, 8}};
    migraphx::module m1;
    {
        auto a   = m1.add_parameter("a", a_shape);
        auto b   = m1.add_parameter("b", b_shape);
        auto dot = m1.add_instruction(migraphx::make_op("dot"), a, b);
        m1.add_return({dot});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto a = m2.add_parameter("a", a_shape);
        auto b = m2.add_parameter("b", b_shape);
        auto a_unsq =
            m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), a);
        auto b_unsq =
            m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), b);
        auto b_trans = m2.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), b_unsq);
        auto a_bc = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {4, 1, 8, 128}}}), a_unsq);
        auto mul = m2.add_instruction(migraphx::make_op("mul"), a_bc, b_trans);
        auto red = m2.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {3}}}), mul);
        auto sq  = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {3}}}), red);
        m2.add_return({sq});
    }
    EXPECT(m1.sort() == m2.sort());
}

// Batched dot with M=2 gets rewritten too.
TEST_CASE(dot_batched_m2_rewrite)
{
    migraphx::shape a_shape{migraphx::shape::float_type, {1, 12, 2, 128}};
    migraphx::shape b_shape{migraphx::shape::float_type, {1, 12, 128, 64}};
    migraphx::module m1;
    {
        auto a   = m1.add_parameter("a", a_shape);
        auto b   = m1.add_parameter("b", b_shape);
        auto dot = m1.add_instruction(migraphx::make_op("dot"), a, b);
        m1.add_return({dot});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto a = m2.add_parameter("a", a_shape);
        auto b = m2.add_parameter("b", b_shape);
        auto a_unsq =
            m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {3}}}), a);
        auto a_bc = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 12, 2, 64, 128}}}), a_unsq);
        auto b_unsq =
            m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), b);
        auto b_trans = m2.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 1, 2, 4, 3}}}), b_unsq);
        auto b_bc = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 12, 2, 64, 128}}}), b_trans);
        auto mul = m2.add_instruction(migraphx::make_op("mul"), a_bc, b_bc);
        auto red = m2.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {4}}}), mul);
        auto sq  = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {4}}}), red);
        m2.add_return({sq});
    }
    EXPECT(m1.sort() == m2.sort());
}

// Batched dot feeding a softmax that returns (no downstream dot) is not
// attention; find_dot rewrites the dot and find_softmax decomposes the softmax.
// Using float_type avoids the fp16->fp32 upcast wrapping.
TEST_CASE(dot_softmax_return_rewrite)
{
    migraphx::shape a_shape{migraphx::shape::float_type, {1, 12, 1, 128}};
    migraphx::shape b_shape{migraphx::shape::float_type, {1, 12, 128, 128}};
    migraphx::module m1;
    {
        auto a       = m1.add_parameter("a", a_shape);
        auto b       = m1.add_parameter("b", b_shape);
        auto dot     = m1.add_instruction(migraphx::make_op("dot"), a, b);
        auto softmax = m1.add_instruction(migraphx::make_op("softmax", {{"axis", 3}}), dot);
        m1.add_return({softmax});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto a = m2.add_parameter("a", a_shape);
        auto b = m2.add_parameter("b", b_shape);
        auto a_unsq =
            m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {3}}}), a);
        auto b_unsq =
            m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {3}}}), b);
        auto b_trans = m2.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 4, 2}}}), b_unsq);
        auto a_bc = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 12, 1, 128, 128}}}), a_unsq);
        auto mul = m2.add_instruction(migraphx::make_op("mul"), a_bc, b_trans);
        auto red = m2.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {4}}}), mul);
        auto sq  = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {4}}}), red);
        auto rmax =
            m2.add_instruction(migraphx::make_op("reduce_max", {{"axes", {3}}}), sq);
        auto rmax_bc = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 12, 1, 128}}}), rmax);
        auto sub  = m2.add_instruction(migraphx::make_op("sub"), sq, rmax_bc);
        auto exp  = m2.add_instruction(migraphx::make_op("exp"), sub);
        auto rsum = m2.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {3}}}), exp);
        auto rsum_bc = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 12, 1, 128}}}), rsum);
        auto div = m2.add_instruction(migraphx::make_op("div"), exp, rsum_bc);
        m2.add_return({div});
    }
    EXPECT(m1.sort() == m2.sort());
}

// dot -> mul -> softmax -> return: still not attention (no dot after softmax).
TEST_CASE(dot_mul_softmax_return_rewrite)
{
    migraphx::shape a_shape{migraphx::shape::float_type, {1, 12, 1, 128}};
    migraphx::shape b_shape{migraphx::shape::float_type, {1, 12, 128, 128}};
    migraphx::shape scale_shape{migraphx::shape::float_type, {1}};
    migraphx::module m1;
    {
        auto a        = m1.add_parameter("a", a_shape);
        auto b        = m1.add_parameter("b", b_shape);
        auto scale    = m1.add_parameter("scale", scale_shape);
        auto scale_bc = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 12, 1, 128}}}), scale);
        auto dot     = m1.add_instruction(migraphx::make_op("dot"), a, b);
        auto mul     = m1.add_instruction(migraphx::make_op("mul"), dot, scale_bc);
        auto softmax = m1.add_instruction(migraphx::make_op("softmax", {{"axis", 3}}), mul);
        m1.add_return({softmax});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto a        = m2.add_parameter("a", a_shape);
        auto b        = m2.add_parameter("b", b_shape);
        auto scale    = m2.add_parameter("scale", scale_shape);
        auto scale_bc = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 12, 1, 128}}}), scale);
        auto a_unsq =
            m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {3}}}), a);
        auto b_unsq =
            m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {3}}}), b);
        auto b_trans = m2.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 4, 2}}}), b_unsq);
        auto a_bc = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 12, 1, 128, 128}}}), a_unsq);
        auto mul_ab = m2.add_instruction(migraphx::make_op("mul"), a_bc, b_trans);
        auto red = m2.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {4}}}), mul_ab);
        auto sq  = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {4}}}), red);
        auto mul_scale = m2.add_instruction(migraphx::make_op("mul"), sq, scale_bc);
        auto rmax      = m2.add_instruction(
            migraphx::make_op("reduce_max", {{"axes", {3}}}), mul_scale);
        auto rmax_bc = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 12, 1, 128}}}), rmax);
        auto sub  = m2.add_instruction(migraphx::make_op("sub"), mul_scale, rmax_bc);
        auto exp  = m2.add_instruction(migraphx::make_op("exp"), sub);
        auto rsum = m2.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {3}}}), exp);
        auto rsum_bc = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 12, 1, 128}}}), rsum);
        auto div = m2.add_instruction(migraphx::make_op("div"), exp, rsum_bc);
        m2.add_return({div});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(softmax_dot_scale_where_fp32_convert_after)
{
    migraphx::shape dot_shape{migraphx::shape::half_type, {1, 12, 1, 128}};
    migraphx::shape k_shape{migraphx::shape::half_type, {1, 12, 128, 128}};
    migraphx::shape v_shape{migraphx::shape::half_type, {1, 12, 128, 128}};
    migraphx::shape scale_shape{migraphx::shape::half_type, {1}};
    migraphx::shape mask_shape{migraphx::shape::bool_type, {1, 12, 1, 128}};
    migraphx::shape f32_dot_shape{migraphx::shape::float_type, dot_shape.lens()};

    auto make_dot = [](auto& mod, auto dot_shape, auto k_shape, auto scale_shape, auto mask_shape) {
        auto q     = mod.add_parameter("q", dot_shape);
        auto k     = mod.add_parameter("k", k_shape);
        auto scale = mod.add_parameter("scale", scale_shape);
        auto mask  = mod.add_parameter("mask", mask_shape);
        auto ninf =
            mod.add_literal(migraphx::literal{migraphx::shape{migraphx::shape::half_type, {1}},
                                              {-std::numeric_limits<float>::infinity()}});
        auto ninf_bc = mod.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", dot_shape.lens()}}), ninf);
        auto scale_bc = mod.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", dot_shape.lens()}}), scale);
        auto dot = mod.add_instruction(migraphx::make_op("dot"), q, k);
        return std::make_tuple(dot, scale_bc, mask, ninf_bc);
    };

    // Input module: dot -> mul -> where -> softmax -> dot(V)
    migraphx::module m1;
    {
        auto [dot, scale_bc, mask, ninf_bc] =
            make_dot(m1, dot_shape, k_shape, scale_shape, mask_shape);
        auto v       = m1.add_parameter("v", v_shape);
        auto mul     = m1.add_instruction(migraphx::make_op("mul"), dot, scale_bc);
        auto where   = m1.add_instruction(migraphx::make_op("where"), mask, ninf_bc, mul);
        auto softmax = m1.add_instruction(migraphx::make_op("softmax", {{"axis", 3}}), where);
        auto dot_v   = m1.add_instruction(migraphx::make_op("dot"), softmax, v);
        m1.add_return({dot_v});
    }

    // Expected module: dot(f16) -> convert(f32) -> mul(f32) -> where(f32) ->
    // softmax_decomposed(f32) -> convert(f16) -> dot(V)
    migraphx::module m2;
    {
        auto [dot, scale_bc, mask, ninf_bc] =
            make_dot(m2, dot_shape, k_shape, scale_shape, mask_shape);
        auto v       = m2.add_parameter("v", v_shape);
        auto cvt_dot = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), dot);
        auto cvt_scale = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), scale_bc);
        auto mul      = m2.add_instruction(migraphx::make_op("mul"), cvt_dot, cvt_scale);
        auto cvt_ninf = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), ninf_bc);
        auto where   = m2.add_instruction(migraphx::make_op("where"), mask, cvt_ninf, mul);
        auto rmax    = m2.add_instruction(migraphx::make_op("reduce_max", {{"axes", {3}}}), where);
        auto rmax_bc = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", f32_dot_shape.lens()}}), rmax);
        auto sub     = m2.add_instruction(migraphx::make_op("sub"), where, rmax_bc);
        auto exp     = m2.add_instruction(migraphx::make_op("exp"), sub);
        auto rsum    = m2.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {3}}}), exp);
        auto rsum_bc = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", f32_dot_shape.lens()}}), rsum);
        auto div     = m2.add_instruction(migraphx::make_op("div"), exp, rsum_bc);
        auto cvt_out = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), div);
        auto dot_v = m2.add_instruction(migraphx::make_op("dot"), cvt_out, v);
        m2.add_return({dot_v});
    }

    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(softmax_dot_scale_fp32_convert_after)
{
    migraphx::shape dot_shape{migraphx::shape::half_type, {1, 12, 1, 128}};
    migraphx::shape k_shape{migraphx::shape::half_type, {1, 12, 128, 128}};
    migraphx::shape v_shape{migraphx::shape::half_type, {1, 12, 128, 128}};
    migraphx::shape scale_shape{migraphx::shape::half_type, {1}};
    migraphx::shape f32_dot_shape{migraphx::shape::float_type, dot_shape.lens()};

    // Input module: dot -> mul -> softmax -> dot(V)
    migraphx::module m1;
    auto q1        = m1.add_parameter("q", dot_shape);
    auto k1        = m1.add_parameter("k", k_shape);
    auto v1        = m1.add_parameter("v", v_shape);
    auto scale1    = m1.add_parameter("scale", scale_shape);
    auto scale_bc1 = m1.add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", dot_shape.lens()}}), scale1);
    auto dot1     = m1.add_instruction(migraphx::make_op("dot"), q1, k1);
    auto mul1     = m1.add_instruction(migraphx::make_op("mul"), dot1, scale_bc1);
    auto softmax1 = m1.add_instruction(migraphx::make_op("softmax", {{"axis", 3}}), mul1);
    auto dot_v1   = m1.add_instruction(migraphx::make_op("dot"), softmax1, v1);
    m1.add_return({dot_v1});

    // Expected module: dot(f16) -> convert(f32) -> mul(f32) -> softmax_decomposed(f32) ->
    // convert(f16) -> dot(V)
    migraphx::module m2;
    auto q2        = m2.add_parameter("q", dot_shape);
    auto k2        = m2.add_parameter("k", k_shape);
    auto v2        = m2.add_parameter("v", v_shape);
    auto scale2    = m2.add_parameter("scale", scale_shape);
    auto scale_bc2 = m2.add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", dot_shape.lens()}}), scale2);
    auto dot2     = m2.add_instruction(migraphx::make_op("dot"), q2, k2);
    auto cvt_dot2 = m2.add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), dot2);
    auto cvt_scale2 = m2.add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), scale_bc2);
    auto mul2     = m2.add_instruction(migraphx::make_op("mul"), cvt_dot2, cvt_scale2);
    auto rmax2    = m2.add_instruction(migraphx::make_op("reduce_max", {{"axes", {3}}}), mul2);
    auto rmax_bc2 = m2.add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", f32_dot_shape.lens()}}), rmax2);
    auto sub2     = m2.add_instruction(migraphx::make_op("sub"), mul2, rmax_bc2);
    auto exp2     = m2.add_instruction(migraphx::make_op("exp"), sub2);
    auto rsum2    = m2.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {3}}}), exp2);
    auto rsum_bc2 = m2.add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", f32_dot_shape.lens()}}), rsum2);
    auto div2     = m2.add_instruction(migraphx::make_op("div"), exp2, rsum_bc2);
    auto cvt_out2 = m2.add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), div2);
    auto dot_v2 = m2.add_instruction(migraphx::make_op("dot"), cvt_out2, v2);
    m2.add_return({dot_v2});

    run_pass(m1);
    EXPECT(m1 == m2);
}

// Verify dot is found directly when feeding softmax with no mul/where
TEST_CASE(softmax_dot_only)
{
    migraphx::shape dot_shape{migraphx::shape::half_type, {1, 12, 1, 128}};
    migraphx::shape k_shape{migraphx::shape::half_type, {1, 12, 128, 128}};
    migraphx::shape v_shape{migraphx::shape::half_type, {1, 12, 128, 128}};
    migraphx::shape f32_dot_shape{migraphx::shape::float_type, dot_shape.lens()};

    migraphx::module m1;
    auto q1      = m1.add_parameter("q", dot_shape);
    auto k1      = m1.add_parameter("k", k_shape);
    auto v1      = m1.add_parameter("v", v_shape);
    auto dot1    = m1.add_instruction(migraphx::make_op("dot"), q1, k1);
    auto softmax = m1.add_instruction(migraphx::make_op("softmax", {{"axis", 3}}), dot1);
    auto dot_v1  = m1.add_instruction(migraphx::make_op("dot"), softmax, v1);
    m1.add_return({dot_v1});

    migraphx::module m2;
    auto q2      = m2.add_parameter("q", dot_shape);
    auto k2      = m2.add_parameter("k", k_shape);
    auto v2      = m2.add_parameter("v", v_shape);
    auto dot2    = m2.add_instruction(migraphx::make_op("dot"), q2, k2);
    auto cvt_dot = m2.add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), dot2);
    auto rmax    = m2.add_instruction(migraphx::make_op("reduce_max", {{"axes", {3}}}), cvt_dot);
    auto rmax_bc = m2.add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", f32_dot_shape.lens()}}), rmax);
    auto sub     = m2.add_instruction(migraphx::make_op("sub"), cvt_dot, rmax_bc);
    auto exp     = m2.add_instruction(migraphx::make_op("exp"), sub);
    auto rsum    = m2.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {3}}}), exp);
    auto rsum_bc = m2.add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", f32_dot_shape.lens()}}), rsum);
    auto div     = m2.add_instruction(migraphx::make_op("div"), exp, rsum_bc);
    auto cvt_out = m2.add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), div);
    auto dot_v2 = m2.add_instruction(migraphx::make_op("dot"), cvt_out, v2);
    m2.add_return({dot_v2});

    run_pass(m1);
    EXPECT(m1 == m2);
}

// Verify no dot found: only softmax internals upcasted (develop behavior)
TEST_CASE(softmax_no_dot_found)
{
    migraphx::shape s{migraphx::shape::half_type, {1, 12, 1, 128}};
    migraphx::shape f32_s{migraphx::shape::float_type, s.lens()};

    migraphx::module m1;
    auto x1       = m1.add_parameter("x", s);
    auto softmax1 = m1.add_instruction(migraphx::make_op("softmax", {{"axis", 3}}), x1);
    m1.add_return({softmax1});

    migraphx::module m2;
    auto x2    = m2.add_parameter("x", s);
    auto cvt_x = m2.add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), x2);
    auto rmax = m2.add_instruction(migraphx::make_op("reduce_max", {{"axes", {3}}}), cvt_x);
    auto rmax_bc =
        m2.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", f32_s.lens()}}), rmax);
    auto sub  = m2.add_instruction(migraphx::make_op("sub"), cvt_x, rmax_bc);
    auto exp  = m2.add_instruction(migraphx::make_op("exp"), sub);
    auto rsum = m2.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {3}}}), exp);
    auto rsum_bc =
        m2.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", f32_s.lens()}}), rsum);
    auto div     = m2.add_instruction(migraphx::make_op("div"), exp, rsum_bc);
    auto cvt_out = m2.add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), div);
    m2.add_return({cvt_out});

    run_pass(m1);
    EXPECT(m1 == m2);
}

// Verify backward walk enters mul but finds no upstream dot.
// The walk follows mul's inputs (both are parameters/broadcasts, not dot),
// returns nullopt, and the pass only upcasts softmax internals (develop behavior).
TEST_CASE(softmax_mul_no_upstream_dot)
{
    migraphx::shape s{migraphx::shape::half_type, {1, 12, 1, 128}};
    migraphx::shape scale_shape{migraphx::shape::half_type, {1}};
    migraphx::shape f32_s{migraphx::shape::float_type, s.lens()};

    migraphx::module m1;
    auto x1     = m1.add_parameter("x", s);
    auto scale1 = m1.add_parameter("scale", scale_shape);
    auto scale_bc1 =
        m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), scale1);
    auto mul1     = m1.add_instruction(migraphx::make_op("mul"), x1, scale_bc1);
    auto softmax1 = m1.add_instruction(migraphx::make_op("softmax", {{"axis", 3}}), mul1);
    m1.add_return({softmax1});

    // Expected: mul stays f16, only softmax internals upcasted
    migraphx::module m2;
    auto x2     = m2.add_parameter("x", s);
    auto scale2 = m2.add_parameter("scale", scale_shape);
    auto scale_bc2 =
        m2.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), scale2);
    auto mul2    = m2.add_instruction(migraphx::make_op("mul"), x2, scale_bc2);
    auto cvt_mul = m2.add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), mul2);
    auto rmax = m2.add_instruction(migraphx::make_op("reduce_max", {{"axes", {3}}}), cvt_mul);
    auto rmax_bc =
        m2.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", f32_s.lens()}}), rmax);
    auto sub  = m2.add_instruction(migraphx::make_op("sub"), cvt_mul, rmax_bc);
    auto exp  = m2.add_instruction(migraphx::make_op("exp"), sub);
    auto rsum = m2.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {3}}}), exp);
    auto rsum_bc =
        m2.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", f32_s.lens()}}), rsum);
    auto div     = m2.add_instruction(migraphx::make_op("div"), exp, rsum_bc);
    auto cvt_out = m2.add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), div);
    m2.add_return({cvt_out});

    run_pass(m1);
    EXPECT(m1 == m2);
}

// Verify double mask: dot -> mul -> where -> where -> softmax
TEST_CASE(softmax_dot_scale_double_where)
{
    migraphx::shape dot_shape{migraphx::shape::half_type, {1, 12, 1, 128}};
    migraphx::shape k_shape{migraphx::shape::half_type, {1, 12, 128, 128}};
    migraphx::shape v_shape{migraphx::shape::half_type, {1, 12, 128, 128}};
    migraphx::shape scale_shape{migraphx::shape::half_type, {1}};
    migraphx::shape mask_shape{migraphx::shape::bool_type, {1, 12, 1, 128}};
    migraphx::shape f32_dot_shape{migraphx::shape::float_type, dot_shape.lens()};

    auto make_inputs =
        [](auto& mod, auto dot_shape, auto k_shape, auto scale_shape, auto mask_shape) {
            auto q     = mod.add_parameter("q", dot_shape);
            auto k     = mod.add_parameter("k", k_shape);
            auto scale = mod.add_parameter("scale", scale_shape);
            auto mask1 = mod.add_parameter("mask1", mask_shape);
            auto mask2 = mod.add_parameter("mask2", mask_shape);
            auto ninf =
                mod.add_literal(migraphx::literal{migraphx::shape{migraphx::shape::half_type, {1}},
                                                  {-std::numeric_limits<float>::infinity()}});
            auto ninf_bc = mod.add_instruction(
                migraphx::make_op("multibroadcast", {{"out_lens", dot_shape.lens()}}), ninf);
            auto scale_bc = mod.add_instruction(
                migraphx::make_op("multibroadcast", {{"out_lens", dot_shape.lens()}}), scale);
            auto dot = mod.add_instruction(migraphx::make_op("dot"), q, k);
            return std::make_tuple(dot, scale_bc, mask1, mask2, ninf_bc);
        };

    migraphx::module m1;
    {
        auto [dot, scale_bc, mask1, mask2, ninf_bc] =
            make_inputs(m1, dot_shape, k_shape, scale_shape, mask_shape);
        auto v       = m1.add_parameter("v", v_shape);
        auto mul     = m1.add_instruction(migraphx::make_op("mul"), dot, scale_bc);
        auto where1  = m1.add_instruction(migraphx::make_op("where"), mask1, ninf_bc, mul);
        auto where2  = m1.add_instruction(migraphx::make_op("where"), mask2, ninf_bc, where1);
        auto softmax = m1.add_instruction(migraphx::make_op("softmax", {{"axis", 3}}), where2);
        auto dot_v   = m1.add_instruction(migraphx::make_op("dot"), softmax, v);
        m1.add_return({dot_v});
    }

    migraphx::module m2;
    {
        auto [dot, scale_bc, mask1, mask2, ninf_bc] =
            make_inputs(m2, dot_shape, k_shape, scale_shape, mask_shape);
        auto v       = m2.add_parameter("v", v_shape);
        auto cvt_dot = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), dot);
        auto cvt_scale = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), scale_bc);
        auto mul      = m2.add_instruction(migraphx::make_op("mul"), cvt_dot, cvt_scale);
        auto cvt_ninf = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), ninf_bc);
        auto where1  = m2.add_instruction(migraphx::make_op("where"), mask1, cvt_ninf, mul);
        auto where2  = m2.add_instruction(migraphx::make_op("where"), mask2, cvt_ninf, where1);
        auto rmax    = m2.add_instruction(migraphx::make_op("reduce_max", {{"axes", {3}}}), where2);
        auto rmax_bc = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", f32_dot_shape.lens()}}), rmax);
        auto sub     = m2.add_instruction(migraphx::make_op("sub"), where2, rmax_bc);
        auto exp     = m2.add_instruction(migraphx::make_op("exp"), sub);
        auto rsum    = m2.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {3}}}), exp);
        auto rsum_bc = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", f32_dot_shape.lens()}}), rsum);
        auto div     = m2.add_instruction(migraphx::make_op("div"), exp, rsum_bc);
        auto cvt_out = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), div);
        auto dot_v = m2.add_instruction(migraphx::make_op("dot"), cvt_out, v);
        m2.add_return({dot_v});
    }

    run_pass(m1);
    EXPECT(m1 == m2);
}

// Verify single-input ops between dot and softmax are traversed.
// dot -> relu -> softmax: the walk follows relu (1 input) to find the dot.
TEST_CASE(softmax_dot_relu_upcast)
{
    migraphx::shape dot_shape{migraphx::shape::half_type, {1, 12, 1, 128}};
    migraphx::shape k_shape{migraphx::shape::half_type, {1, 12, 128, 128}};
    migraphx::shape v_shape{migraphx::shape::half_type, {1, 12, 128, 128}};
    migraphx::shape f32_dot_shape{migraphx::shape::float_type, dot_shape.lens()};

    migraphx::module m1;
    auto q1      = m1.add_parameter("q", dot_shape);
    auto k1      = m1.add_parameter("k", k_shape);
    auto v1      = m1.add_parameter("v", v_shape);
    auto dot1    = m1.add_instruction(migraphx::make_op("dot"), q1, k1);
    auto relu1   = m1.add_instruction(migraphx::make_op("relu"), dot1);
    auto softmax = m1.add_instruction(migraphx::make_op("softmax", {{"axis", 3}}), relu1);
    auto dot_v1  = m1.add_instruction(migraphx::make_op("dot"), softmax, v1);
    m1.add_return({dot_v1});

    // Expected: dot stays f16, convert(f16->f32) after dot, relu upcasted to f32
    migraphx::module m2;
    auto q2      = m2.add_parameter("q", dot_shape);
    auto k2      = m2.add_parameter("k", k_shape);
    auto v2      = m2.add_parameter("v", v_shape);
    auto dot2    = m2.add_instruction(migraphx::make_op("dot"), q2, k2);
    auto cvt_dot = m2.add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), dot2);
    auto relu2   = m2.add_instruction(migraphx::make_op("relu"), cvt_dot);
    auto rmax    = m2.add_instruction(migraphx::make_op("reduce_max", {{"axes", {3}}}), relu2);
    auto rmax_bc = m2.add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", f32_dot_shape.lens()}}), rmax);
    auto sub     = m2.add_instruction(migraphx::make_op("sub"), relu2, rmax_bc);
    auto exp     = m2.add_instruction(migraphx::make_op("exp"), sub);
    auto rsum    = m2.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {3}}}), exp);
    auto rsum_bc = m2.add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", f32_dot_shape.lens()}}), rsum);
    auto div     = m2.add_instruction(migraphx::make_op("div"), exp, rsum_bc);
    auto cvt_out = m2.add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), div);
    auto dot_v2 = m2.add_instruction(migraphx::make_op("dot"), cvt_out, v2);
    m2.add_return({dot_v2});

    run_pass(m1);
    EXPECT(m1 == m2);
}

// Verify mul with scale on left: mul(scale_literal, dot) instead of mul(dot, scale)
// Scale is a literal (evaluable) so can_eval() correctly identifies it as
// the non-data path, regardless of input ordering.
TEST_CASE(softmax_dot_scale_left)
{
    migraphx::shape dot_shape{migraphx::shape::half_type, {1, 12, 1, 128}};
    migraphx::shape k_shape{migraphx::shape::half_type, {1, 12, 128, 128}};
    migraphx::shape v_shape{migraphx::shape::half_type, {1, 12, 128, 128}};
    migraphx::shape scale_shape{migraphx::shape::half_type, {1}};
    migraphx::shape f32_dot_shape{migraphx::shape::float_type, dot_shape.lens()};

    auto make_graph = [](auto& mod, auto dot_shape, auto k_shape, auto scale_shape) {
        auto q        = mod.add_parameter("q", dot_shape);
        auto k        = mod.add_parameter("k", k_shape);
        auto scale    = mod.add_literal(migraphx::literal{scale_shape, {1.0f / std::sqrt(128.0f)}});
        auto scale_bc = mod.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", dot_shape.lens()}}), scale);
        auto dot = mod.add_instruction(migraphx::make_op("dot"), q, k);
        return std::make_tuple(dot, scale_bc);
    };

    migraphx::module m1;
    {
        auto [dot, scale_bc] = make_graph(m1, dot_shape, k_shape, scale_shape);
        auto v               = m1.add_parameter("v", v_shape);
        auto mul             = m1.add_instruction(migraphx::make_op("mul"), scale_bc, dot);
        auto softmax         = m1.add_instruction(migraphx::make_op("softmax", {{"axis", 3}}), mul);
        auto dot_v           = m1.add_instruction(migraphx::make_op("dot"), softmax, v);
        m1.add_return({dot_v});
    }

    migraphx::module m2;
    {
        auto [dot, scale_bc] = make_graph(m2, dot_shape, k_shape, scale_shape);
        auto v               = m2.add_parameter("v", v_shape);
        auto cvt_scale       = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), scale_bc);
        auto cvt_dot = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), dot);
        auto mul     = m2.add_instruction(migraphx::make_op("mul"), cvt_scale, cvt_dot);
        auto rmax    = m2.add_instruction(migraphx::make_op("reduce_max", {{"axes", {3}}}), mul);
        auto rmax_bc = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", f32_dot_shape.lens()}}), rmax);
        auto sub     = m2.add_instruction(migraphx::make_op("sub"), mul, rmax_bc);
        auto exp     = m2.add_instruction(migraphx::make_op("exp"), sub);
        auto rsum    = m2.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {3}}}), exp);
        auto rsum_bc = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", f32_dot_shape.lens()}}), rsum);
        auto div     = m2.add_instruction(migraphx::make_op("div"), exp, rsum_bc);
        auto cvt_out = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), div);
        auto dot_v = m2.add_instruction(migraphx::make_op("dot"), cvt_out, v);
        m2.add_return({dot_v});
    }

    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(softmax_lse_upcast)
{
    migraphx::shape s{migraphx::shape::half_type, {10, 1000}};
    migraphx::module m;
    auto x    = m.add_parameter("x", s);
    auto rmax = m.add_instruction(migraphx::make_op("reduce_max", {{"axes", {1}}}), x);
    auto rmax_mb =
        m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), rmax);
    auto sub  = m.add_instruction(migraphx::make_op("sub"), x, rmax_mb);
    auto exp  = m.add_instruction(migraphx::make_op("exp"), sub);
    auto rsum = m.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), exp);
    auto rsum_mb =
        m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), rsum);
    auto div = m.add_instruction(migraphx::make_op("div"), exp, rsum_mb);
    auto log = m.add_instruction(migraphx::make_op("log"), rsum);
    auto add = m.add_instruction(migraphx::make_op("add"), log, rmax);
    m.add_return({div, add});

    run_pass(m);

    auto reduces = find_all(migraphx::iterator_for(m),
                            [&](auto ins) { return migraphx::contains(ins->name(), "reduce"); });
    EXPECT(all_of(reduces,
                  [](auto ins) { return ins->get_shape().type() == migraphx::shape::float_type; }));
    EXPECT(all_of(m.get_returns(), [&](auto ins) { return ins->get_shape().type() == s.type(); }));
}

TEST_CASE(reduce_mean)
{
    migraphx::shape s{migraphx::shape::float_type, {1, 3, 9}};
    migraphx::module m;
    auto x           = m.add_parameter("x", s);
    auto reduce_mean = m.add_instruction(migraphx::make_op("reduce_mean", {{"axes", {-1}}}), x);
    m.add_return({reduce_mean});
    run_pass(m);
    EXPECT(
        none_of(migraphx::iterator_for(m), [](auto ins) { return ins->name() == "reduce_mean"; }));

    auto reduces = find_all(migraphx::iterator_for(m), [&](auto ins) {
        return migraphx::contains(ins->name(), "reduce_sum");
    });
    EXPECT(all_of(reduces, [](auto ins) {
        auto axes = ins->get_operator().to_value()["axes"].template to_vector<int64_t>();
        return axes.size() == 1 and axes[0] == -1;
    }));
}

TEST_CASE(reduce_mean_large)
{
    migraphx::shape s{migraphx::shape::half_type, {1, 3, 65536}};
    migraphx::module m;
    auto x           = m.add_parameter("x", s);
    auto reduce_mean = m.add_instruction(migraphx::make_op("reduce_mean", {{"axes", {-1}}}), x);
    m.add_return({reduce_mean});
    run_pass(m);
    EXPECT(
        none_of(migraphx::iterator_for(m), [](auto ins) { return ins->name() == "reduce_mean"; }));

    auto reduces = find_all(migraphx::iterator_for(m), [&](auto ins) {
        return migraphx::contains(ins->name(), "reduce_sum");
    });
    EXPECT(all_of(reduces, [](migraphx::instruction_ref ins) {
        auto axes = ins->get_operator().to_value()["axes"].template to_vector<int64_t>();
        return axes.size() == 1 and axes[0] == -1 and
               ins->get_shape().type() == migraphx::shape::float_type;
    }));
}

TEST_CASE(reduce_mean_accuracy)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape s{migraphx::shape::float_type, {1, 3, 9}};
    auto x           = mm->add_parameter("x", s);
    auto reduce_mean = mm->add_instruction(migraphx::make_op("reduce_mean", {{"axes", {-1}}}), x);
    mm->add_return({reduce_mean});
    run_pass(*mm);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> data(s.elements());
    std::iota(data.begin(), data.end(), 0);
    migraphx::parameter_map params;
    params["x"] = migraphx::argument(s, data.data());
    auto result = p.eval(params).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{4.f, 13.f, 22.f};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(reduce_mean_accuracy2)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape s{migraphx::shape::float_type, {1, 3, 3, 3}};
    auto x           = mm->add_parameter("x", s);
    auto reduce_mean = mm->add_instruction(migraphx::make_op("reduce_mean", {{"axes", {2, 3}}}), x);
    mm->add_return({reduce_mean});
    run_pass(*mm);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> data(s.elements());
    std::iota(data.begin(), data.end(), 0);
    migraphx::parameter_map params;
    params["x"] = migraphx::argument(s, data.data());
    auto result = p.eval(params).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{4.f, 13.f, 22.f};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(reduce_mean_accuracy3)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape s{migraphx::shape::float_type, {1, 3, 3, 3}};
    auto x           = mm->add_parameter("x", s);
    auto reduce_mean = mm->add_instruction(migraphx::make_op("reduce_mean", {{"axes", {2}}}), x);
    mm->add_return({reduce_mean});
    run_pass(*mm);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> data(s.elements());
    std::iota(data.begin(), data.end(), 0);
    migraphx::parameter_map params;
    params["x"] = migraphx::argument(s, data.data());
    auto result = p.eval(params).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold{3.f, 4.f, 5.f, 12.f, 13.f, 14.f, 21.f, 22.f, 23.f};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(reduce_mean_accuracy4)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape s{migraphx::shape::int32_type, {1, 3, 2, 2}};
    auto x           = mm->add_parameter("x", s);
    auto reduce_mean = mm->add_instruction(migraphx::make_op("reduce_mean", {{"axes", {2}}}), x);
    mm->add_return({reduce_mean});
    run_pass(*mm);
    p.compile(migraphx::make_target("ref"));

    std::vector<int32_t> data(s.elements());
    std::iota(data.begin(), data.end(), 0);
    migraphx::parameter_map params;
    params["x"] = migraphx::argument(s, data.data());
    auto result = p.eval(params).back();
    std::vector<int32_t> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<int32_t> gold{1, 2, 5, 6, 9, 10};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(reduce_mean_accuracy5)
{
    using migraphx::half;
    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape s{migraphx::shape::half_type, {1, 3, 2, 2}};
    auto x           = mm->add_parameter("x", s);
    auto reduce_mean = mm->add_instruction(migraphx::make_op("reduce_mean", {{"axes", {2}}}), x);
    mm->add_return({reduce_mean});
    run_pass(*mm);
    p.compile(migraphx::make_target("ref"));

    std::vector<half> data(s.elements());
    std::iota(data.begin(), data.end(), 0);
    migraphx::parameter_map params;
    params["x"] = migraphx::argument(s, data.data());
    auto result = p.eval(params).back();
    std::vector<half> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<half> gold{half{1}, half{2}, half{5}, half{6}, half{9}, half{10}};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

static migraphx::instruction_ref
add_reduce_mean(migraphx::module& m, std::vector<std::size_t> axes, migraphx::instruction_ref input)
{
    auto reduce_size = migraphx::transform_accumulate(
        axes.begin(), axes.end(), std::size_t{1}, std::multiplies<>{}, [&](auto axis) {
            return input->get_shape().lens()[axis];
        });
    auto t   = input->get_shape().type();
    auto rl  = m.add_literal(migraphx::literal{{t, {1}}, {reduce_size}});
    auto div = migraphx::add_common_op(m, migraphx::make_op("div"), {input, rl});
    return m.add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), div);
}

TEST_CASE(reduce_mean_variance_sqdiff)
{
    migraphx::shape s{migraphx::shape::float_type, {1, 3, 9}};
    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", s);
        auto mean = m1.add_instruction(migraphx::make_op("reduce_mean", {{"axes", {2}}}), x);
        auto meanb =
            m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), mean);
        auto sqdiff = m1.add_instruction(migraphx::make_op("sqdiff"), x, meanb);
        auto variance =
            m1.add_instruction(migraphx::make_op("reduce_mean", {{"axes", {2}}}), sqdiff);
        m1.add_return({mean, variance});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x        = m2.add_parameter("x", s);
        auto mean     = add_reduce_mean(m2, {2}, x);
        auto x2       = m2.add_instruction(migraphx::make_op("mul"), x, x);
        auto mean_x2  = add_reduce_mean(m2, {2}, x2);
        auto mean2    = m2.add_instruction(migraphx::make_op("mul"), mean, mean);
        auto variance = m2.add_instruction(migraphx::make_op("sub"), mean_x2, mean2);
        m2.add_return({mean, variance});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(reduce_mean_variance_mul_x_minus)
{
    migraphx::shape s{migraphx::shape::float_type, {1, 3, 9}};
    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", s);
        auto mean = m1.add_instruction(migraphx::make_op("reduce_mean", {{"axes", {2}}}), x);
        auto meanb =
            m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), mean);
        auto sub      = m1.add_instruction(migraphx::make_op("sub"), x, meanb);
        auto mul      = m1.add_instruction(migraphx::make_op("mul"), sub, sub);
        auto variance = m1.add_instruction(migraphx::make_op("reduce_mean", {{"axes", {2}}}), mul);
        m1.add_return({mean, variance});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x        = m2.add_parameter("x", s);
        auto mean     = add_reduce_mean(m2, {2}, x);
        auto x2       = m2.add_instruction(migraphx::make_op("mul"), x, x);
        auto mean_x2  = add_reduce_mean(m2, {2}, x2);
        auto mean2    = m2.add_instruction(migraphx::make_op("mul"), mean, mean);
        auto variance = m2.add_instruction(migraphx::make_op("sub"), mean_x2, mean2);
        m2.add_return({mean, variance});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(reduce_mean_variance_pow_x_minus)
{
    migraphx::shape s{migraphx::shape::float_type, {1, 3, 9}};
    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", s);
        auto mean = m1.add_instruction(migraphx::make_op("reduce_mean", {{"axes", {2}}}), x);
        auto meanb =
            m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), mean);
        auto sub      = m1.add_instruction(migraphx::make_op("sub"), x, meanb);
        auto two      = m1.add_literal(migraphx::literal{migraphx::shape::float_type, {2}});
        auto pow      = migraphx::add_common_op(m1, migraphx::make_op("pow"), {sub, two});
        auto variance = m1.add_instruction(migraphx::make_op("reduce_mean", {{"axes", {2}}}), pow);
        m1.add_return({mean, variance});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x        = m2.add_parameter("x", s);
        auto mean     = add_reduce_mean(m2, {2}, x);
        auto x2       = m2.add_instruction(migraphx::make_op("mul"), x, x);
        auto mean_x2  = add_reduce_mean(m2, {2}, x2);
        auto mean2    = m2.add_instruction(migraphx::make_op("mul"), mean, mean);
        auto variance = m2.add_instruction(migraphx::make_op("sub"), mean_x2, mean2);
        m2.add_return({mean, variance});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(reduce_mean_variance_diff_inputs)
{
    migraphx::shape s{migraphx::shape::float_type, {1, 3, 9}};
    migraphx::module m1;
    {
        auto x      = m1.add_parameter("x", s);
        auto y      = m1.add_parameter("y", s);
        auto mean1  = m1.add_instruction(migraphx::make_op("reduce_mean", {{"axes", {2}}}), x);
        auto mean2b = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), mean1);
        auto sqdiff = m1.add_instruction(migraphx::make_op("sqdiff"), y, mean2b);
        auto mean2  = m1.add_instruction(migraphx::make_op("reduce_mean", {{"axes", {2}}}), sqdiff);
        m1.add_return({mean1, mean2});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x      = m2.add_parameter("x", s);
        auto y      = m2.add_parameter("y", s);
        auto mean1  = add_reduce_mean(m2, {2}, x);
        auto mean2b = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), mean1);
        auto sqdiff = m2.add_instruction(migraphx::make_op("sqdiff"), y, mean2b);
        auto mean2  = add_reduce_mean(m2, {2}, sqdiff);
        m2.add_return({mean1, mean2});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(reduce_mean_variance_sqdiff_diff_axes)
{
    migraphx::shape s{migraphx::shape::float_type, {1, 3, 9}};
    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", s);
        auto mean = m1.add_instruction(migraphx::make_op("reduce_mean", {{"axes", {2}}}), x);
        auto meanb =
            m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), mean);
        auto sqdiff = m1.add_instruction(migraphx::make_op("sqdiff"), x, meanb);
        auto variance =
            m1.add_instruction(migraphx::make_op("reduce_mean", {{"axes", {0, 2}}}), sqdiff);
        m1.add_return({mean, variance});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x    = m2.add_parameter("x", s);
        auto mean = add_reduce_mean(m2, {2}, x);
        auto meanb =
            m2.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), mean);
        auto sqdiff   = m2.add_instruction(migraphx::make_op("sqdiff"), x, meanb);
        auto variance = add_reduce_mean(m2, {0, 2}, sqdiff);
        m2.add_return({mean, variance});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(logsoftmax)
{
    migraphx::shape s{migraphx::shape::float_type, {1, 3, 9}};
    migraphx::module m1;
    {
        auto x          = m1.add_parameter("x", s);
        auto logsoftmax = m1.add_instruction(migraphx::make_op("logsoftmax", {{"axis", 2}}), x);
        m1.add_return({logsoftmax});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x   = m2.add_parameter("x", s);
        auto max = m2.add_instruction(migraphx::make_op("reduce_max", {{"axes", {2}}}), x);
        auto maxb =
            m2.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), max);
        auto sub = m2.add_instruction(migraphx::make_op("sub"), x, maxb);
        auto exp = m2.add_instruction(migraphx::make_op("exp"), sub);
        auto sum = m2.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), exp);
        auto sumb =
            m2.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), sum);
        auto div = m2.add_instruction(migraphx::make_op("div"), exp, sumb);
        auto log = m2.add_instruction(migraphx::make_op("log"), div);
        m2.add_return({log});
    }
    EXPECT(m1.sort() == m2.sort());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
