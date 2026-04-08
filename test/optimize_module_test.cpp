/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <migraphx/literal.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/module.hpp>
#include <migraphx/optimize_module.hpp>
#include <migraphx/propagate_constant.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/serialize.hpp>
#include <test.hpp>

static void run_pass(migraphx::module& m)
{
    migraphx::run_passes(m, {migraphx::optimize_module{}});
}

TEST_CASE(broadcast_transpose_inner_broadcast)
{
    // first optimizes broadcast+transpose to just a broadcast,
    // then finds inner broadcast to become mul+broadcast
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::float_type, {1}, {0}});
        auto y = m1.add_parameter("y", {migraphx::shape::float_type, {1}, {0}});
        auto mb1 =
            m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 2, 3}}}), x);
        auto mb2 =
            m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 3, 2}}}), y);
        auto t1 =
            m1.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), mb1);
        auto mul = m1.add_instruction(migraphx::make_op("mul"), mb2, t1);
        m1.add_return({mul});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x   = m2.add_parameter("x", {migraphx::shape::float_type, {1}, {0}});
        auto y   = m2.add_parameter("y", {migraphx::shape::float_type, {1}, {0}});
        auto mul = m2.add_instruction(migraphx::make_op("mul"), y, x);
        auto mb =
            m2.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 3, 2}}}), mul);
        m2.add_return({mb});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(broadcast_transpose_inner_broadcast_generic)
{
    // first optimizes broadcast+transpose to unsqueeze+transpose+broadcast,
    // then finds inner broadcast to become mul+broadcast
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::float_type, {5, 10}});
        auto y = m1.add_parameter("y", {migraphx::shape::float_type, {5}});
        auto mb1 =
            m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {3, 5, 10}}}), x);
        auto mb2 =
            m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {3, 10, 5}}}), y);
        auto t1 =
            m1.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), mb2);
        auto mul = m1.add_instruction(migraphx::make_op("mul"), mb1, t1);
        m1.add_return({mul});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x         = m2.add_parameter("x", {migraphx::shape::float_type, {5, 10}});
        auto y         = m2.add_parameter("y", {migraphx::shape::float_type, {5}});
        auto yb        = m2.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 0}, {"out_lens", {5, 10}}}), y);
        auto mul = m2.add_instruction(migraphx::make_op("mul"), x, yb);
        auto mb2 = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {3, 5, 10}}}), mul);
        m2.add_return({mb2});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(mul_add_transpose_dot)
{
    auto lit1 = migraphx::generate_literal({migraphx::shape::float_type, {64}}, 0);
    auto lit2 = migraphx::generate_literal({migraphx::shape::float_type, {64}}, 1);
    auto lit3 = migraphx::generate_literal({migraphx::shape::float_type, {64, 64}}, 2);
    migraphx::module m1;
    {
        auto in1      = m1.add_parameter("x", {migraphx::shape::float_type, {2, 64, 4, 4}});
        auto lit1_ins = m1.add_literal(lit1);
        auto lit1_unsq =
            m1.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0, 2, 3}}}), lit1_ins);
        auto lit1_mb = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 64, 4, 4}}}), lit1_unsq);
        auto mul = m1.add_instruction(migraphx::make_op("mul"), lit1_mb, in1);

        auto lit2_ins = m1.add_literal(lit2);
        auto lit2_unsq =
            m1.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0, 2, 3}}}), lit2_ins);
        auto lit2_tp = m1.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), lit2_unsq);
        auto lit2_mb = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 4, 4, 64}}}), lit2_tp);

        auto mul_tp = m1.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), mul);
        auto add = m1.add_instruction(migraphx::make_op("add"), mul_tp, lit2_mb);

        auto lit3_ins = m1.add_literal(lit3);
        auto lit3_mb  = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 4, 64, 64}}}), lit3_ins);
        auto dot = m1.add_instruction(migraphx::make_op("dot"), add, lit3_mb);

        m1.add_return({dot});
    }
    run_pass(m1);

    // Compute const propagated literals
    migraphx::literal lit13;
    migraphx::literal lit23;
    migraphx::module lit_mod;
    {
        auto lit1_ins  = lit_mod.add_literal(lit1);
        auto lit1_b    = lit_mod.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 0}, {"out_lens", {64, 64}}}), lit1_ins);

        auto lit3_ins = lit_mod.add_literal(lit3);

        auto mul_lit   = lit_mod.add_instruction(migraphx::make_op("mul"), lit1_b, lit3_ins);
        auto lit13_arg = mul_lit->eval();
        lit13          = migraphx::literal(lit13_arg.get_shape(), lit13_arg.data());

        auto lit2_ins = lit_mod.add_literal(lit2);
        auto lit2_unsq =
            lit_mod.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), lit2_ins);
        auto lit2_mb = lit_mod.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {4, 64}}}), lit2_unsq);

        auto dot_lit   = lit_mod.add_instruction(migraphx::make_op("dot"), lit2_mb, lit3_ins);
        auto lit23_arg = dot_lit->eval();
        lit23          = migraphx::literal(lit23_arg.get_shape(), lit23_arg.data());
    }

    migraphx::module m2;
    {
        auto in1   = m2.add_parameter("x", {migraphx::shape::float_type, {2, 64, 4, 4}});
        auto in_tp = m2.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), in1);

        auto lit13_ins = m2.add_literal(lit13);
        auto lit13_b   = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 4, 64, 64}}}), lit13_ins);
        auto dot = m2.add_instruction(migraphx::make_op("dot"), in_tp, lit13_b);

        auto lit23_ins = m2.add_literal(lit23);
        auto lit23_mb  = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 4, 4, 64}}}), lit23_ins);

        auto add = m2.add_instruction(migraphx::make_op("add"), dot, lit23_mb);
        m2.add_return({add});
    }

    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(slice_squeeze_pw_unary)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 4}};
    migraphx::module m1;
    {
        auto input = m1.add_parameter("input", s);
        auto s0    = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}), input);
        auto sq0  = m1.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), s0);
        auto rel0 = m1.add_instruction(migraphx::make_op("relu"), sq0);
        auto s1   = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {2}}}), input);
        auto sq1  = m1.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), s1);
        auto rel1 = m1.add_instruction(migraphx::make_op("relu"), sq1);
        m1.add_return({rel0, rel1});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto input = m2.add_parameter("input", s);
        auto relu  = m2.add_instruction(migraphx::make_op("relu"), input);
        auto s0    = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}), relu);
        auto sq0 = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), s0);
        auto s1  = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {2}}}), relu);
        auto sq1 = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), s1);
        m2.add_return({sq0, sq1});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(slice_squeeze_pw_unary_3d)
{
    migraphx::shape s{migraphx::shape::float_type, {3, 2, 4}};
    migraphx::module m1;
    {
        auto input = m1.add_parameter("input", s);
        auto s0    = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}), input);
        auto sq0  = m1.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), s0);
        auto rel0 = m1.add_instruction(migraphx::make_op("relu"), sq0);
        auto s1   = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {2}}}), input);
        auto sq1  = m1.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), s1);
        auto rel1 = m1.add_instruction(migraphx::make_op("relu"), sq1);
        auto s2   = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {2}}, {"ends", {3}}}), input);
        auto sq2  = m1.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), s2);
        auto rel2 = m1.add_instruction(migraphx::make_op("relu"), sq2);
        m1.add_return({rel0, rel1, rel2});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto input = m2.add_parameter("input", s);
        auto relu  = m2.add_instruction(migraphx::make_op("relu"), input);
        auto s0    = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}), relu);
        auto sq0 = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), s0);
        auto s1  = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {2}}}), relu);
        auto sq1 = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), s1);
        auto s2  = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {2}}, {"ends", {3}}}), relu);
        auto sq2 = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), s2);
        m2.add_return({sq0, sq1, sq2});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(slice_squeeze_pw_binary_const)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 4}};
    migraphx::shape bs{migraphx::shape::float_type, {4}};
    migraphx::module m1;
    {
        auto input = m1.add_parameter("input", s);
        auto b0    = m1.add_literal(migraphx::generate_literal(bs, 0));
        auto b1    = m1.add_literal(migraphx::generate_literal(bs, 1));

        auto s0 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}), input);
        auto sq0  = m1.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), s0);
        auto add0 = m1.add_instruction(migraphx::make_op("add"), sq0, b0);

        auto s1 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {2}}}), input);
        auto sq1  = m1.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), s1);
        auto add1 = m1.add_instruction(migraphx::make_op("add"), sq1, b1);

        m1.add_return({add0, add1});
    }
    run_pass(m1);

    // propagate_constant folds unsqueeze+concat of literals into one literal
    migraphx::literal stacked_lit;
    {
        migraphx::module tmp;
        auto b0  = tmp.add_literal(migraphx::generate_literal(bs, 0));
        auto b1  = tmp.add_literal(migraphx::generate_literal(bs, 1));
        auto bu0 = tmp.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), b0);
        auto bu1 = tmp.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), b1);
        auto cat = tmp.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), bu0, bu1);
        auto ev  = cat->eval();
        stacked_lit = migraphx::literal(ev.get_shape(), ev.data());
    }

    migraphx::module m2;
    {
        auto input   = m2.add_parameter("input", s);
        auto stacked = m2.add_literal(stacked_lit);
        auto add     = m2.add_instruction(migraphx::make_op("add"), input, stacked);

        auto s0 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}), add);
        auto sq0 = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), s0);
        auto s1  = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {2}}}), add);
        auto sq1 = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), s1);

        m2.add_return({sq0, sq1});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(slice_squeeze_pw_silu_chain)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3, 4}};
    migraphx::shape bs{migraphx::shape::float_type, {3, 4}};
    migraphx::module m1;
    {
        auto input = m1.add_parameter("input", s);
        auto b0    = m1.add_literal(migraphx::generate_literal(bs, 0));
        auto b1    = m1.add_literal(migraphx::generate_literal(bs, 1));

        auto s0 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}), input);
        auto sq0  = m1.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), s0);
        auto add0 = m1.add_instruction(migraphx::make_op("add"), sq0, b0);
        auto sig0 = m1.add_instruction(migraphx::make_op("sigmoid"), add0);
        auto mul0 = m1.add_instruction(migraphx::make_op("mul"), sig0, add0);

        auto s1 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {2}}}), input);
        auto sq1  = m1.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), s1);
        auto add1 = m1.add_instruction(migraphx::make_op("add"), sq1, b1);
        auto sig1 = m1.add_instruction(migraphx::make_op("sigmoid"), add1);
        auto mul1 = m1.add_instruction(migraphx::make_op("mul"), sig1, add1);

        m1.add_return({mul0, mul1});
    }
    run_pass(m1);

    // propagate_constant folds the stacked bias literal; the SiLU DAG
    // (sigmoid feeds back into mul) prevents find_slice_pw_subgraph from
    // fully merging, so each slice keeps its own sigmoid→mul.
    migraphx::literal stacked_lit;
    {
        migraphx::module tmp;
        auto b0  = tmp.add_literal(migraphx::generate_literal(bs, 0));
        auto b1  = tmp.add_literal(migraphx::generate_literal(bs, 1));
        auto bu0 = tmp.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), b0);
        auto bu1 = tmp.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), b1);
        auto cat = tmp.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), bu0, bu1);
        auto ev  = cat->eval();
        stacked_lit = migraphx::literal(ev.get_shape(), ev.data());
    }

    migraphx::module m2;
    {
        auto input   = m2.add_parameter("input", s);
        auto stacked = m2.add_literal(stacked_lit);
        auto add     = m2.add_instruction(migraphx::make_op("add"), input, stacked);

        auto s0 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}), add);
        auto sig0 = m2.add_instruction(migraphx::make_op("sigmoid"), s0);
        auto mul0 = m2.add_instruction(migraphx::make_op("mul"), sig0, s0);
        auto sq0  = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), mul0);

        auto s1 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {2}}}), add);
        auto sig1 = m2.add_instruction(migraphx::make_op("sigmoid"), s1);
        auto mul1 = m2.add_instruction(migraphx::make_op("mul"), sig1, s1);
        auto sq1  = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), mul1);

        m2.add_return({sq0, sq1});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(slice_squeeze_non_zero_axis)
{
    migraphx::shape s{migraphx::shape::float_type, {3, 2, 4}};
    migraphx::module m1;
    {
        auto input = m1.add_parameter("input", s);
        auto s0    = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {1}}}), input);
        auto sq0  = m1.add_instruction(migraphx::make_op("squeeze", {{"axes", {1}}}), s0);
        auto rel0 = m1.add_instruction(migraphx::make_op("relu"), sq0);
        auto s1   = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {1}}, {"ends", {2}}}), input);
        auto sq1  = m1.add_instruction(migraphx::make_op("squeeze", {{"axes", {1}}}), s1);
        auto rel1 = m1.add_instruction(migraphx::make_op("relu"), sq1);
        m1.add_return({rel0, rel1});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto input = m2.add_parameter("input", s);
        auto relu  = m2.add_instruction(migraphx::make_op("relu"), input);
        auto s0    = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {1}}}), relu);
        auto sq0 = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {1}}}), s0);
        auto s1  = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {1}}, {"ends", {2}}}), relu);
        auto sq1 = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {1}}}), s1);
        m2.add_return({sq0, sq1});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(hoist_silu_above_slices_with_unsqueeze_concat)
{
    migraphx::shape s{migraphx::shape::float_type, {1, 64, 384}};
    migraphx::module m1;
    {
        auto input = m1.add_parameter("input", s);
        auto s0    = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {0}}, {"ends", {128}}}), input);
        auto sig0 = m1.add_instruction(migraphx::make_op("sigmoid"), s0);
        auto mul0 = m1.add_instruction(migraphx::make_op("mul"), s0, sig0);
        auto u0   = m1.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), mul0);

        auto s1 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {128}}, {"ends", {256}}}),
            input);
        auto sig1 = m1.add_instruction(migraphx::make_op("sigmoid"), s1);
        auto mul1 = m1.add_instruction(migraphx::make_op("mul"), s1, sig1);
        auto u1   = m1.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), mul1);

        auto s2 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {256}}, {"ends", {384}}}),
            input);
        auto sig2 = m1.add_instruction(migraphx::make_op("sigmoid"), s2);
        auto mul2 = m1.add_instruction(migraphx::make_op("mul"), s2, sig2);
        auto u2   = m1.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), mul2);

        auto cat = m1.add_instruction(migraphx::make_op("concat", {{"axis", 0}}), u0, u1, u2);
        m1.add_return({cat});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto input = m2.add_parameter("input", s);
        auto sig   = m2.add_instruction(migraphx::make_op("sigmoid"), input);
        auto mul   = m2.add_instruction(migraphx::make_op("mul"), input, sig);
        auto rs    = m2.add_instruction(
            migraphx::make_op("reshape", {{"dims", {3, 1, 64, 128}}}), mul);
        m2.add_return({rs});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(hoist_relu_above_slices_different_axis)
{
    migraphx::shape s{migraphx::shape::float_type, {4, 64, 32}};
    migraphx::module m1;
    {
        auto input = m1.add_parameter("input", s);
        auto s0    = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}), input);
        auto r0 = m1.add_instruction(migraphx::make_op("relu"), s0);
        auto s1 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {2}}}), input);
        auto r1 = m1.add_instruction(migraphx::make_op("relu"), s1);
        auto s2 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {2}}, {"ends", {3}}}), input);
        auto r2 = m1.add_instruction(migraphx::make_op("relu"), s2);
        auto s3 = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {3}}, {"ends", {4}}}), input);
        auto r3 = m1.add_instruction(migraphx::make_op("relu"), s3);
        m1.add_return({r0, r1, r2, r3});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto input = m2.add_parameter("input", s);
        auto relu  = m2.add_instruction(migraphx::make_op("relu"), input);
        auto r0    = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}), relu);
        auto r1 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {2}}}), relu);
        auto r2 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {2}}, {"ends", {3}}}), relu);
        auto r3 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {3}}, {"ends", {4}}}), relu);
        m2.add_return({r0, r1, r2, r3});
    }
    EXPECT(m1.sort() == m2.sort());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
