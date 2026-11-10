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
 *
 */
#include <migraphx/rewrite_gather_dot.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/module.hpp>
#include <migraphx/make_op.hpp>

#include <test.hpp>

static void run_pass(migraphx::module& m, const migraphx::rewrite_gather_dot& pass = {})
{
    migraphx::run_passes(m, {pass, migraphx::dead_code_elimination{}});
}

// Each of the 4 tokens is multiplied with the matrices of 2 of the 4 entries
TEST_CASE(gather_dot)
{
    migraphx::module m1;
    {
        auto x   = m1.add_parameter("x", {migraphx::shape::float_type, {4, 3}});
        auto w   = m1.add_parameter("w", {migraphx::shape::float_type, {4, 3, 5}});
        auto idx = m1.add_parameter("idx", {migraphx::shape::int32_type, {8}});
        auto xu  = m1.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), x);
        auto xb =
            m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {4, 2, 3}}}), xu);
        auto xr  = m1.add_instruction(migraphx::make_op("reshape", {{"dims", {8, 1, 3}}}), xb);
        auto wg  = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), w, idx);
        auto dot = m1.add_instruction(migraphx::make_op("dot"), xr, wg);
        m1.add_return({dot});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x      = m2.add_parameter("x", {migraphx::shape::float_type, {4, 3}});
        auto w      = m2.add_parameter("w", {migraphx::shape::float_type, {4, 3, 5}});
        auto idx    = m2.add_parameter("idx", {migraphx::shape::int32_type, {8}});
        auto tokens = m2.add_literal(
            migraphx::literal{{migraphx::shape::int32_type, {8, 1}}, {0, 0, 1, 1, 2, 2, 3, 3}});
        auto xu = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), x);
        auto wd = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), w);
        auto xm = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 4, 3}}}), xu);
        auto xe =
            m2.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {4, 4, 3}}}), xm);
        auto wm      = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {1}}}), wd);
        auto dot     = m2.add_instruction(migraphx::make_op("dot"), xe, wm);
        auto du      = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), dot);
        auto entries = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), idx);
        auto nd   = m2.add_instruction(migraphx::make_op("concat", {{"axis", 1}}), entries, tokens);
        auto rows = m2.add_instruction(migraphx::make_op("gathernd"), du, nd);
        m2.add_return({rows});
    }
    EXPECT(m1.sort() == m2.sort());
}

// Two dots with a gathered bias and pointwise operators in between, as in the
// experts of a mixture of experts layer
TEST_CASE(gather_dot_pointwise_dot)
{
    migraphx::module m1;
    {
        auto x     = m1.add_parameter("x", {migraphx::shape::float_type, {4, 3}});
        auto w1    = m1.add_parameter("w1", {migraphx::shape::float_type, {4, 5, 3}});
        auto b1    = m1.add_parameter("b1", {migraphx::shape::float_type, {4, 5}});
        auto w2    = m1.add_parameter("w2", {migraphx::shape::float_type, {4, 5, 2}});
        auto idx   = m1.add_parameter("idx", {migraphx::shape::int32_type, {8}});
        auto scale = m1.add_literal(migraphx::literal{{migraphx::shape::float_type, {1}}, {2}});
        auto xu    = m1.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), x);
        auto xb =
            m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {4, 2, 3}}}), xu);
        auto xr  = m1.add_instruction(migraphx::make_op("reshape", {{"dims", {8, 1, 3}}}), xb);
        auto w1g = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), w1, idx);
        auto w1t =
            m1.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), w1g);
        auto d1  = m1.add_instruction(migraphx::make_op("dot"), xr, w1t);
        auto b1g = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), b1, idx);
        auto b1u = m1.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), b1g);
        auto a1  = m1.add_instruction(migraphx::make_op("add"), d1, b1u);
        auto r   = m1.add_instruction(migraphx::make_op("relu"), a1);
        auto sb = m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {8, 1, 5}}}),
                                     scale);
        auto s  = m1.add_instruction(migraphx::make_op("mul"), r, sb);
        auto w2g = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), w2, idx);
        auto d2  = m1.add_instruction(migraphx::make_op("dot"), s, w2g);
        m1.add_return({d2});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x      = m2.add_parameter("x", {migraphx::shape::float_type, {4, 3}});
        auto w1     = m2.add_parameter("w1", {migraphx::shape::float_type, {4, 5, 3}});
        auto b1     = m2.add_parameter("b1", {migraphx::shape::float_type, {4, 5}});
        auto w2     = m2.add_parameter("w2", {migraphx::shape::float_type, {4, 5, 2}});
        auto idx    = m2.add_parameter("idx", {migraphx::shape::int32_type, {8}});
        auto scale  = m2.add_literal(migraphx::literal{{migraphx::shape::float_type, {1}}, {2}});
        auto tokens = m2.add_literal(
            migraphx::literal{{migraphx::shape::int32_type, {8, 1}}, {0, 0, 1, 1, 2, 2, 3, 3}});
        auto xu  = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), x);
        auto w1d = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), w1);
        auto w1t = m2.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), w1d);
        auto xm = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 4, 3}}}), xu);
        auto xe =
            m2.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {4, 4, 3}}}), xm);
        auto w1m = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {1}}}), w1t);
        auto d1  = m2.add_instruction(migraphx::make_op("dot"), xe, w1m);
        auto d1u = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), d1);
        auto b1d = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), b1);
        auto b1u = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), b1d);
        auto b1b = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {4, 4, 1, 5}}}), b1u);
        auto a1 = m2.add_instruction(migraphx::make_op("add"), d1u, b1b);
        auto r  = m2.add_instruction(migraphx::make_op("relu"), a1);
        auto sb = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {4, 4, 1, 5}}}), scale);
        auto s       = m2.add_instruction(migraphx::make_op("mul"), r, sb);
        auto w2d     = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), w2);
        auto sm      = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {2}}}), s);
        auto w2m     = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {1}}}), w2d);
        auto d2      = m2.add_instruction(migraphx::make_op("dot"), sm, w2m);
        auto d2u     = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), d2);
        auto entries = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), idx);
        auto nd   = m2.add_instruction(migraphx::make_op("concat", {{"axis", 1}}), entries, tokens);
        auto rows = m2.add_instruction(migraphx::make_op("gathernd"), d2u, nd);
        m2.add_return({rows});
    }
    EXPECT(m1.sort() == m2.sort());
}

// Views of the dot result that keep the rows apart are computed for every entry
TEST_CASE(gather_dot_views)
{
    migraphx::module m1;
    {
        auto x   = m1.add_parameter("x", {migraphx::shape::float_type, {4, 3}});
        auto w   = m1.add_parameter("w", {migraphx::shape::float_type, {4, 3, 6}});
        auto idx = m1.add_parameter("idx", {migraphx::shape::int32_type, {8}});
        auto xu  = m1.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), x);
        auto xb =
            m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {4, 2, 3}}}), xu);
        auto xr  = m1.add_instruction(migraphx::make_op("reshape", {{"dims", {8, 1, 3}}}), xb);
        auto wg  = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), w, idx);
        auto dot = m1.add_instruction(migraphx::make_op("dot"), xr, wg);
        auto pairs =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {8, 1, 3, 2}}}), dot);
        auto first = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {3}}, {"starts", {0}}, {"ends", {1}}}), pairs);
        auto sq = m1.add_instruction(migraphx::make_op("squeeze", {{"axes", {3}}}), first);
        m1.add_return({sq});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x      = m2.add_parameter("x", {migraphx::shape::float_type, {4, 3}});
        auto w      = m2.add_parameter("w", {migraphx::shape::float_type, {4, 3, 6}});
        auto idx    = m2.add_parameter("idx", {migraphx::shape::int32_type, {8}});
        auto tokens = m2.add_literal(
            migraphx::literal{{migraphx::shape::int32_type, {8, 1}}, {0, 0, 1, 1, 2, 2, 3, 3}});
        auto xu = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), x);
        auto wd = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), w);
        auto xm = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 4, 3}}}), xu);
        auto xe =
            m2.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {4, 4, 3}}}), xm);
        auto wm  = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {1}}}), wd);
        auto dot = m2.add_instruction(migraphx::make_op("dot"), xe, wm);
        auto du  = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), dot);
        auto pairs =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {4, 4, 1, 3, 2}}}), du);
        auto first = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {4}}, {"starts", {0}}, {"ends", {1}}}), pairs);
        auto sq      = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {4}}}), first);
        auto entries = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), idx);
        auto nd   = m2.add_instruction(migraphx::make_op("concat", {{"axis", 1}}), entries, tokens);
        auto rows = m2.add_instruction(migraphx::make_op("gathernd"), sq, nd);
        m2.add_return({rows});
    }
    EXPECT(m1.sort() == m2.sort());
}

// Packed weights are unpacked and dequantized for every entry
TEST_CASE(gather_dot_unpack_int4)
{
    migraphx::module m1;
    {
        auto x      = m1.add_parameter("x", {migraphx::shape::half_type, {4, 4}});
        auto w      = m1.add_parameter("w", {migraphx::shape::uint8_type, {4, 5, 2}});
        auto scales = m1.add_parameter("scales", {migraphx::shape::float_type, {4, 5}});
        auto idx    = m1.add_parameter("idx", {migraphx::shape::int32_type, {8}});
        auto zero   = m1.add_literal(migraphx::literal{{migraphx::shape::uint8_type, {1}}, {8}});
        auto xu     = m1.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), x);
        auto xb =
            m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {4, 2, 4}}}), xu);
        auto xr = m1.add_instruction(migraphx::make_op("reshape", {{"dims", {8, 1, 4}}}), xb);
        auto wg = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), w, idx);
        auto wu = m1.add_instruction(migraphx::make_op("unpack_int4"), wg);
        auto sg = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), scales, idx);
        auto su = m1.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), sg);
        auto sb =
            m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {8, 5, 4}}}), su);
        auto zb = m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {8, 5, 4}}}),
                                     zero);
        auto dq = m1.add_instruction(migraphx::make_op("dequantizelinear"), wu, sb, zb);
        auto wh = m1.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), dq);
        auto wt =
            m1.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), wh);
        auto dot = m1.add_instruction(migraphx::make_op("dot"), xr, wt);
        m1.add_return({dot});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x      = m2.add_parameter("x", {migraphx::shape::half_type, {4, 4}});
        auto w      = m2.add_parameter("w", {migraphx::shape::uint8_type, {4, 5, 2}});
        auto scales = m2.add_parameter("scales", {migraphx::shape::float_type, {4, 5}});
        auto idx    = m2.add_parameter("idx", {migraphx::shape::int32_type, {8}});
        auto zero   = m2.add_literal(migraphx::literal{{migraphx::shape::uint8_type, {1}}, {8}});
        auto tokens = m2.add_literal(
            migraphx::literal{{migraphx::shape::int32_type, {8, 1}}, {0, 0, 1, 1, 2, 2, 3, 3}});
        auto xu = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), x);
        auto wd = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), w);
        auto wu = m2.add_instruction(migraphx::make_op("unpack_int4", {{"axis", 3}}), wd);
        auto sd = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), scales);
        auto su = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {3}}}), sd);
        auto sb = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {4, 1, 5, 4}}}), su);
        auto zb = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {4, 1, 5, 4}}}), zero);
        auto dq = m2.add_instruction(migraphx::make_op("dequantizelinear"), wu, sb, zb);
        auto wh = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), dq);
        auto wt =
            m2.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), wh);
        auto xm = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 4, 4}}}), xu);
        auto xe =
            m2.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {4, 4, 4}}}), xm);
        auto wm      = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {1}}}), wt);
        auto dot     = m2.add_instruction(migraphx::make_op("dot"), xe, wm);
        auto du      = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), dot);
        auto entries = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), idx);
        auto nd   = m2.add_instruction(migraphx::make_op("concat", {{"axis", 1}}), entries, tokens);
        auto rows = m2.add_instruction(migraphx::make_op("gathernd"), du, nd);
        m2.add_return({rows});
    }
    EXPECT(m1.sort() == m2.sort());
}

// Both inputs of the dot are gathered, so each row depends only on its entry
// and the result is gathered with the indices alone
TEST_CASE(gather_dot_gathered_rows)
{
    migraphx::module m1;
    {
        auto a   = m1.add_parameter("a", {migraphx::shape::float_type, {4, 3}});
        auto w   = m1.add_parameter("w", {migraphx::shape::float_type, {4, 3, 5}});
        auto idx = m1.add_parameter("idx", {migraphx::shape::int32_type, {8}});
        auto ag  = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), a, idx);
        auto au  = m1.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), ag);
        auto wg  = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), w, idx);
        auto dot = m1.add_instruction(migraphx::make_op("dot"), au, wg);
        m1.add_return({dot});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto a     = m2.add_parameter("a", {migraphx::shape::float_type, {4, 3}});
        auto w     = m2.add_parameter("w", {migraphx::shape::float_type, {4, 3, 5}});
        auto idx   = m2.add_parameter("idx", {migraphx::shape::int32_type, {8}});
        auto ad    = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), a);
        auto au    = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), ad);
        auto wd    = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), w);
        auto am    = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {2}}}), au);
        auto wm    = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {1}}}), wd);
        auto dot   = m2.add_instruction(migraphx::make_op("dot"), am, wm);
        auto du    = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), dot);
        auto table = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {1}}}), du);
        auto rows  = m2.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), table, idx);
        m2.add_return({rows});
    }
    EXPECT(m1.sort() == m2.sort());
}

// With fewer rows than entries the gather reads less than the whole table
TEST_CASE(gather_dot_fewer_rows)
{
    migraphx::module m1;
    {
        auto x   = m1.add_parameter("x", {migraphx::shape::float_type, {3, 3}});
        auto w   = m1.add_parameter("w", {migraphx::shape::float_type, {8, 3, 5}});
        auto idx = m1.add_parameter("idx", {migraphx::shape::int32_type, {6}});
        auto xu  = m1.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), x);
        auto xb =
            m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {3, 2, 3}}}), xu);
        auto xr  = m1.add_instruction(migraphx::make_op("reshape", {{"dims", {6, 1, 3}}}), xb);
        auto wg  = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), w, idx);
        auto dot = m1.add_instruction(migraphx::make_op("dot"), xr, wg);
        m1.add_return({dot});
    }
    migraphx::module m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

// A lower crossover uses the batched dot with fewer rows than entries: 6 rows
// from 8 entries
TEST_CASE(gather_dot_lower_crossover)
{
    migraphx::module m1;
    {
        auto x   = m1.add_parameter("x", {migraphx::shape::float_type, {3, 3}});
        auto w   = m1.add_parameter("w", {migraphx::shape::float_type, {8, 3, 5}});
        auto idx = m1.add_parameter("idx", {migraphx::shape::int32_type, {6}});
        auto xu  = m1.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), x);
        auto xb =
            m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {3, 2, 3}}}), xu);
        auto xr  = m1.add_instruction(migraphx::make_op("reshape", {{"dims", {6, 1, 3}}}), xb);
        auto wg  = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), w, idx);
        auto dot = m1.add_instruction(migraphx::make_op("dot"), xr, wg);
        m1.add_return({dot});
    }
    run_pass(m1, {.min_rows_per_entry = 0.75f});

    migraphx::module m2;
    {
        auto x      = m2.add_parameter("x", {migraphx::shape::float_type, {3, 3}});
        auto w      = m2.add_parameter("w", {migraphx::shape::float_type, {8, 3, 5}});
        auto idx    = m2.add_parameter("idx", {migraphx::shape::int32_type, {6}});
        auto tokens = m2.add_literal(
            migraphx::literal{{migraphx::shape::int32_type, {6, 1}}, {0, 0, 1, 1, 2, 2}});
        auto xu = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), x);
        auto wd = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), w);
        auto xm = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 3, 3}}}), xu);
        auto xe =
            m2.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {8, 3, 3}}}), xm);
        auto wm      = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {1}}}), wd);
        auto dot     = m2.add_instruction(migraphx::make_op("dot"), xe, wm);
        auto du      = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), dot);
        auto entries = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), idx);
        auto nd   = m2.add_instruction(migraphx::make_op("concat", {{"axis", 1}}), entries, tokens);
        auto rows = m2.add_instruction(migraphx::make_op("gathernd"), du, nd);
        m2.add_return({rows});
    }
    EXPECT(m1.sort() == m2.sort());
}

// A higher crossover keeps the gathered dots: 8 rows from 4 entries is 2 rows
// per entry
TEST_CASE(gather_dot_higher_crossover)
{
    migraphx::module m1;
    {
        auto x   = m1.add_parameter("x", {migraphx::shape::float_type, {4, 3}});
        auto w   = m1.add_parameter("w", {migraphx::shape::float_type, {4, 3, 5}});
        auto idx = m1.add_parameter("idx", {migraphx::shape::int32_type, {8}});
        auto xu  = m1.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), x);
        auto xb =
            m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {4, 2, 3}}}), xu);
        auto xr  = m1.add_instruction(migraphx::make_op("reshape", {{"dims", {8, 1, 3}}}), xb);
        auto wg  = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), w, idx);
        auto dot = m1.add_instruction(migraphx::make_op("dot"), xr, wg);
        m1.add_return({dot});
    }
    migraphx::module m2 = m1;
    run_pass(m1, {.min_rows_per_entry = 2.5f});
    EXPECT(m1 == m2);
}

// The rows of the first input are not a broadcast over the rows of a token
TEST_CASE(gather_dot_distinct_rows)
{
    migraphx::module m1;
    {
        auto x   = m1.add_parameter("x", {migraphx::shape::float_type, {8, 1, 3}});
        auto w   = m1.add_parameter("w", {migraphx::shape::float_type, {4, 3, 5}});
        auto idx = m1.add_parameter("idx", {migraphx::shape::int32_type, {8}});
        auto wg  = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), w, idx);
        auto dot = m1.add_instruction(migraphx::make_op("dot"), x, wg);
        m1.add_return({dot});
    }
    migraphx::module m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

// A gather along another axis does not select whole matrices
TEST_CASE(gather_dot_other_axis)
{
    migraphx::module m1;
    {
        auto x   = m1.add_parameter("x", {migraphx::shape::float_type, {8, 1, 3}});
        auto w   = m1.add_parameter("w", {migraphx::shape::float_type, {8, 4, 5}});
        auto idx = m1.add_parameter("idx", {migraphx::shape::int32_type, {3}});
        auto wg  = m1.add_instruction(migraphx::make_op("gather", {{"axis", 1}}), w, idx);
        auto dot = m1.add_instruction(migraphx::make_op("dot"), x, wg);
        m1.add_return({dot});
    }
    migraphx::module m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

// The gathered rows are only read by pointwise operators, so there is no dot
// to compute for every entry
TEST_CASE(gather_pointwise)
{
    migraphx::module m1;
    {
        auto b   = m1.add_parameter("b", {migraphx::shape::float_type, {4, 5}});
        auto y   = m1.add_parameter("y", {migraphx::shape::float_type, {8, 5}});
        auto idx = m1.add_parameter("idx", {migraphx::shape::int32_type, {8}});
        auto bg  = m1.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), b, idx);
        auto r   = m1.add_instruction(migraphx::make_op("relu"), bg);
        auto add = m1.add_instruction(migraphx::make_op("add"), r, y);
        m1.add_return({add});
    }
    migraphx::module m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
