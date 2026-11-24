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
#include <migraphx/rewrite_convolution.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/module.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/verify.hpp>
#include <algorithm>
#include <test.hpp>

static void run_pass(migraphx::module& m)
{
    migraphx::run_passes(m, {migraphx::rewrite_convolution{}, migraphx::dead_code_elimination{}});
}

static migraphx::shape sf(std::vector<std::size_t> lens)
{
    return migraphx::shape{migraphx::shape::float_type, std::move(lens)};
}

// stride 1: a backward-data convolution is one flipped stride-1 forward convolution.
TEST_CASE(stride1_single_forward_conv)
{
    migraphx::module m1;
    {
        auto inp = m1.add_parameter("inp", sf({1, 2, 5}));
        auto w   = m1.add_parameter("w", sf({2, 3, 3}));
        auto r   = m1.add_instruction(
            migraphx::make_op("convolution_backwards",
                                {{"padding", {0}}, {"stride", {1}}, {"dilation", {1}}, {"group", 1}}),
            inp,
            w);
        m1.add_return({r});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto inp = m2.add_parameter("inp", sf({1, 2, 5}));
        auto w   = m2.add_parameter("w", sf({2, 3, 3}));
        auto t =
            m2.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0, 2}}}), w);
        auto rv = m2.add_instruction(migraphx::make_op("reverse", {{"axes", {2}}}), t);
        auto c  = m2.add_instruction(
            migraphx::make_op(
                "convolution",
                {{"padding", {2, 2}}, {"stride", {1}}, {"dilation", {1}}, {"group", 1}}),
            inp,
            rv);
        m2.add_return({c});
    }
    EXPECT(m1 == m2);
}

// stride 2 (no dilation): two residues -> two stride-1 forward convolutions interleaved back
// together with a pixel-shuffle (concat + reshape), no pad/add kernels.
TEST_CASE(stride2_two_residue_interleave)
{
    migraphx::module m1;
    {
        auto inp = m1.add_parameter("inp", sf({1, 1, 3}));
        auto w   = m1.add_parameter("w", sf({1, 1, 2}));
        auto r   = m1.add_instruction(
            migraphx::make_op("convolution_backwards",
                                {{"padding", {0}}, {"stride", {2}}, {"dilation", {1}}, {"group", 1}}),
            inp,
            w);
        m1.add_return({r});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto inp = m2.add_parameter("inp", sf({1, 1, 3}));
        auto w   = m2.add_parameter("w", sf({1, 1, 2}));

        auto residue_conv = [&](migraphx::instruction_ref wslice) {
            auto st = m2.add_instruction(migraphx::make_op("step", {{"axes", {2}}, {"steps", {2}}}),
                                         wslice);
            auto t  = m2.add_instruction(
                migraphx::make_op("transpose", {{"permutation", {1, 0, 2}}}), st);
            auto rv = m2.add_instruction(migraphx::make_op("reverse", {{"axes", {2}}}), t);
            return m2.add_instruction(
                migraphx::make_op(
                    "convolution",
                    {{"padding", {0, 0}}, {"stride", {1}}, {"dilation", {1}}, {"group", 1}}),
                inp,
                rv);
        };

        // residue 0 (itilda=0): no leading slice (full axis)
        auto c0 = residue_conv(w);
        // residue 1 (itilda=1): tap-slice starting at 1
        auto s1 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {1}}, {"ends", {2}}}), w);
        auto c1  = residue_conv(s1);
        auto u0  = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {3}}}), c0);
        auto u1  = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {3}}}), c1);
        auto cat = m2.add_instruction(migraphx::make_op("concat", {{"axis", 3}}), u0, u1);
        auto rs  = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 1, 6}}}), cat);
        m2.add_return({rs});
    }
    EXPECT(m1 == m2);
}

// grouped backward-data convolution: the in/out channel swap keeps groups intact via reshape.
TEST_CASE(grouped_weight_reshape)
{
    migraphx::module m1;
    {
        auto inp = m1.add_parameter("inp", sf({1, 4, 5}));
        auto w   = m1.add_parameter("w", sf({4, 2, 3}));
        auto r   = m1.add_instruction(
            migraphx::make_op("convolution_backwards",
                                {{"padding", {0}}, {"stride", {1}}, {"dilation", {1}}, {"group", 2}}),
            inp,
            w);
        m1.add_return({r});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto inp   = m2.add_parameter("inp", sf({1, 4, 5}));
        auto w     = m2.add_parameter("w", sf({4, 2, 3}));
        auto split = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 2, 2, 3}}}), w);
        auto trans = m2.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), split);
        auto merge = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {4, 2, 3}}}), trans);
        auto rv    = m2.add_instruction(migraphx::make_op("reverse", {{"axes", {2}}}), merge);
        auto c     = m2.add_instruction(
            migraphx::make_op(
                "convolution",
                {{"padding", {2, 2}}, {"stride", {1}}, {"dilation", {1}}, {"group", 2}}),
            inp,
            rv);
        m2.add_return({c});
    }
    EXPECT(m1 == m2);
}

// dynamic shapes are left untouched (existing path handles them).
TEST_CASE(dynamic_shape_no_rewrite)
{
    auto make = [] {
        migraphx::module m;
        migraphx::shape inp_shape{migraphx::shape::float_type, {{1, 1}, {1, 1}, {3, 8}}};
        auto inp = m.add_parameter("inp", inp_shape);
        auto w   = m.add_parameter("w", sf({1, 1, 2}));
        auto r   = m.add_instruction(
            migraphx::make_op("convolution_backwards",
                                {{"padding", {0}}, {"stride", {2}}, {"dilation", {1}}, {"group", 1}}),
            inp,
            w);
        m.add_return({r});
        return m;
    };
    migraphx::module m1 = make();
    run_pass(m1);
    migraphx::module m2 = make();
    EXPECT(m1 == m2);
}

// stride 1 in 2-D: one flipped stride-1 forward convolution (channels swapped, taps reversed).
TEST_CASE(rewrite_2d_stride1)
{
    migraphx::module m1;
    {
        auto inp = m1.add_parameter("inp", sf({1, 2, 5, 5}));
        auto w   = m1.add_parameter("w", sf({2, 3, 3, 3}));
        auto r   = m1.add_instruction(
            migraphx::make_op(
                "convolution_backwards",
                {{"padding", {0, 0}}, {"stride", {1, 1}}, {"dilation", {1, 1}}, {"group", 1}}),
            inp,
            w);
        m1.add_return({r});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto inp = m2.add_parameter("inp", sf({1, 2, 5, 5}));
        auto w   = m2.add_parameter("w", sf({2, 3, 3, 3}));
        auto t =
            m2.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0, 2, 3}}}), w);
        auto rv = m2.add_instruction(migraphx::make_op("reverse", {{"axes", {2, 3}}}), t);
        auto c  = m2.add_instruction(migraphx::make_op("convolution",
                                                       {{"padding", {2, 2, 2, 2}},
                                                        {"stride", {1, 1}},
                                                        {"dilation", {1, 1}},
                                                        {"group", 1}}),
                                    inp,
                                    rv);
        m2.add_return({c});
    }
    EXPECT(m1 == m2);
}

// stride 1 in 3-D.
TEST_CASE(rewrite_3d_stride1)
{
    migraphx::module m1;
    {
        auto inp = m1.add_parameter("inp", sf({1, 2, 5, 5, 5}));
        auto w   = m1.add_parameter("w", sf({2, 3, 3, 3, 3}));
        auto r   = m1.add_instruction(migraphx::make_op("convolution_backwards",
                                                        {{"padding", {0, 0, 0}},
                                                         {"stride", {1, 1, 1}},
                                                         {"dilation", {1, 1, 1}},
                                                         {"group", 1}}),
                                    inp,
                                    w);
        m1.add_return({r});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto inp = m2.add_parameter("inp", sf({1, 2, 5, 5, 5}));
        auto w   = m2.add_parameter("w", sf({2, 3, 3, 3, 3}));
        auto t   = m2.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {1, 0, 2, 3, 4}}}), w);
        auto rv = m2.add_instruction(migraphx::make_op("reverse", {{"axes", {2, 3, 4}}}), t);
        auto c  = m2.add_instruction(migraphx::make_op("convolution",
                                                       {{"padding", {2, 2, 2, 2, 2, 2}},
                                                        {"stride", {1, 1, 1}},
                                                        {"dilation", {1, 1, 1}},
                                                        {"group", 1}}),
                                    inp,
                                    rv);
        m2.add_return({c});
    }
    EXPECT(m1 == m2);
}

// Build out = convolution_backwards(inp, w), compile on the reference target, and return the
// result.
static migraphx::argument eval_conv_backwards(const migraphx::operation& op,
                                              bool rewrite,
                                              migraphx::shape inp_shape,
                                              migraphx::shape w_shape)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto inp = mm->add_parameter("inp", inp_shape);
    auto w   = mm->add_parameter("w", w_shape);
    auto r   = mm->add_instruction(op, inp, w);
    mm->add_return({r});
    if(rewrite)
        run_pass(*mm);
    // The rewrite must actually fire, otherwise the comparison below is vacuous.
    bool has_bwd = std::any_of(
        mm->begin(), mm->end(), [](auto&& ins) { return ins.name() == "convolution_backwards"; });
    EXPECT(has_bwd != rewrite);
    p.compile(migraphx::make_target("ref"));
    migraphx::parameter_map params;
    params["inp"] = migraphx::generate_argument(inp_shape, 1);
    params["w"]   = migraphx::generate_argument(w_shape, 2);
    return p.eval(params).back();
}

// The rewritten subgraph must compute the same out as the reference convolution_backwards.
static void
verify_rewrite(const migraphx::operation& op, migraphx::shape inp_shape, migraphx::shape w_shape)
{
    auto ref = eval_conv_backwards(op, false, inp_shape, w_shape);
    auto got = eval_conv_backwards(op, true, inp_shape, w_shape);
    std::vector<float> ref_data;
    std::vector<float> got_data;
    ref.visit([&](auto v) { ref_data.assign(v.begin(), v.end()); });
    got.visit([&](auto v) { got_data.assign(v.begin(), v.end()); });
    EXPECT(migraphx::verify::allclose(got_data, migraphx::verify::expected{ref_data}));
}

// stride 1: single flipped forward convolution.
TEST_CASE(numeric_stride1)
{
    verify_rewrite(
        migraphx::make_op("convolution_backwards",
                          {{"padding", {1}}, {"stride", {1}}, {"dilation", {1}}, {"group", 1}}),
        sf({1, 2, 5}),
        sf({2, 3, 3}));
}

// stride 2, no dilation: pixel-shuffle interleave reassembly.
TEST_CASE(numeric_stride2_interleave)
{
    verify_rewrite(
        migraphx::make_op("convolution_backwards",
                          {{"padding", {1, 1}}, {"stride", {2, 2}}, {"dilation", {1, 1}}}),
        sf({1, 2, 4, 4}),
        sf({2, 3, 3, 3}));
}

// asymmetric stride, no dilation: interleave with different per-dim strides.
TEST_CASE(numeric_stride_asymmetric)
{
    verify_rewrite(
        migraphx::make_op("convolution_backwards",
                          {{"padding", {1, 1}}, {"stride", {2, 3}}, {"dilation", {1, 1}}}),
        sf({1, 2, 4, 4}),
        sf({2, 3, 3, 3}));
}

// stride 2, dilation 2: general (zero-stuff + pad + add) reassembly.
TEST_CASE(numeric_stride2_dilation2_general)
{
    verify_rewrite(
        migraphx::make_op("convolution_backwards",
                          {{"padding", {2, 2}}, {"stride", {2, 2}}, {"dilation", {2, 2}}}),
        sf({1, 2, 4, 4}),
        sf({2, 3, 3, 3}));
}

// grouped backward convolution.
TEST_CASE(numeric_grouped)
{
    verify_rewrite(
        migraphx::make_op(
            "convolution_backwards",
            {{"padding", {1, 1}}, {"stride", {2, 2}}, {"dilation", {1, 1}}, {"group", 2}}),
        sf({1, 2, 4, 4}),
        sf({2, 3, 3, 3}));
}

// 1-D backward convolution.
TEST_CASE(numeric_1d)
{
    verify_rewrite(
        migraphx::make_op("convolution_backwards",
                          {{"padding", {0}}, {"stride", {2}}, {"dilation", {1}}, {"group", 1}}),
        sf({1, 2, 5}),
        sf({2, 3, 3}));
}

// 2-D, stride 1: no-upsample path.
TEST_CASE(numeric_2d_stride1)
{
    verify_rewrite(
        migraphx::make_op(
            "convolution_backwards",
            {{"padding", {1, 1}}, {"stride", {1, 1}}, {"dilation", {1, 1}}, {"group", 1}}),
        sf({1, 2, 5, 5}),
        sf({2, 3, 3, 3}));
}

// 3-D, stride 1: no-upsample path.
TEST_CASE(numeric_3d_stride1)
{
    verify_rewrite(
        migraphx::make_op(
            "convolution_backwards",
            {{"padding", {1, 1, 1}}, {"stride", {1, 1, 1}}, {"dilation", {1, 1, 1}}, {"group", 1}}),
        sf({1, 2, 4, 4, 4}),
        sf({2, 3, 3, 3, 3}));
}

// 3-D, stride 2, no dilation: interleave reassembly.
TEST_CASE(numeric_3d_stride2_interleave)
{
    verify_rewrite(
        migraphx::make_op("convolution_backwards",
                          {{"padding", {1, 1, 1}}, {"stride", {2, 2, 2}}, {"dilation", {1, 1, 1}}}),
        sf({1, 2, 3, 3, 3}),
        sf({2, 3, 3, 3, 3}));
}

// 3-D, stride 2, dilation 2: general (zero-stuff + pad + add) reassembly.
TEST_CASE(numeric_3d_stride2_dilation2_general)
{
    verify_rewrite(
        migraphx::make_op("convolution_backwards",
                          {{"padding", {2, 2, 2}}, {"stride", {2, 2, 2}}, {"dilation", {2, 2, 2}}}),
        sf({1, 2, 3, 3, 3}),
        sf({2, 3, 3, 3, 3}));
}

static void run_nd_pass(migraphx::module& m)
{
    migraphx::run_passes(m,
                         {migraphx::rewrite_convolution{.enable_nd_convolution = true},
                          migraphx::dead_code_elimination{}});
}

// A pixel shuffle of a convolution is a convolution over (H, 2, W, 2) that has one input
// element along the dims of size 2, with the output channels of each shuffle position as
// their filter taps
TEST_CASE(convolution_pixel_shuffle)
{
    migraphx::shape xs{migraphx::shape::float_type, {1, 4, 3, 3}};
    migraphx::shape ws{migraphx::shape::float_type, {8, 4, 3, 3}};
    migraphx::shape bs{migraphx::shape::float_type, {8}};
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", xs);
        auto w = m1.add_literal(migraphx::generate_literal(ws, 1));
        auto b = m1.add_literal(migraphx::generate_literal(bs, 2));
        auto conv =
            m1.add_instruction(migraphx::make_op("convolution", {{"padding", {1, 1}}}), x, w);
        auto bias = m1.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {1, 8, 3, 3}}}), b);
        auto add = m1.add_instruction(migraphx::make_op("add"), conv, bias);
        auto rsp =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 2, 2, 2, 3, 3}}}), add);
        auto t = m1.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 1, 4, 2, 5, 3}}}), rsp);
        auto out = m1.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 2, 6, 6}}}), t);
        m1.add_return({out});
    }
    run_nd_pass(m1);

    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", xs);
        auto w = m2.add_literal(migraphx::generate_literal(ws, 1));
        auto b = m2.add_literal(migraphx::generate_literal(bs, 2));
        auto xr =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 4, 3, 1, 3, 1}}}), x);
        auto wr =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 2, 2, 4, 3, 3}}}), w);
        auto wt = m2.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 3, 4, 1, 5, 2}}}), wr);
        auto wv   = m2.add_instruction(migraphx::make_op("reverse", {{"axes", {3, 5}}}), wt);
        auto conv = m2.add_instruction(migraphx::make_op("convolution",
                                                         {{"padding", {1, 1, 1, 1, 1, 1, 1, 1}},
                                                          {"stride", {1, 1, 1, 1}},
                                                          {"dilation", {1, 1, 1, 1}}}),
                                       xr,
                                       wv);
        auto br =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 2, 2, 2, 1, 1}}}), b);
        auto bt = m2.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 1, 4, 2, 5, 3}}}), br);
        auto bias = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 2, 3, 2, 3, 2}}}), bt);
        auto add = m2.add_instruction(migraphx::make_op("add"), conv, bias);
        auto out = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 2, 6, 6}}}), add);
        m2.add_return({out});
    }
    EXPECT(m1.sort() == m2.sort());
}

// Unary ops between the bias and the reshape follow the new convolution
TEST_CASE(convolution_pixel_shuffle_unary)
{
    migraphx::shape xs{migraphx::shape::float_type, {1, 4, 3, 3}};
    migraphx::shape ws{migraphx::shape::float_type, {8, 4, 3, 3}};
    migraphx::shape bs{migraphx::shape::float_type, {8}};
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", xs);
        auto w = m1.add_literal(migraphx::generate_literal(ws, 1));
        auto b = m1.add_literal(migraphx::generate_literal(bs, 2));
        auto conv =
            m1.add_instruction(migraphx::make_op("convolution", {{"padding", {1, 1}}}), x, w);
        auto bias = m1.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {1, 8, 3, 3}}}), b);
        auto add  = m1.add_instruction(migraphx::make_op("add"), conv, bias);
        auto relu = m1.add_instruction(migraphx::make_op("relu"), add);
        auto neg  = m1.add_instruction(migraphx::make_op("neg"), relu);
        auto rsp =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 2, 2, 2, 3, 3}}}), neg);
        auto t = m1.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 1, 4, 2, 5, 3}}}), rsp);
        auto out = m1.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 2, 6, 6}}}), t);
        m1.add_return({out});
    }
    run_nd_pass(m1);

    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", xs);
        auto w = m2.add_literal(migraphx::generate_literal(ws, 1));
        auto b = m2.add_literal(migraphx::generate_literal(bs, 2));
        auto xr =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 4, 3, 1, 3, 1}}}), x);
        auto wr =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 2, 2, 4, 3, 3}}}), w);
        auto wt = m2.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 3, 4, 1, 5, 2}}}), wr);
        auto wv   = m2.add_instruction(migraphx::make_op("reverse", {{"axes", {3, 5}}}), wt);
        auto conv = m2.add_instruction(migraphx::make_op("convolution",
                                                         {{"padding", {1, 1, 1, 1, 1, 1, 1, 1}},
                                                          {"stride", {1, 1, 1, 1}},
                                                          {"dilation", {1, 1, 1, 1}}}),
                                       xr,
                                       wv);
        auto br =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 2, 2, 2, 1, 1}}}), b);
        auto bt = m2.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 1, 4, 2, 5, 3}}}), br);
        auto bias = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 2, 3, 2, 3, 2}}}), bt);
        auto add  = m2.add_instruction(migraphx::make_op("add"), conv, bias);
        auto relu = m2.add_instruction(migraphx::make_op("relu"), add);
        auto neg  = m2.add_instruction(migraphx::make_op("neg"), relu);
        auto out  = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 2, 6, 6}}}), neg);
        m2.add_return({out});
    }
    EXPECT(m1.sort() == m2.sort());
}

// A depth to space that splits the channels last, (2, 3, K), on a strided and dilated 1-D
// convolution without a bias, with an elementwise op before the reshape that merges the dims
TEST_CASE(convolution_pixel_shuffle_channels_last_split)
{
    migraphx::shape xs{migraphx::shape::float_type, {2, 4, 9}};
    migraphx::shape ws{migraphx::shape::float_type, {6, 4, 3}};
    migraphx::value attrs = {{"padding", {2, 2}}, {"stride", {2}}, {"dilation", {2}}};
    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", xs);
        auto w    = m1.add_literal(migraphx::generate_literal(ws, 1));
        auto conv = m1.add_instruction(migraphx::make_op("convolution", attrs), x, w);
        auto rsp = m1.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 3, 2, 5}}}), conv);
        auto t = m1.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}),
                                    rsp);
        auto relu = m1.add_instruction(migraphx::make_op("relu"), t);
        auto out  = m1.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 2, 15}}}), relu);
        m1.add_return({out});
    }
    run_nd_pass(m1);

    migraphx::module m2;
    {
        auto x  = m2.add_parameter("x", xs);
        auto w  = m2.add_literal(migraphx::generate_literal(ws, 1));
        auto xr = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 4, 9, 1}}}), x);
        auto wr = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {3, 2, 4, 3}}}), w);
        auto wt =
            m2.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 2, 3, 0}}}), wr);
        auto wv   = m2.add_instruction(migraphx::make_op("reverse", {{"axes", {3}}}), wt);
        auto conv = m2.add_instruction(
            migraphx::make_op(
                "convolution",
                {{"padding", {2, 2, 2, 2}}, {"stride", {2, 1}}, {"dilation", {2, 1}}}),
            xr,
            wv);
        auto relu = m2.add_instruction(migraphx::make_op("relu"), conv);
        auto out  = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 2, 15}}}), relu);
        m2.add_return({out});
    }
    EXPECT(m1.sort() == m2.sort());
}

// A split of the output channels that no reshape merges into the spatial dims, as before a
// group normalization, is not a pixel shuffle
TEST_CASE(convolution_pixel_shuffle_split_only)
{
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::float_type, {1, 4, 3, 3}});
        auto w = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {8, 4, 3, 3}}, 1));
        auto conv =
            m1.add_instruction(migraphx::make_op("convolution", {{"padding", {1, 1}}}), x, w);
        auto rsp =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 2, 4, 3, 3}}}), conv);
        auto t = m1.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 4, 2}}}), rsp);
        m1.add_return({t});
    }
    migraphx::module m2 = m1;
    run_nd_pass(m1);
    EXPECT(m1.sort() == m2.sort());
}

// The reordered filter of weights that are not constant would be computed at runtime
TEST_CASE(convolution_pixel_shuffle_weights_not_constant)
{
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::float_type, {1, 4, 3, 3}});
        auto w = m1.add_parameter("w", {migraphx::shape::float_type, {8, 4, 3, 3}});
        auto conv =
            m1.add_instruction(migraphx::make_op("convolution", {{"padding", {1, 1}}}), x, w);
        auto rsp =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 2, 2, 2, 3, 3}}}), conv);
        auto t = m1.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 1, 4, 2, 5, 3}}}), rsp);
        auto out = m1.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 2, 6, 6}}}), t);
        m1.add_return({out});
    }
    migraphx::module m2 = m1;
    run_nd_pass(m1);
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(convolution_pixel_shuffle_grouped)
{
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::float_type, {1, 4, 3, 3}});
        auto w = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {8, 2, 3, 3}}, 1));
        auto conv = m1.add_instruction(
            migraphx::make_op("convolution", {{"padding", {1, 1}}, {"group", 2}}), x, w);
        auto rsp =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 2, 2, 2, 3, 3}}}), conv);
        auto t = m1.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 1, 4, 2, 5, 3}}}), rsp);
        auto out = m1.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 2, 6, 6}}}), t);
        m1.add_return({out});
    }
    migraphx::module m2 = m1;
    run_nd_pass(m1);
    EXPECT(m1.sort() == m2.sort());
}

// An asymmetric padding becomes a pad of the input, as insert_pad would make it, so that the
// dims where the input has one element are not padded
TEST_CASE(convolution_pixel_shuffle_asymmetric_padding)
{
    migraphx::shape xs{migraphx::shape::float_type, {1, 4, 3, 3}};
    migraphx::shape ws{migraphx::shape::float_type, {8, 4, 3, 3}};
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", xs);
        auto w = m1.add_literal(migraphx::generate_literal(ws, 1));
        auto conv =
            m1.add_instruction(migraphx::make_op("convolution", {{"padding", {0, 1, 2, 1}}}), x, w);
        auto rsp =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 2, 2, 2, 3, 3}}}), conv);
        auto t = m1.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 1, 4, 2, 5, 3}}}), rsp);
        auto out = m1.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 2, 6, 6}}}), t);
        m1.add_return({out});
    }
    run_nd_pass(m1);

    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", xs);
        auto w = m2.add_literal(migraphx::generate_literal(ws, 1));
        auto xp =
            m2.add_instruction(migraphx::make_op("pad", {{"pads", {0, 0, 0, 1, 0, 0, 2, 1}}}), x);
        auto xr =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 4, 5, 1, 5, 1}}}), xp);
        auto wr =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 2, 2, 4, 3, 3}}}), w);
        auto wt = m2.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 3, 4, 1, 5, 2}}}), wr);
        auto wv   = m2.add_instruction(migraphx::make_op("reverse", {{"axes", {3, 5}}}), wt);
        auto conv = m2.add_instruction(migraphx::make_op("convolution",
                                                         {{"padding", {0, 1, 0, 1, 0, 1, 0, 1}},
                                                          {"stride", {1, 1, 1, 1}},
                                                          {"dilation", {1, 1, 1, 1}}}),
                                       xr,
                                       wv);
        auto out = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 2, 6, 6}}}), conv);
        m2.add_return({out});
    }
    EXPECT(m1.sort() == m2.sort());
}

// A convolution of one input channel is left as a channelwise convolution
TEST_CASE(convolution_pixel_shuffle_one_input_channel)
{
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::float_type, {1, 1, 3, 3}});
        auto w = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {8, 1, 3, 3}}, 1));
        auto conv =
            m1.add_instruction(migraphx::make_op("convolution", {{"padding", {1, 1}}}), x, w);
        auto rsp =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 2, 2, 2, 3, 3}}}), conv);
        auto t = m1.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 1, 4, 2, 5, 3}}}), rsp);
        auto out = m1.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 2, 6, 6}}}), t);
        m1.add_return({out});
    }
    migraphx::module m2 = m1;
    run_nd_pass(m1);
    EXPECT(m1.sort() == m2.sort());
}

// The rewrite is off by default
TEST_CASE(convolution_pixel_shuffle_disabled)
{
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::float_type, {1, 4, 3, 3}});
        auto w = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {8, 4, 3, 3}}, 1));
        auto conv =
            m1.add_instruction(migraphx::make_op("convolution", {{"padding", {1, 1}}}), x, w);
        auto rsp =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 2, 2, 2, 3, 3}}}), conv);
        auto t = m1.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 1, 4, 2, 5, 3}}}), rsp);
        auto out = m1.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 2, 6, 6}}}), t);
        m1.add_return({out});
    }
    migraphx::module m2 = m1;
    run_pass(m1);
    EXPECT(m1.sort() == m2.sort());
}

// A backward convolution of stride 2 is a convolution over (H, 2, W, 2) that has one input
// element along the dims of size 2. The flipped 4x4 filter splits into 2x2 taps for each of
// the 2x2 residues.
TEST_CASE(backwards_nd_convolution)
{
    migraphx::module m1;
    {
        auto inp = m1.add_parameter("inp", sf({1, 2, 3, 3}));
        auto w   = m1.add_literal(migraphx::generate_literal(sf({2, 3, 4, 4}), 2));
        auto r   = m1.add_instruction(
            migraphx::make_op("convolution_backwards",
                              {{"padding", {1, 1}}, {"stride", {2, 2}}, {"dilation", {1, 1}}}),
            inp,
            w);
        m1.add_return({r});
    }
    run_nd_pass(m1);

    migraphx::module m2;
    {
        auto inp = m2.add_parameter("inp", sf({1, 2, 3, 3}));
        auto w   = m2.add_literal(migraphx::generate_literal(sf({2, 3, 4, 4}), 2));
        auto wt =
            m2.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0, 2, 3}}}), w);
        auto wv = m2.add_instruction(migraphx::make_op("reverse", {{"axes", {2, 3}}}), wt);
        auto wr =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {3, 2, 2, 2, 2, 2}}}), wv);
        auto xr =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 2, 3, 1, 3, 1}}}), inp);
        auto conv = m2.add_instruction(migraphx::make_op("convolution",
                                                         {{"padding", {1, 1, 1, 1, 1, 1, 1, 1}},
                                                          {"stride", {1, 1, 1, 1}},
                                                          {"dilation", {1, 1, 1, 1}}}),
                                       xr,
                                       wr);
        auto merge =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 3, 8, 8}}}), conv);
        auto s0 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {1}}, {"ends", {7}}}), merge);
        auto s1 = m2.add_instruction(
            migraphx::make_op("slice", {{"axes", {3}}, {"starts", {1}}, {"ends", {7}}}), s0);
        m2.add_return({s1});
    }
    EXPECT(m1 == m2);
}

// The dim with stride 1 has one residue and gets no dim for it
TEST_CASE(backwards_nd_convolution_mixed_stride)
{
    migraphx::module m1;
    {
        auto inp = m1.add_parameter("inp", sf({1, 2, 3, 3}));
        auto w   = m1.add_literal(migraphx::generate_literal(sf({2, 3, 4, 3}), 2));
        auto r   = m1.add_instruction(
            migraphx::make_op("convolution_backwards",
                              {{"padding", {0, 0}}, {"stride", {2, 1}}, {"dilation", {1, 1}}}),
            inp,
            w);
        m1.add_return({r});
    }
    run_nd_pass(m1);

    migraphx::module m2;
    {
        auto inp = m2.add_parameter("inp", sf({1, 2, 3, 3}));
        auto w   = m2.add_literal(migraphx::generate_literal(sf({2, 3, 4, 3}), 2));
        auto wt =
            m2.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0, 2, 3}}}), w);
        auto wv = m2.add_instruction(migraphx::make_op("reverse", {{"axes", {2, 3}}}), wt);
        auto wr = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {3, 2, 2, 2, 3}}}), wv);
        auto xr =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 2, 3, 1, 3}}}), inp);
        auto conv = m2.add_instruction(
            migraphx::make_op(
                "convolution",
                {{"padding", {1, 1, 2, 1, 1, 2}}, {"stride", {1, 1, 1}}, {"dilation", {1, 1, 1}}}),
            xr,
            wr);
        auto merge =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 3, 8, 5}}}), conv);
        m2.add_return({merge});
    }
    EXPECT(m1 == m2);
}

// A filter of 3 taps at stride 2 gives the residues 2 taps and 1 tap, so they stay separate
// convolutions
TEST_CASE(backwards_nd_convolution_uneven_taps)
{
    migraphx::module m1;
    {
        auto inp = m1.add_parameter("inp", sf({1, 2, 3, 3}));
        auto w   = m1.add_literal(migraphx::generate_literal(sf({2, 3, 3, 3}), 2));
        auto r   = m1.add_instruction(
            migraphx::make_op("convolution_backwards",
                              {{"padding", {1, 1}}, {"stride", {2, 2}}, {"dilation", {1, 1}}}),
            inp,
            w);
        m1.add_return({r});
    }
    migraphx::module m2 = m1;
    run_pass(m2);
    run_nd_pass(m1);
    EXPECT(m1 == m2);
}

// A grouped backward convolution keeps the residue convolutions
TEST_CASE(backwards_nd_convolution_grouped)
{
    migraphx::module m1;
    {
        auto inp = m1.add_parameter("inp", sf({1, 4, 3, 3}));
        auto w   = m1.add_literal(migraphx::generate_literal(sf({4, 3, 4, 4}), 2));
        auto r   = m1.add_instruction(
            migraphx::make_op(
                "convolution_backwards",
                {{"padding", {1, 1}}, {"stride", {2, 2}}, {"dilation", {1, 1}}, {"group", 2}}),
            inp,
            w);
        m1.add_return({r});
    }
    migraphx::module m2 = m1;
    run_pass(m2);
    run_nd_pass(m1);
    EXPECT(m1 == m2);
}

// A backward convolution of one input channel keeps the residue convolutions, which are
// channelwise
TEST_CASE(backwards_nd_convolution_one_input_channel)
{
    migraphx::module m1;
    {
        auto inp = m1.add_parameter("inp", sf({1, 1, 3, 3}));
        auto w   = m1.add_literal(migraphx::generate_literal(sf({1, 3, 4, 4}), 2));
        auto r   = m1.add_instruction(
            migraphx::make_op("convolution_backwards",
                              {{"padding", {1, 1}}, {"stride", {2, 2}}, {"dilation", {1, 1}}}),
            inp,
            w);
        m1.add_return({r});
    }
    migraphx::module m2 = m1;
    run_pass(m2);
    run_nd_pass(m1);
    EXPECT(m1 == m2);
}

// With a dilation the residues do not interleave, so they stay separate convolutions
TEST_CASE(backwards_nd_convolution_dilated)
{
    migraphx::module m1;
    {
        auto inp = m1.add_parameter("inp", sf({1, 2, 3, 3}));
        auto w   = m1.add_literal(migraphx::generate_literal(sf({2, 3, 3, 3}), 2));
        auto r   = m1.add_instruction(
            migraphx::make_op("convolution_backwards",
                              {{"padding", {2, 2}}, {"stride", {2, 2}}, {"dilation", {2, 2}}}),
            inp,
            w);
        m1.add_return({r});
    }
    migraphx::module m2 = m1;
    run_pass(m2);
    run_nd_pass(m1);
    EXPECT(m1 == m2);
}

// The reordered filter of weights that are not constant would be computed at runtime
TEST_CASE(backwards_nd_convolution_weights_not_constant)
{
    migraphx::module m1;
    {
        auto inp = m1.add_parameter("inp", sf({1, 2, 3, 3}));
        auto w   = m1.add_parameter("w", sf({2, 3, 4, 4}));
        auto r   = m1.add_instruction(
            migraphx::make_op("convolution_backwards",
                              {{"padding", {1, 1}}, {"stride", {2, 2}}, {"dilation", {1, 1}}}),
            inp,
            w);
        m1.add_return({r});
    }
    migraphx::module m2 = m1;
    run_pass(m2);
    run_nd_pass(m1);
    EXPECT(m1 == m2);
}

// The convolution over more spatial dims must compute the same out as the reference
// convolution_backwards.
static void verify_nd_rewrite(const migraphx::operation& op,
                              const migraphx::shape& inp_shape,
                              const migraphx::shape& w_shape)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto inp = mm->add_parameter("inp", inp_shape);
    auto w   = mm->add_literal(migraphx::generate_literal(w_shape, 2));
    auto r   = mm->add_instruction(op, inp, w);
    mm->add_return({r});
    run_nd_pass(*mm);
    // One convolution over more dims than the input must replace the backward convolution,
    // otherwise the comparison below is vacuous.
    EXPECT(std::count_if(mm->begin(), mm->end(), [&](auto&& ins) {
               return ins.name() == "convolution" and ins.get_shape().ndim() > inp_shape.ndim();
           }) == 1);
    EXPECT(std::none_of(mm->begin(), mm->end(), [](auto&& ins) {
        return ins.name() == "convolution_backwards" or ins.name() == "concat";
    }));
    p.compile(migraphx::make_target("ref"));
    migraphx::parameter_map params;
    params["inp"] = migraphx::generate_argument(inp_shape, 1);
    auto got      = p.eval(params).back().to_vector<float>();
    auto ref      = eval_conv_backwards(op, false, inp_shape, w_shape).to_vector<float>();
    EXPECT(migraphx::verify::allclose(got, migraphx::verify::expected{ref}));
}

TEST_CASE(numeric_nd_kernel4_stride2)
{
    verify_nd_rewrite(
        migraphx::make_op("convolution_backwards",
                          {{"padding", {1, 1}}, {"stride", {2, 2}}, {"dilation", {1, 1}}}),
        sf({1, 2, 4, 4}),
        sf({2, 3, 4, 4}));
}

TEST_CASE(numeric_nd_kernel6_stride3)
{
    verify_nd_rewrite(
        migraphx::make_op("convolution_backwards",
                          {{"padding", {2, 1}}, {"stride", {3, 3}}, {"dilation", {1, 1}}}),
        sf({1, 2, 4, 4}),
        sf({2, 3, 6, 6}));
}

TEST_CASE(numeric_nd_stride_equals_kernel)
{
    verify_nd_rewrite(
        migraphx::make_op("convolution_backwards",
                          {{"padding", {0, 0}}, {"stride", {2, 2}}, {"dilation", {1, 1}}}),
        sf({1, 2, 4, 4}),
        sf({2, 3, 2, 2}));
}

TEST_CASE(numeric_nd_stride_asymmetric)
{
    verify_nd_rewrite(
        migraphx::make_op("convolution_backwards",
                          {{"padding", {1, 2}}, {"stride", {2, 3}}, {"dilation", {1, 1}}}),
        sf({1, 2, 4, 4}),
        sf({2, 3, 4, 6}));
}

// One dim has stride 1 and so a single residue
TEST_CASE(numeric_nd_stride_mixed)
{
    verify_nd_rewrite(
        migraphx::make_op("convolution_backwards",
                          {{"padding", {0, 1}}, {"stride", {1, 2}}, {"dilation", {1, 1}}}),
        sf({1, 2, 4, 4}),
        sf({2, 3, 3, 4}));
}

TEST_CASE(numeric_nd_1d)
{
    verify_nd_rewrite(migraphx::make_op("convolution_backwards",
                                        {{"padding", {1}}, {"stride", {2}}, {"dilation", {1}}}),
                      sf({1, 2, 5}),
                      sf({2, 3, 4}));
}

TEST_CASE(numeric_nd_3d)
{
    verify_nd_rewrite(
        migraphx::make_op("convolution_backwards",
                          {{"padding", {1, 1, 1}}, {"stride", {2, 2, 2}}, {"dilation", {1, 1, 1}}}),
        sf({1, 2, 3, 3, 3}),
        sf({2, 3, 4, 4, 4}));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
