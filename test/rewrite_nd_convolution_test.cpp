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
#include <migraphx/rewrite_nd_convolution.hpp>
#include <migraphx/rewrite_convolution.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/propagate_constant.hpp>
#include <migraphx/simplify_reshapes.hpp>
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
    migraphx::run_passes(m,
                         {migraphx::rewrite_nd_convolution{}, migraphx::dead_code_elimination{}});
}

static migraphx::shape sf(std::vector<std::size_t> lens)
{
    return migraphx::shape{migraphx::shape::float_type, std::move(lens)};
}

static migraphx::literal taps(const std::vector<int64_t>& values)
{
    return migraphx::literal{migraphx::shape{migraphx::shape::int64_type, {values.size()}}, values};
}

// Dims 3 and 5 have one input element and two outputs, each through one tap of the filter:
// the convolution is a 3x3 one with four times the output channels
TEST_CASE(expanding_dims)
{
    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", sf({1, 4, 3, 1, 3, 1}));
        auto w    = m1.add_parameter("w", sf({2, 4, 3, 2, 3, 2}));
        auto conv = m1.add_instruction(migraphx::make_op("convolution",
                                                         {{"padding", {1, 1, 1, 1, 1, 1, 1, 1}},
                                                          {"stride", {1, 1, 1, 1}},
                                                          {"dilation", {1, 1, 1, 1}}}),
                                       x,
                                       w);
        m1.add_return({conv});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x  = m2.add_parameter("x", sf({1, 4, 3, 1, 3, 1}));
        auto w  = m2.add_parameter("w", sf({2, 4, 3, 2, 3, 2}));
        auto i1 = m2.add_literal(taps({1, 0}));
        auto i2 = m2.add_literal(taps({1, 0}));
        auto xs = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {3, 5}}}), x);
        auto g1 = m2.add_instruction(migraphx::make_op("gather", {{"axis", 3}}), w, i1);
        auto r1 =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 4, 3, 2, 1, 3, 2}}}), g1);
        auto t1 = m2.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 3, 1, 4, 2, 5, 6}}}), r1);
        auto f1 = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {4, 4, 3, 3, 2}}}), t1);
        auto g2 = m2.add_instruction(migraphx::make_op("gather", {{"axis", 4}}), f1, i2);
        auto r2 =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {4, 4, 3, 3, 2, 1}}}), g2);
        auto t2 = m2.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 4, 1, 5, 2, 3}}}), r2);
        auto f2   = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {8, 4, 3, 3}}}), t2);
        auto conv = m2.add_instruction(
            migraphx::make_op(
                "convolution",
                {{"padding", {1, 1, 1, 1}}, {"stride", {1, 1}}, {"dilation", {1, 1}}}),
            xs,
            f2);
        auto rsp =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 2, 2, 2, 3, 3}}}), conv);
        auto out = m2.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 1, 4, 2, 5, 3}}}), rsp);
        m2.add_return({out});
    }
    EXPECT(m1 == m2);
}

// The input has its channels as the inner dim in memory, and so has the output: the folded
// convolution keeps them inner by putting the folded dims before the channels
TEST_CASE(expanding_dims_channels_inner)
{
    auto xs = migraphx::shape::from_permutation(
        migraphx::shape::float_type, {1, 4, 3, 1, 3, 1}, {0, 2, 3, 4, 5, 1});
    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", xs);
        auto w    = m1.add_parameter("w", sf({2, 4, 3, 2, 3, 2}));
        auto conv = m1.add_instruction(migraphx::make_op("convolution",
                                                         {{"padding", {1, 1, 1, 1, 1, 1, 1, 1}},
                                                          {"stride", {1, 1, 1, 1}},
                                                          {"dilation", {1, 1, 1, 1}}}),
                                       x,
                                       w);
        m1.add_return({conv});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x  = m2.add_parameter("x", xs);
        auto w  = m2.add_parameter("w", sf({2, 4, 3, 2, 3, 2}));
        auto i1 = m2.add_literal(taps({1, 0}));
        auto i2 = m2.add_literal(taps({1, 0}));
        auto xq = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {3, 5}}}), x);
        auto g1 = m2.add_instruction(migraphx::make_op("gather", {{"axis", 5}}), w, i1);
        auto r1 =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 4, 3, 2, 3, 2, 1}}}), g1);
        auto t1 = m2.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {5, 0, 6, 1, 2, 3, 4}}}), r1);
        auto f1 = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {4, 4, 3, 2, 3}}}), t1);
        auto g2 = m2.add_instruction(migraphx::make_op("gather", {{"axis", 3}}), f1, i2);
        auto r2 =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {4, 4, 3, 2, 1, 3}}}), g2);
        auto t2 = m2.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {3, 0, 4, 1, 2, 5}}}), r2);
        auto f2   = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {8, 4, 3, 3}}}), t2);
        auto conv = m2.add_instruction(
            migraphx::make_op(
                "convolution",
                {{"padding", {1, 1, 1, 1}}, {"stride", {1, 1}}, {"dilation", {1, 1}}}),
            xq,
            f2);
        auto rsp =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 2, 2, 2, 3, 3}}}), conv);
        auto out = m2.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 3, 4, 1, 5, 2}}}), rsp);
        m2.add_return({out});
    }
    EXPECT(m1 == m2);
}

// A contiguous and a bias that varies along the channels and the folded dim move to the
// folded convolution, where the bias has one value per output channel
TEST_CASE(expanding_dim_contiguous_bias)
{
    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", sf({1, 4, 1, 3, 3}));
        auto w    = m1.add_parameter("w", sf({2, 4, 2, 3, 3}));
        auto b    = m1.add_parameter("b", sf({1, 2, 2, 1, 1}));
        auto conv = m1.add_instruction(
            migraphx::make_op(
                "convolution",
                {{"padding", {1, 1, 1, 1, 1, 1}}, {"stride", {1, 1, 1}}, {"dilation", {1, 1, 1}}}),
            x,
            w);
        auto c    = m1.add_instruction(migraphx::make_op("contiguous"), conv);
        auto bias = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 2, 2, 3, 3}}}), b);
        auto add = m1.add_instruction(migraphx::make_op("add"), c, bias);
        m1.add_return({add});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x  = m2.add_parameter("x", sf({1, 4, 1, 3, 3}));
        auto w  = m2.add_parameter("w", sf({2, 4, 2, 3, 3}));
        auto b  = m2.add_parameter("b", sf({1, 2, 2, 1, 1}));
        auto i  = m2.add_literal(taps({1, 0}));
        auto xs = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {2}}}), x);
        auto g  = m2.add_instruction(migraphx::make_op("gather", {{"axis", 2}}), w, i);
        auto r =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 4, 2, 1, 3, 3}}}), g);
        auto t = m2.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 2, 1, 3, 4, 5}}}), r);
        auto f    = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {4, 4, 3, 3}}}), t);
        auto conv = m2.add_instruction(
            migraphx::make_op(
                "convolution",
                {{"padding", {1, 1, 1, 1}}, {"stride", {1, 1}}, {"dilation", {1, 1}}}),
            xs,
            f);
        auto c  = m2.add_instruction(migraphx::make_op("contiguous"), conv);
        auto bm = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 2, 2, 1, 1}}}), b);
        auto br   = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {4}}}), bm);
        auto bias = m2.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {1, 4, 3, 3}}}), br);
        auto add = m2.add_instruction(migraphx::make_op("add"), c, bias);
        auto out =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 2, 2, 3, 3}}}), add);
        m2.add_return({out});
    }
    EXPECT(m1 == m2);
}

// Four spatial dims are one more than max_spatial_dims: the first is folded into both
// channels. No tap connects output 0 to input 1 or output 2 to input 0, which read a tap of
// zeros padded after the filter.
TEST_CASE(fold_to_max_spatial_dims)
{
    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", sf({1, 2, 2, 3, 3, 3}));
        auto w    = m1.add_parameter("w", sf({3, 2, 2, 2, 2, 2}));
        auto conv = m1.add_instruction(migraphx::make_op("convolution",
                                                         {{"padding", {1, 0, 0, 0, 1, 0, 0, 0}},
                                                          {"stride", {1, 1, 1, 1}},
                                                          {"dilation", {1, 1, 1, 1}}}),
                                       x,
                                       w);
        m1.add_return({conv});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x  = m2.add_parameter("x", sf({1, 2, 2, 3, 3, 3}));
        auto w  = m2.add_parameter("w", sf({3, 2, 2, 2, 2, 2}));
        auto i  = m2.add_literal(taps({1, 2, 0, 1, 2, 0}));
        auto xr = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 4, 3, 3, 3}}}), x);
        auto p  = m2.add_instruction(
            migraphx::make_op("pad", {{"pads", {0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0}}}), w);
        auto g = m2.add_instruction(migraphx::make_op("gather", {{"axis", 2}}), p, i);
        auto r =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {3, 2, 3, 2, 2, 2, 2}}}), g);
        auto t = m2.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 2, 1, 3, 4, 5, 6}}}), r);
        auto f = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {9, 4, 2, 2, 2}}}), t);
        auto conv = m2.add_instruction(
            migraphx::make_op(
                "convolution",
                {{"padding", {0, 0, 0, 0, 0, 0}}, {"stride", {1, 1, 1}}, {"dilation", {1, 1, 1}}}),
            xr,
            f);
        auto out =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 3, 3, 2, 2, 2}}}), conv);
        m2.add_return({out});
    }
    EXPECT(m1 == m2);
}

// A dim with one input element and one output is an ordinary dim of a 3-D convolution
TEST_CASE(unit_dim_not_expanding)
{
    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", sf({1, 4, 1, 3, 3}));
        auto w    = m1.add_parameter("w", sf({2, 4, 3, 3, 3}));
        auto conv = m1.add_instruction(
            migraphx::make_op(
                "convolution",
                {{"padding", {1, 1, 1}}, {"stride", {1, 1, 1}}, {"dilation", {1, 1, 1}}}),
            x,
            w);
        m1.add_return({conv});
    }
    migraphx::module m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(grouped_not_folded)
{
    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", sf({1, 4, 1, 3, 3}));
        auto w    = m1.add_parameter("w", sf({4, 2, 2, 3, 3}));
        auto conv = m1.add_instruction(migraphx::make_op("convolution",
                                                         {{"padding", {1, 1, 1}},
                                                          {"stride", {1, 1, 1}},
                                                          {"dilation", {1, 1, 1}},
                                                          {"group", 2}}),
                                       x,
                                       w);
        m1.add_return({conv});
    }
    migraphx::module m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

// The pass undoes the convolution rewrite_convolution makes from a pixel shuffle
TEST_CASE(pixel_shuffle_round_trip)
{
    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", sf({1, 4, 3, 3}));
        auto w    = m1.add_literal(migraphx::generate_literal(sf({8, 4, 3, 3}), 1));
        auto b    = m1.add_literal(migraphx::generate_literal(sf({8}), 2));
        auto conv = m1.add_instruction(
            migraphx::make_op(
                "convolution",
                {{"padding", {1, 1, 1, 1}}, {"stride", {1, 1}}, {"dilation", {1, 1}}}),
            x,
            w);
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
    migraphx::module m2 = m1;
    migraphx::run_passes(m1,
                         {migraphx::rewrite_convolution{.enable_nd_convolution = true},
                          migraphx::dead_code_elimination{},
                          migraphx::rewrite_nd_convolution{},
                          migraphx::dead_code_elimination{},
                          migraphx::propagate_constant{},
                          migraphx::dead_code_elimination{},
                          migraphx::simplify_reshapes{},
                          migraphx::dead_code_elimination{}});
    EXPECT(m1.sort() == m2.sort());
}

// Evaluate convolution(x, w) on the reference target, with or without the pass
static std::vector<float> eval_conv(const migraphx::value& attrs,
                                    bool rewrite,
                                    const migraphx::shape& x_shape,
                                    const migraphx::shape& w_shape)
{
    migraphx::program p;
    auto* mm  = p.get_main_module();
    auto x    = mm->add_parameter("x", x_shape);
    auto w    = mm->add_parameter("w", w_shape);
    auto conv = mm->add_instruction(migraphx::make_op("convolution", attrs), x, w);
    mm->add_return({conv});
    auto ndim = conv->get_shape().ndim();
    if(rewrite)
        run_pass(*mm);
    // The rewrite must fire, otherwise the comparison is vacuous
    bool folded = std::none_of(mm->begin(), mm->end(), [&](const auto& ins) {
        return ins.name() == "convolution" and ins.get_shape().ndim() == ndim;
    });
    EXPECT(folded == rewrite);
    p.compile(migraphx::make_target("ref"));
    migraphx::parameter_map params;
    params["x"] = migraphx::generate_argument(x_shape, 1);
    params["w"] = migraphx::generate_argument(w_shape, 2);
    return p.eval(params).back().to_vector<float>();
}

static void verify_fold(const migraphx::value& attrs,
                        const migraphx::shape& x_shape,
                        const migraphx::shape& w_shape)
{
    auto ref = eval_conv(attrs, false, x_shape, w_shape);
    auto got = eval_conv(attrs, true, x_shape, w_shape);
    EXPECT(migraphx::verify::allclose(got, migraphx::verify::expected{ref}));
}

TEST_CASE(numeric_pixel_shuffle)
{
    verify_fold({{"padding", {1, 1, 1, 1, 1, 1, 1, 1}},
                 {"stride", {1, 1, 1, 1}},
                 {"dilation", {1, 1, 1, 1}}},
                sf({1, 4, 3, 1, 3, 1}),
                sf({2, 4, 3, 2, 3, 2}));
}

TEST_CASE(numeric_pixel_shuffle_channels_inner)
{
    verify_fold({{"padding", {1, 1, 1, 1, 1, 1, 1, 1}},
                 {"stride", {1, 1, 1, 1}},
                 {"dilation", {1, 1, 1, 1}}},
                migraphx::shape::from_permutation(
                    migraphx::shape::float_type, {1, 4, 3, 1, 3, 1}, {0, 2, 3, 4, 5, 1}),
                sf({3, 4, 3, 2, 3, 2}));
}

// A dim with several input elements is folded along with the channels of the input
TEST_CASE(numeric_fold_two_dims_channels_inner)
{
    verify_fold({{"padding", {1, 1, 1, 1, 0, 1, 1, 1, 1, 0}},
                 {"stride", {1, 1, 1, 1, 1}},
                 {"dilation", {1, 1, 1, 1, 1}}},
                migraphx::shape::from_permutation(
                    migraphx::shape::float_type, {1, 2, 3, 1, 3, 2, 2}, {0, 2, 3, 4, 5, 6, 1}),
                sf({3, 2, 3, 2, 3, 2, 2}));
}

// Outputs 0 and 4 of the first dim read no input: a padding of 4 with a dilation of 2 puts
// taps 2, 1 and 0 on outputs 0, 2 and 4, and the filter has 2 taps
TEST_CASE(numeric_expanding_dim_missing_taps)
{
    verify_fold({{"padding", {4, 1, 4, 1}}, {"stride", {1, 1}}, {"dilation", {2, 1}}},
                sf({2, 3, 1, 5}),
                sf({2, 3, 2, 3}));
}

TEST_CASE(numeric_expanding_dim_stride)
{
    verify_fold({{"padding", {2, 0, 2, 0}}, {"stride", {2, 1}}, {"dilation", {1, 1}}},
                sf({1, 3, 1, 5}),
                sf({2, 3, 3, 2}));
}

TEST_CASE(numeric_fold_to_max_spatial_dims)
{
    verify_fold({{"padding", {1, 0, 0, 0, 1, 0, 0, 0}},
                 {"stride", {1, 1, 1, 1}},
                 {"dilation", {1, 1, 1, 1}}},
                sf({1, 2, 2, 3, 3, 3}),
                sf({3, 2, 2, 2, 2, 2}));
}

// Two dims are folded: a strided one and one with asymmetric padding
TEST_CASE(numeric_fold_two_dims)
{
    verify_fold({{"padding", {1, 0, 1, 0, 0, 0, 1, 2, 0, 1}},
                 {"stride", {2, 1, 1, 1, 1}},
                 {"dilation", {1, 1, 1, 1, 1}}},
                sf({2, 2, 3, 4, 2, 4, 4}),
                sf({3, 2, 2, 3, 2, 2, 3}));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
