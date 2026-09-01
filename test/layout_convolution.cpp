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
#include <migraphx/layout_convolution.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <basic_ops.hpp>
#include <migraphx/make_op.hpp>

#include <test.hpp>

static void run_pass(migraphx::module& m, const migraphx::layout_convolution& lc = {})
{
    migraphx::run_passes(m, {lc, migraphx::dead_code_elimination{}});
}

static migraphx::operation layout(std::vector<int64_t> permutation = {0, 1, 2, 3})
{
    return migraphx::make_op("layout", {{"permutation", permutation}});
}

static migraphx::instruction_ref add_layout_nhwc(migraphx::module& m, migraphx::instruction_ref ins)
{
    return m.add_instruction(layout({0, 2, 3, 1}), ins);
}

static migraphx::instruction_ref add_layout_nchw(migraphx::module& m, migraphx::instruction_ref ins)
{
    return m.add_instruction(layout({0, 1, 2, 3}), ins);
}

static migraphx::instruction_ref add_layout_yxck(migraphx::module& m, migraphx::instruction_ref ins)
{
    return m.add_instruction(layout({2, 3, 1, 0}), ins);
}

TEST_CASE(auto_conv_nchw)
{
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::float_type, {1, 8, 16, 16}});
        auto w = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {16, 8, 3, 3}}));
        auto conv = m1.add_instruction(
            migraphx::make_op("convolution",
                              {{"padding", {1, 1}}, {"stride", {2, 2}}, {"dilation", {1, 1}}}),
            x,
            w);
        auto relu = m1.add_instruction(migraphx::make_op("relu"), conv);
        m1.add_return({relu});
    }
    migraphx::module m2 = m1;
    run_pass(m1);
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(auto_conv_nhwc)
{
    auto transpose = migraphx::make_op("transpose", {{"permutation", {0, 3, 1, 2}}});
    migraphx::module m1;
    {
        auto x          = m1.add_parameter("x", {migraphx::shape::float_type, {1, 16, 16, 8}});
        auto xtranspose = m1.add_instruction(transpose, x);
        auto w          = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {16, 3, 3, 8}}));
        auto wtranspose = m1.add_instruction(transpose, w);
        auto conv       = m1.add_instruction(
            migraphx::make_op("convolution",
                                    {{"padding", {1, 1}}, {"stride", {2, 2}}, {"dilation", {1, 1}}}),
            xtranspose,
            wtranspose);
        auto relu = m1.add_instruction(migraphx::make_op("relu"), conv);
        m1.add_return({relu});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x          = m2.add_parameter("x", {migraphx::shape::float_type, {1, 16, 16, 8}});
        auto xtranspose = add_layout_nchw(m2, m2.add_instruction(transpose, x));
        auto w          = m2.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {16, 3, 3, 8}}));
        auto wtranspose = add_layout_nchw(m2, m2.add_instruction(transpose, w));
        auto conv       = m2.add_instruction(
            migraphx::make_op("convolution",
                                    {{"padding", {1, 1}}, {"stride", {2, 2}}, {"dilation", {1, 1}}}),
            xtranspose,
            wtranspose);
        auto relu = add_layout_nhwc(m2, m2.add_instruction(migraphx::make_op("relu"), conv));
        m2.add_return({relu});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(auto_conv_mixed)
{
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::float_type, {1, 8, 16, 16}});
        auto w = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {3, 3, 16, 8}}));
        auto wtranspose =
            m1.add_instruction(migraphx::make_op("transpose", {{"permutation", {2, 3, 0, 1}}}), w);
        auto conv = m1.add_instruction(
            migraphx::make_op("convolution",
                              {{"padding", {1, 1}}, {"stride", {2, 2}}, {"dilation", {1, 1}}}),
            x,
            wtranspose);
        auto relu = m1.add_instruction(migraphx::make_op("relu"), conv);
        m1.add_return({relu});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", {migraphx::shape::float_type, {1, 8, 16, 16}});
        auto w = m2.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {3, 3, 16, 8}}));
        auto wtranspose =
            m2.add_instruction(migraphx::make_op("transpose", {{"permutation", {2, 3, 0, 1}}}), w);
        auto wlayout = m2.add_instruction(
            migraphx::make_op("layout", {{"permutation", {0, 1, 2, 3}}}), wtranspose);
        auto conv = m2.add_instruction(
            migraphx::make_op("convolution",
                              {{"padding", {1, 1}}, {"stride", {2, 2}}, {"dilation", {1, 1}}}),
            x,
            wlayout);
        auto relu = m2.add_instruction(migraphx::make_op("relu"), conv);
        m2.add_return({relu});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(auto_quant_conv_mixed)
{
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::int8_type, {1, 8, 16, 16}});
        auto w =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::int8_type, {3, 3, 16, 8}}));
        auto wtranspose =
            m1.add_instruction(migraphx::make_op("transpose", {{"permutation", {2, 3, 0, 1}}}), w);
        auto conv = m1.add_instruction(
            migraphx::make_op("quant_convolution",
                              {{"padding", {1, 1}}, {"stride", {2, 2}}, {"dilation", {1, 1}}}),
            x,
            wtranspose);
        auto relu = m1.add_instruction(migraphx::make_op("relu"), conv);
        m1.add_return({relu});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", {migraphx::shape::int8_type, {1, 8, 16, 16}});
        auto w =
            m2.add_literal(migraphx::generate_literal({migraphx::shape::int8_type, {3, 3, 16, 8}}));
        auto wtranspose =
            m2.add_instruction(migraphx::make_op("transpose", {{"permutation", {2, 3, 0, 1}}}), w);
        auto wlayout = m2.add_instruction(
            migraphx::make_op("layout", {{"permutation", {0, 1, 2, 3}}}), wtranspose);
        auto conv = m2.add_instruction(
            migraphx::make_op("quant_convolution",
                              {{"padding", {1, 1}}, {"stride", {2, 2}}, {"dilation", {1, 1}}}),
            x,
            wlayout);
        auto relu = m2.add_instruction(migraphx::make_op("relu"), conv);
        m2.add_return({relu});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(nhwc_conv_relu)
{
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::float_type, {1, 8, 16, 16}});
        auto w = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {16, 8, 3, 3}}));
        auto conv = m1.add_instruction(
            migraphx::make_op("convolution",
                              {{"padding", {1, 1}}, {"stride", {2, 2}}, {"dilation", {1, 1}}}),
            x,
            w);
        m1.add_instruction(migraphx::make_op("relu"), conv);
    }
    run_pass(m1, {.order = migraphx::layout_convolution::channels_last});

    migraphx::module m2;
    {
        auto x = add_layout_nhwc(
            m2, m2.add_parameter("x", {migraphx::shape::float_type, {1, 8, 16, 16}}));
        auto w    = add_layout_nhwc(m2,
                                 m2.add_literal(migraphx::generate_literal(
                                     {migraphx::shape::float_type, {16, 8, 3, 3}})));
        auto conv = m2.add_instruction(
            migraphx::make_op("convolution",
                              {{"padding", {1, 1}}, {"stride", {2, 2}}, {"dilation", {1, 1}}}),
            x,
            w);
        auto relu = m2.add_instruction(migraphx::make_op("relu"), conv);
        m2.add_instruction(layout(), relu);
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(nhwc_conv_add)
{
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::float_type, {1, 8, 16, 16}});
        auto w = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {16, 8, 3, 3}}));
        auto y    = m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {16}}));
        auto conv = m1.add_instruction(
            migraphx::make_op("convolution",
                              {{"padding", {1, 1}}, {"stride", {2, 2}}, {"dilation", {1, 1}}}),
            x,
            w);
        auto b = m1.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", conv->get_shape().lens()}}),
            y);
        m1.add_instruction(migraphx::make_op("add"), conv, b);
    }
    run_pass(m1, {.order = migraphx::layout_convolution::channels_last});

    migraphx::module m2;
    {
        auto x = add_layout_nhwc(
            m2, m2.add_parameter("x", {migraphx::shape::float_type, {1, 8, 16, 16}}));
        auto w    = add_layout_nhwc(m2,
                                 m2.add_literal(migraphx::generate_literal(
                                     {migraphx::shape::float_type, {16, 8, 3, 3}})));
        auto y    = m2.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {16}}));
        auto conv = m2.add_instruction(
            migraphx::make_op("convolution",
                              {{"padding", {1, 1}}, {"stride", {2, 2}}, {"dilation", {1, 1}}}),
            x,
            w);
        auto b = m2.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", conv->get_shape().lens()}}),
            y);
        auto add = m2.add_instruction(migraphx::make_op("add"), conv, b);
        m2.add_instruction(layout(), add);
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(nhwc_quant_conv_add)
{
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::int8_type, {1, 8, 16, 16}});
        auto w =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::int8_type, {16, 8, 3, 3}}));
        auto y    = m1.add_literal(migraphx::generate_literal({migraphx::shape::int32_type, {16}}));
        auto conv = m1.add_instruction(
            migraphx::make_op("quant_convolution",
                              {{"padding", {1, 1}}, {"stride", {2, 2}}, {"dilation", {1, 1}}}),
            x,
            w);
        auto b = m1.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", conv->get_shape().lens()}}),
            y);
        m1.add_instruction(migraphx::make_op("add"), conv, b);
    }
    run_pass(m1, {.order = migraphx::layout_convolution::channels_last});

    migraphx::module m2;
    {
        auto x = add_layout_nhwc(
            m2, m2.add_parameter("x", {migraphx::shape::int8_type, {1, 8, 16, 16}}));
        auto w    = add_layout_nhwc(m2,
                                 m2.add_literal(migraphx::generate_literal(
                                     {migraphx::shape::int8_type, {16, 8, 3, 3}})));
        auto y    = m2.add_literal(migraphx::generate_literal({migraphx::shape::int32_type, {16}}));
        auto conv = m2.add_instruction(
            migraphx::make_op("quant_convolution",
                              {{"padding", {1, 1}}, {"stride", {2, 2}}, {"dilation", {1, 1}}}),
            x,
            w);
        auto b = m2.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", conv->get_shape().lens()}}),
            y);
        auto add = m2.add_instruction(migraphx::make_op("add"), conv, b);
        m2.add_instruction(layout(), add);
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(nhwc_conv_conv)
{
    migraphx::module m1;
    {
        auto x  = m1.add_parameter("x", {migraphx::shape::float_type, {1, 2048, 7, 7}});
        auto w1 = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {512, 2048, 1, 1}}));
        auto conv1 = m1.add_instruction(migraphx::make_op("convolution"), x, w1);
        auto y1 = m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {512}}));
        auto b1 = m1.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", conv1->get_shape().lens()}}),
            y1);
        auto add1  = m1.add_instruction(migraphx::make_op("add"), conv1, b1);
        auto relu1 = m1.add_instruction(migraphx::make_op("relu"), add1);
        auto w2    = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {512, 512, 3, 3}}));
        auto conv2 = m1.add_instruction(
            migraphx::make_op("convolution", {{"padding", {1, 1, 1, 1}}}), relu1, w2);
        auto y2 = m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {512}}));
        auto b2 = m1.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", conv1->get_shape().lens()}}),
            y2);
        auto add2  = m1.add_instruction(migraphx::make_op("add"), conv2, b2);
        auto relu2 = m1.add_instruction(migraphx::make_op("relu"), add2);
        m1.add_return({relu2});
    }
    run_pass(m1, {.order = migraphx::layout_convolution::channels_last});

    migraphx::module m2;
    {
        auto x = add_layout_nhwc(
            m2, m2.add_parameter("x", {migraphx::shape::float_type, {1, 2048, 7, 7}}));
        auto w1    = add_layout_nhwc(m2,
                                  m2.add_literal(migraphx::generate_literal(
                                      {migraphx::shape::float_type, {512, 2048, 1, 1}})));
        auto conv1 = m2.add_instruction(migraphx::make_op("convolution"), x, w1);
        auto y1 = m2.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {512}}));
        auto b1 = m2.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", conv1->get_shape().lens()}}),
            y1);
        auto add1  = m2.add_instruction(migraphx::make_op("add"), conv1, b1);
        auto relu1 = m2.add_instruction(migraphx::make_op("relu"), add1);
        auto w2    = add_layout_nhwc(m2,
                                  m2.add_literal(migraphx::generate_literal(
                                      {migraphx::shape::float_type, {512, 512, 3, 3}})));
        auto conv2 = m2.add_instruction(
            migraphx::make_op("convolution", {{"padding", {1, 1, 1, 1}}}), relu1, w2);
        auto y2 = m2.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {512}}));
        auto b2 = m2.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", conv1->get_shape().lens()}}),
            y2);
        auto add2        = m2.add_instruction(migraphx::make_op("add"), conv2, b2);
        auto relu2       = m2.add_instruction(migraphx::make_op("relu"), add2);
        auto relu_layout = m2.add_instruction(layout(), relu2);
        m2.add_return({relu_layout});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(nhwc_conv_concat_conv)
{
    migraphx::module m1;
    {
        auto x  = m1.add_parameter("x", {migraphx::shape::float_type, {1, 8, 16, 16}});
        auto w1 = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {15, 8, 3, 3}}));
        auto conv1 = m1.add_instruction(
            migraphx::make_op("convolution", {{"padding", {1, 1, 1, 1}}}), x, w1);
        auto c = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {1, 1, 16, 16}}));
        auto concat = m1.add_instruction(migraphx::make_op("concat", {{"axis", 1}}), conv1, c);
        auto w2     = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {4, 16, 3, 3}}));
        auto conv2 = m1.add_instruction(
            migraphx::make_op("convolution", {{"padding", {1, 1, 1, 1}}}), concat, w2);
        m1.add_return({conv2});
    }
    run_pass(m1, {.order = migraphx::layout_convolution::channels_last});

    migraphx::module m2;
    {
        auto x = add_layout_nhwc(
            m2, m2.add_parameter("x", {migraphx::shape::float_type, {1, 8, 16, 16}}));
        auto w1    = add_layout_nhwc(m2,
                                  m2.add_literal(migraphx::generate_literal(
                                      {migraphx::shape::float_type, {15, 8, 3, 3}})));
        auto conv1 = m2.add_instruction(
            migraphx::make_op("convolution", {{"padding", {1, 1, 1, 1}}}), x, w1);
        auto c = m2.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {1, 1, 16, 16}}));
        // The singleton-channel literal is layout-ambiguous, so the concat
        // output stays channels-last and needs no relayout before conv2.
        auto concat = m2.add_instruction(migraphx::make_op("concat", {{"axis", 1}}), conv1, c);
        auto w2     = add_layout_nhwc(m2,
                                  m2.add_literal(migraphx::generate_literal(
                                      {migraphx::shape::float_type, {4, 16, 3, 3}})));
        auto conv2  = m2.add_instruction(
            migraphx::make_op("convolution", {{"padding", {1, 1, 1, 1}}}), concat, w2);
        auto conv2_layout = m2.add_instruction(layout(), conv2);
        m2.add_return({conv2_layout});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(nchw_conv_output_channels_last_unchanged)
{
    // output_channels_last only applies to channels_last: NCHW graphs are untouched.
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::float_type, {1, 8, 16, 16}});
        auto w =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {4, 8, 3, 3}}));
        auto conv =
            m1.add_instruction(migraphx::make_op("convolution", {{"padding", {1, 1, 1, 1}}}), x, w);
        m1.add_return({conv});
    }
    migraphx::module m2 = m1;
    run_pass(m1,
             {.order                          = migraphx::layout_convolution::channels_first,
              .output_channels_last_threshold = 1});
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(nhwc_conv_output_channels_last)
{
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::float_type, {1, 8, 16, 16}});
        auto w =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {8, 8, 3, 3}}));
        auto conv =
            m1.add_instruction(migraphx::make_op("convolution", {{"padding", {1, 1, 1, 1}}}), x, w);
        m1.add_return({conv});
    }
    run_pass(m1,
             {.order                          = migraphx::layout_convolution::channels_last,
              .output_channels_last_threshold = 8});

    migraphx::module m2;
    {
        auto x = add_layout_nhwc(
            m2, m2.add_parameter("x", {migraphx::shape::float_type, {1, 8, 16, 16}}));
        auto w = add_layout_yxck(m2,
                                 m2.add_literal(migraphx::generate_literal(
                                     {migraphx::shape::float_type, {8, 8, 3, 3}})));
        auto conv =
            m2.add_instruction(migraphx::make_op("convolution", {{"padding", {1, 1, 1, 1}}}), x, w);
        auto conv_layout = m2.add_instruction(layout(), conv);
        m2.add_return({conv_layout});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(nhwc_conv_output_channels_last_small_k)
{
    // Output channels below the threshold keep kyxc: same result as with the flag off.
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::float_type, {1, 8, 16, 16}});
        auto w =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {3, 8, 3, 3}}));
        auto conv =
            m1.add_instruction(migraphx::make_op("convolution", {{"padding", {1, 1, 1, 1}}}), x, w);
        m1.add_return({conv});
    }
    migraphx::module m2 = m1;
    run_pass(m1,
             {.order                          = migraphx::layout_convolution::channels_last,
              .output_channels_last_threshold = 8});
    run_pass(m2, {.order = migraphx::layout_convolution::channels_last});
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(nhwc_conv_output_channels_last_always)
{
    // A threshold of 1 always stores the weights output-channels-last.
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::float_type, {1, 8, 16, 16}});
        auto w =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {3, 8, 3, 3}}));
        auto conv =
            m1.add_instruction(migraphx::make_op("convolution", {{"padding", {1, 1, 1, 1}}}), x, w);
        m1.add_return({conv});
    }
    run_pass(m1,
             {.order                          = migraphx::layout_convolution::channels_last,
              .output_channels_last_threshold = 1});

    migraphx::module m2;
    {
        auto x = add_layout_nhwc(
            m2, m2.add_parameter("x", {migraphx::shape::float_type, {1, 8, 16, 16}}));
        auto w = add_layout_yxck(m2,
                                 m2.add_literal(migraphx::generate_literal(
                                     {migraphx::shape::float_type, {3, 8, 3, 3}})));
        auto conv =
            m2.add_instruction(migraphx::make_op("convolution", {{"padding", {1, 1, 1, 1}}}), x, w);
        auto conv_layout = m2.add_instruction(layout(), conv);
        m2.add_return({conv_layout});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(nhwc_conv_output_channels_last_all_types)
{
    // An empty type list applies output_channels_last to every type.
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::half_type, {1, 8, 16, 16}});
        auto w =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::half_type, {8, 8, 3, 3}}));
        auto conv =
            m1.add_instruction(migraphx::make_op("convolution", {{"padding", {1, 1, 1, 1}}}), x, w);
        m1.add_return({conv});
    }
    run_pass(m1,
             {.order                          = migraphx::layout_convolution::channels_last,
              .output_channels_last_threshold = 1});

    migraphx::module m2;
    {
        auto x = add_layout_nhwc(
            m2, m2.add_parameter("x", {migraphx::shape::half_type, {1, 8, 16, 16}}));
        auto w = add_layout_yxck(
            m2,
            m2.add_literal(migraphx::generate_literal({migraphx::shape::half_type, {8, 8, 3, 3}})));
        auto conv =
            m2.add_instruction(migraphx::make_op("convolution", {{"padding", {1, 1, 1, 1}}}), x, w);
        auto conv_layout = m2.add_instruction(layout(), conv);
        m2.add_return({conv_layout});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(nhwc_conv_output_channels_last_type_filtered)
{
    // A type not in output_channels_last_types keeps kyxc.
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::half_type, {1, 8, 16, 16}});
        auto w =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::half_type, {8, 8, 3, 3}}));
        auto conv =
            m1.add_instruction(migraphx::make_op("convolution", {{"padding", {1, 1, 1, 1}}}), x, w);
        m1.add_return({conv});
    }
    migraphx::module m2 = m1;
    run_pass(m1,
             {.order                          = migraphx::layout_convolution::channels_last,
              .output_channels_last_threshold = 1,
              .output_channels_last_types     = {migraphx::shape::float_type}});
    run_pass(m2, {.order = migraphx::layout_convolution::channels_last});
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(nhwc_quant_conv_output_channels_last_unchanged)
{
    // int8 is not in output_channels_last_types, so quant_convolution keeps kyxc.
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::int8_type, {1, 8, 16, 16}});
        auto w =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::int8_type, {4, 8, 3, 3}}));
        auto conv = m1.add_instruction(
            migraphx::make_op("quant_convolution", {{"padding", {1, 1, 1, 1}}}), x, w);
        m1.add_return({conv});
    }
    migraphx::module m2 = m1;
    run_pass(m1,
             {.order                          = migraphx::layout_convolution::channels_last,
              .output_channels_last_threshold = 1,
              .output_channels_last_types     = {migraphx::shape::float_type}});
    run_pass(m2, {.order = migraphx::layout_convolution::channels_last});
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(nhwc_conv_reduce)
{
    migraphx::module m1;
    {
        auto x  = m1.add_parameter("x", {migraphx::shape::float_type, {1, 2048, 7, 7}});
        auto w1 = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {512, 2048, 1, 1}}));
        auto conv1 = m1.add_instruction(migraphx::make_op("convolution"), x, w1);
        auto y1 = m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {512}}));
        auto b1 = m1.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", conv1->get_shape().lens()}}),
            y1);
        auto add1  = m1.add_instruction(migraphx::make_op("add"), conv1, b1);
        auto relu1 = m1.add_instruction(migraphx::make_op("relu"), add1);
        auto reduce =
            m1.add_instruction(migraphx::make_op("reduce_mean", {{"axes", {2, 3}}}), relu1);
        auto squeeze = m1.add_instruction(migraphx::make_op("squeeze", {{"axes", {2, 3}}}), reduce);
        m1.add_return({squeeze});
    }
    run_pass(m1, {.order = migraphx::layout_convolution::channels_last});

    migraphx::module m2;
    {
        auto x = add_layout_nhwc(
            m2, m2.add_parameter("x", {migraphx::shape::float_type, {1, 2048, 7, 7}}));
        auto w1    = add_layout_nhwc(m2,
                                  m2.add_literal(migraphx::generate_literal(
                                      {migraphx::shape::float_type, {512, 2048, 1, 1}})));
        auto conv1 = m2.add_instruction(migraphx::make_op("convolution"), x, w1);
        auto y1 = m2.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {512}}));
        auto b1 = m2.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", conv1->get_shape().lens()}}),
            y1);
        auto add1  = m2.add_instruction(migraphx::make_op("add"), conv1, b1);
        auto relu1 = m2.add_instruction(migraphx::make_op("relu"), add1);
        auto reduce =
            m2.add_instruction(migraphx::make_op("reduce_mean", {{"axes", {2, 3}}}), relu1);
        auto squeeze = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {2, 3}}}), reduce);
        m2.add_return({squeeze});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(nhwc_group_conv)
{
    migraphx::module m1;
    {
        auto x  = m1.add_parameter("x", {migraphx::shape::float_type, {64, 96, 80, 80}});
        auto w1 = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {64, 96, 1, 1}}));
        auto w2 = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {64, 1, 3, 3}}));
        auto conv1 = m1.add_instruction(migraphx::make_op("convolution"), x, w1);
        auto relu1 = m1.add_instruction(migraphx::make_op("relu"), conv1);
        auto conv2 =
            m1.add_instruction(migraphx::make_op("convolution", {{"group", 64}}), relu1, w2);
        auto relu2 = m1.add_instruction(migraphx::make_op("relu"), conv2);
        m1.add_return({relu2});
    }
    run_pass(m1, {.order = migraphx::layout_convolution::channels_last});

    migraphx::module m2;
    {
        auto x = add_layout_nhwc(
            m2, m2.add_parameter("x", {migraphx::shape::float_type, {64, 96, 80, 80}}));
        auto w1    = add_layout_nhwc(m2,
                                  m2.add_literal(migraphx::generate_literal(
                                      {migraphx::shape::float_type, {64, 96, 1, 1}})));
        auto w2    = add_layout_nchw(m2,
                                  m2.add_literal(migraphx::generate_literal(
                                      {migraphx::shape::float_type, {64, 1, 3, 3}})));
        auto conv1 = m2.add_instruction(migraphx::make_op("convolution"), x, w1);
        auto relu1 = add_layout_nhwc(m2, m2.add_instruction(migraphx::make_op("relu"), conv1));
        auto conv2 =
            m2.add_instruction(migraphx::make_op("convolution", {{"group", 64}}), relu1, w2);
        auto relu2 = add_layout_nchw(m2, m2.add_instruction(migraphx::make_op("relu"), conv2));
        m2.add_return({relu2});
    }
}

// channels_auto runs both layouts and keeps whichever leaves fewer layout/contiguous ops.
// This graph is already in channels-last form, so channels_last is a no-op while
// channels_first would insert layout ops, so auto must select channels_last.
TEST_CASE(channels_auto_selects_channels_last)
{
    auto transpose = migraphx::make_op("transpose", {{"permutation", {0, 3, 1, 2}}});
    migraphx::module m1;
    {
        auto x          = m1.add_parameter("x", {migraphx::shape::float_type, {1, 16, 16, 8}});
        auto xtranspose = m1.add_instruction(transpose, x);
        auto w          = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {16, 3, 3, 8}}));
        auto wtranspose = m1.add_instruction(transpose, w);
        auto conv       = m1.add_instruction(
            migraphx::make_op("convolution",
                                    {{"padding", {1, 1}}, {"stride", {2, 2}}, {"dilation", {1, 1}}}),
            xtranspose,
            wtranspose);
        auto relu = m1.add_instruction(migraphx::make_op("relu"), conv);
        m1.add_return({relu});
    }
    migraphx::module m2 = m1;
    run_pass(m1, {.order = migraphx::layout_convolution::channels_auto});
    EXPECT(m1.sort() == m2.sort());
}

// A grouped convolution stays NCHW, so a normal-group-normal chain forces channels_last to
// insert four non-foldable layouts (the input, both convolution boundaries, and the output),
// while the layouts on the constant weights fold away and are not scored. That exceeds the
// two-per-param allowance, so auto selects channels_first (a no-op here).
TEST_CASE(channels_auto_selects_channels_first)
{
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::float_type, {1, 8, 8, 8}});
        auto w1 =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {8, 8, 1, 1}}));
        auto w2 =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {8, 1, 3, 3}}));
        auto w3 =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {8, 8, 1, 1}}));
        auto conv1 = m1.add_instruction(migraphx::make_op("convolution"), x, w1);
        auto relu1 = m1.add_instruction(migraphx::make_op("relu"), conv1);
        auto conv2 = m1.add_instruction(
            migraphx::make_op("convolution", {{"group", 8}, {"padding", {1, 1}}}), relu1, w2);
        auto relu2 = m1.add_instruction(migraphx::make_op("relu"), conv2);
        auto conv3 = m1.add_instruction(migraphx::make_op("convolution"), relu2, w3);
        auto relu3 = m1.add_instruction(migraphx::make_op("relu"), conv3);
        m1.add_return({relu3});
    }
    migraphx::module m2 = m1;
    run_pass(m1, {.order = migraphx::layout_convolution::channels_auto});
    EXPECT(m1.sort() == m2.sort());
}

// channels_first would need one layout (on the transposed weights) versus three for
// channels_last, but the two-layouts-per-param allowance offsets that difference, so auto
// selects channels_last. The expected module is the non-trivial channels_last result.
TEST_CASE(channels_auto_allows_two_layouts_per_param)
{
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::float_type, {1, 8, 16, 16}});
        auto w = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {3, 3, 16, 8}}));
        auto wtranspose =
            m1.add_instruction(migraphx::make_op("transpose", {{"permutation", {2, 3, 0, 1}}}), w);
        auto conv = m1.add_instruction(
            migraphx::make_op("convolution",
                              {{"padding", {1, 1}}, {"stride", {2, 2}}, {"dilation", {1, 1}}}),
            x,
            wtranspose);
        auto relu = m1.add_instruction(migraphx::make_op("relu"), conv);
        m1.add_return({relu});
    }
    run_pass(m1, {.order = migraphx::layout_convolution::channels_auto});

    migraphx::module m2;
    {
        auto x = add_layout_nhwc(
            m2, m2.add_parameter("x", {migraphx::shape::float_type, {1, 8, 16, 16}}));
        auto w = m2.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {3, 3, 16, 8}}));
        auto wtranspose =
            m2.add_instruction(migraphx::make_op("transpose", {{"permutation", {2, 3, 0, 1}}}), w);
        auto wlayout = add_layout_nhwc(m2, wtranspose);
        auto conv    = m2.add_instruction(
            migraphx::make_op("convolution",
                                 {{"padding", {1, 1}}, {"stride", {2, 2}}, {"dilation", {1, 1}}}),
            x,
            wlayout);
        auto relu = m2.add_instruction(migraphx::make_op("relu"), conv);
        auto out  = m2.add_instruction(layout(), relu);
        m2.add_return({out});
    }
    EXPECT(m1.sort() == m2.sort());
}

// Under channels_last the convolution output is NHWC, so a reshape that folds the channel dim
// into the spatial dims cannot alias its input as a view (reshape_lazy) and needs a copy, so
// it is scored like a contiguous. Two such reshapes push channels_last to a score of three,
// past the two-per-param allowance, so auto selects channels_first (a no-op here, since a
// standard output keeps the reshapes lazy).
TEST_CASE(channels_auto_counts_nonlazy_reshapes)
{
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::float_type, {1, 8, 16, 16}});
        auto w = m1.add_literal(
            migraphx::generate_literal({migraphx::shape::float_type, {16, 8, 3, 3}}));
        auto conv = m1.add_instruction(
            migraphx::make_op("convolution",
                              {{"padding", {1, 1}}, {"stride", {2, 2}}, {"dilation", {1, 1}}}),
            x,
            w);
        auto relu = m1.add_instruction(migraphx::make_op("relu"), conv);
        auto reshape1 =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {1, 1024}}}), relu);
        auto reshape2 =
            m1.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 512}}}), relu);
        m1.add_return({reshape1, reshape2});
    }
    migraphx::module m2 = m1;
    run_pass(m1, {.order = migraphx::layout_convolution::channels_auto});
    EXPECT(m1.sort() == m2.sort());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
