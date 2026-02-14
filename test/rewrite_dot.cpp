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
 *
 */

#include <migraphx/rewrite_dot.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/module.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <test.hpp>

static void run_pass(migraphx::module& m)
{
    migraphx::run_passes(m, {migraphx::rewrite_dot{}, migraphx::dead_code_elimination{}});
}

TEST_CASE(nchw_conv_1x1)
{
    migraphx::shape s1{migraphx::shape::float_type, {64, 128, 28, 28}};
    migraphx::shape s2{migraphx::shape::float_type, {512, 128, 1, 1}};
    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", s1);
        auto w    = m1.add_literal(migraphx::generate_literal(s2));
        auto conv = m1.add_instruction(migraphx::make_op("convolution"), x, w);
        m1.add_return({conv});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x         = m2.add_parameter("x", s1);
        auto w         = m2.add_literal(migraphx::generate_literal(s2));
        auto squeeze   = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {2, 3}}}), w);
        auto broadcast = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {64, 512, 128}}}), squeeze);
        auto reshape1 =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {64, 128, 784}}}), x);
        auto dot = m2.add_instruction(migraphx::make_op("dot"), broadcast, reshape1);
        auto reshape2 =
            m2.add_instruction(migraphx::make_op("reshape", {{"dims", {64, 512, 28, 28}}}), dot);
        m2.add_return({reshape2});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(nhwc_conv_1x1)
{
    auto s1 = migraphx::shape::from_permutation(
        migraphx::shape::float_type, {64, 128, 28, 28}, {0, 2, 3, 1});
    auto s2 = migraphx::shape::from_permutation(
        migraphx::shape::float_type, {512, 128, 1, 1}, {0, 2, 3, 1});
    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", s1);
        auto w    = m1.add_literal(migraphx::generate_literal(s2));
        auto conv = m1.add_instruction(migraphx::make_op("convolution"), x, w);
        m1.add_return({conv});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x       = m2.add_parameter("x", s1);
        auto w       = m2.add_literal(migraphx::generate_literal(s2));
        auto squeeze = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {2, 3}}}), w);
        auto transpose1 =
            m2.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), squeeze);
        auto broadcast = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {64, 28, 128, 512}}}), transpose1);
        auto transpose2 =
            m2.add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), x);
        auto dot        = m2.add_instruction(migraphx::make_op("dot"), transpose2, broadcast);
        auto transpose3 = m2.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 3, 1, 2}}}), dot);
        m2.add_return({transpose3});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(nhwc_group_conv_1x1)
{
    auto s1 = migraphx::shape::from_permutation(
        migraphx::shape::float_type, {64, 192, 83, 83}, {0, 2, 3, 1});
    auto s2 = migraphx::shape::from_permutation(
        migraphx::shape::float_type, {84, 96, 1, 1}, {0, 2, 3, 1});
    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", s1);
        auto w    = m1.add_literal(migraphx::generate_literal(s2));
        auto conv = m1.add_instruction(migraphx::make_op("convolution", {{"group", 2}}), x, w);
        m1.add_return({conv});
    }
    migraphx::module m2 = m1;
    run_pass(m1);
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(nchw_depthwise_conv_1x1)
{
    migraphx::shape s1{migraphx::shape::float_type, {2, 4, 3, 3}};
    migraphx::shape s2{migraphx::shape::float_type, {4, 1, 1, 1}};
    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", s1);
        auto w    = m1.add_literal(migraphx::generate_literal(s2));
        auto conv = m1.add_instruction(migraphx::make_op("convolution", {{"group", 4}}), x, w);
        m1.add_return({conv});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x         = m2.add_parameter("x", s1);
        auto w         = m2.add_literal(migraphx::generate_literal(s2));
        auto squeeze   = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {1, 2, 3}}}), w);
        auto broadcast = m2.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {2, 4, 3, 3}}}), squeeze);
        auto mul = m2.add_instruction(migraphx::make_op("mul"), x, broadcast);
        m2.add_return({mul});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(nchw_depthwise_conv_1x1_non_constant)
{
    migraphx::shape s1{migraphx::shape::float_type, {2, 4, 3, 3}};
    migraphx::shape s2{migraphx::shape::float_type, {4, 1, 1, 1}};
    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", s1);
        auto w    = m1.add_parameter("w", s2);
        auto conv = m1.add_instruction(migraphx::make_op("convolution", {{"group", 4}}), x, w);
        m1.add_return({conv});
    }
    migraphx::module m2 = m1;
    run_pass(m1);
    EXPECT(m1.sort() == m2.sort());
}

// Helper to build the expected broadcast+slice module for channelwise convolution
static migraphx::instruction_ref
build_channelwise_expected(migraphx::module& m2,
                           migraphx::instruction_ref x,
                           migraphx::instruction_ref w,
                           const std::vector<std::size_t>& out_lens,
                           const std::vector<std::size_t>& /*x_lens*/,
                           const std::vector<std::size_t>& w_lens,
                           const std::vector<std::size_t>& prod_lens,
                           std::size_t num_spatial)
{
    // Unsqueeze input
    std::vector<int64_t> input_unsq_axes(num_spatial);
    std::iota(input_unsq_axes.begin(), input_unsq_axes.end(), 2);
    auto unsq_x =
        m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", input_unsq_axes}}), x);

    // Squeeze weight axis 1, then unsqueeze for product shape
    auto sq_w = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {1}}}), w);
    std::vector<int64_t> w_unsq_axes;
    w_unsq_axes.push_back(0);
    for(std::size_t d = 0; d < num_spatial; ++d)
        w_unsq_axes.push_back(static_cast<int64_t>(2 + num_spatial + d));
    auto unsq_w = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", w_unsq_axes}}), sq_w);

    // Broadcast both to product shape
    auto bcast_x =
        m2.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", prod_lens}}), unsq_x);
    auto bcast_w =
        m2.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", prod_lens}}), unsq_w);

    // Multiply
    auto product = m2.add_instruction(migraphx::make_op("mul"), bcast_x, bcast_w);

    // Reduce each spatial dimension in reverse
    auto current = product;
    for(int d = static_cast<int>(num_spatial) - 1; d >= 0; --d)
    {
        auto kernel_axis  = static_cast<int64_t>(2 + d);
        auto spatial_axis = static_cast<int64_t>(3 + 2 * d);
        auto kernel_size  = w_lens[2 + d];
        auto out_size     = out_lens[2 + d];

        migraphx::instruction_ref accum;
        for(std::size_t ki = 0; ki < kernel_size; ++ki)
        {
            auto ki_start = static_cast<int64_t>(ki);
            auto sliced   = m2.add_instruction(
                migraphx::make_op(
                    "slice",
                    {{"axes", {kernel_axis, spatial_axis}},
                       {"starts", {ki_start, ki_start}},
                       {"ends", {ki_start + 1, ki_start + static_cast<int64_t>(out_size)}}}),
                current);
            auto squeezed =
                m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {kernel_axis}}}), sliced);

            if(ki == 0)
                accum = squeezed;
            else
                accum = m2.add_instruction(migraphx::make_op("add"), accum, squeezed);
        }
        current = accum;
    }
    return current;
}

TEST_CASE(nchw_depthwise_conv_3x3)
{
    migraphx::shape s1{migraphx::shape::float_type, {2, 4, 5, 5}};
    migraphx::shape s2{migraphx::shape::float_type, {4, 1, 3, 3}};
    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", s1);
        auto w    = m1.add_literal(migraphx::generate_literal(s2));
        auto conv = m1.add_instruction(migraphx::make_op("convolution", {{"group", 4}}), x, w);
        m1.add_return({conv});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x      = m2.add_parameter("x", s1);
        auto w      = m2.add_literal(migraphx::generate_literal(s2));
        auto result = build_channelwise_expected(
            m2, x, w, {2, 4, 3, 3}, {2, 4, 5, 5}, {4, 1, 3, 3}, {2, 4, 3, 3, 5, 5}, 2);
        m2.add_return({result});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(nchw_depthwise_conv_1x3)
{
    migraphx::shape s1{migraphx::shape::float_type, {1, 8, 4, 6}};
    migraphx::shape s2{migraphx::shape::float_type, {8, 1, 1, 3}};
    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", s1);
        auto w    = m1.add_literal(migraphx::generate_literal(s2));
        auto conv = m1.add_instruction(migraphx::make_op("convolution", {{"group", 8}}), x, w);
        m1.add_return({conv});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x      = m2.add_parameter("x", s1);
        auto w      = m2.add_literal(migraphx::generate_literal(s2));
        auto result = build_channelwise_expected(
            m2, x, w, {1, 8, 4, 4}, {1, 8, 4, 6}, {8, 1, 1, 3}, {1, 8, 1, 3, 4, 6}, 2);
        m2.add_return({result});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(nchw_depthwise_conv_3x3_non_constant)
{
    migraphx::shape s1{migraphx::shape::float_type, {2, 4, 5, 5}};
    migraphx::shape s2{migraphx::shape::float_type, {4, 1, 3, 3}};
    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", s1);
        auto w    = m1.add_parameter("w", s2);
        auto conv = m1.add_instruction(migraphx::make_op("convolution", {{"group", 4}}), x, w);
        m1.add_return({conv});
    }
    migraphx::module m2 = m1;
    run_pass(m1);
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(nchw_depthwise_conv_3x3_stride)
{
    migraphx::shape s1{migraphx::shape::float_type, {2, 4, 7, 7}};
    migraphx::shape s2{migraphx::shape::float_type, {4, 1, 3, 3}};
    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", s1);
        auto w    = m1.add_literal(migraphx::generate_literal(s2));
        auto conv = m1.add_instruction(
            migraphx::make_op("convolution", {{"group", 4}, {"stride", {2, 2}}}), x, w);
        m1.add_return({conv});
    }
    migraphx::module m2 = m1;
    run_pass(m1);
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(nchw_depthwise_conv_1x1_multiplier)
{
    migraphx::shape s1{migraphx::shape::float_type, {2, 4, 3, 3}};
    migraphx::shape s2{migraphx::shape::float_type, {8, 1, 1, 1}};
    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", s1);
        auto w    = m1.add_literal(migraphx::generate_literal(s2));
        auto conv = m1.add_instruction(migraphx::make_op("convolution", {{"group", 4}}), x, w);
        m1.add_return({conv});
    }
    migraphx::module m2 = m1;
    run_pass(m1);
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(nchw_conv_c1_1x1)
{
    migraphx::shape s1{migraphx::shape::float_type, {2, 1, 4, 4}};
    migraphx::shape s2{migraphx::shape::float_type, {3, 1, 1, 1}};
    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", s1);
        auto w    = m1.add_literal(migraphx::generate_literal(s2));
        auto conv = m1.add_instruction(migraphx::make_op("convolution"), x, w);
        m1.add_return({conv});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x       = m2.add_parameter("x", s1);
        auto w       = m2.add_literal(migraphx::generate_literal(s2));
        auto squeeze = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {1, 2, 3}}}), w);
        auto bcast_w = m2.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {2, 3, 4, 4}}}), squeeze);
        auto bcast_x = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 3, 4, 4}}}), x);
        auto mul = m2.add_instruction(migraphx::make_op("mul"), bcast_x, bcast_w);
        m2.add_return({mul});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(nchw_conv_c1_3x3)
{
    migraphx::shape s1{migraphx::shape::float_type, {2, 1, 5, 5}};
    migraphx::shape s2{migraphx::shape::float_type, {3, 1, 3, 3}};
    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", s1);
        auto w    = m1.add_literal(migraphx::generate_literal(s2));
        auto conv = m1.add_instruction(migraphx::make_op("convolution"), x, w);
        m1.add_return({conv});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x      = m2.add_parameter("x", s1);
        auto w      = m2.add_literal(migraphx::generate_literal(s2));
        auto result = build_channelwise_expected(
            m2, x, w, {2, 3, 3, 3}, {2, 1, 5, 5}, {3, 1, 3, 3}, {2, 3, 3, 3, 5, 5}, 2);
        m2.add_return({result});
    }
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(nchw_conv_c1_3x3_non_constant)
{
    migraphx::shape s1{migraphx::shape::float_type, {2, 1, 5, 5}};
    migraphx::shape s2{migraphx::shape::float_type, {3, 1, 3, 3}};
    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", s1);
        auto w    = m1.add_parameter("w", s2);
        auto conv = m1.add_instruction(migraphx::make_op("convolution"), x, w);
        m1.add_return({conv});
    }
    migraphx::module m2 = m1;
    run_pass(m1);
    EXPECT(m1.sort() == m2.sort());
}

TEST_CASE(nchw_conv_c1_3x3_stride)
{
    migraphx::shape s1{migraphx::shape::float_type, {2, 1, 7, 7}};
    migraphx::shape s2{migraphx::shape::float_type, {3, 1, 3, 3}};
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", s1);
        auto w = m1.add_literal(migraphx::generate_literal(s2));
        auto conv =
            m1.add_instruction(migraphx::make_op("convolution", {{"stride", {2, 2}}}), x, w);
        m1.add_return({conv});
    }
    migraphx::module m2 = m1;
    run_pass(m1);
    EXPECT(m1.sort() == m2.sort());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
