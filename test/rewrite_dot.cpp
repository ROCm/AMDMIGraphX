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
#include <migraphx/literal.hpp>
#include <migraphx/op/common.hpp>
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
        auto conv = m1.add_instruction(
            migraphx::make_op("convolution", {{"group", 4}}), x, w);
        m1.add_return({conv});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x       = m2.add_parameter("x", s1);
        auto w       = m2.add_literal(migraphx::generate_literal(s2));
        auto squeeze = m2.add_instruction(
            migraphx::make_op("squeeze", {{"axes", {1, 2, 3}}}), w);
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
        auto conv = m1.add_instruction(
            migraphx::make_op("convolution", {{"group", 4}}), x, w);
        m1.add_return({conv});
    }
    migraphx::module m2 = m1;
    run_pass(m1);
    EXPECT(m1.sort() == m2.sort());
}

// Helper to build the expected pooling-based module for channelwise convolution
static migraphx::instruction_ref
build_channelwise_expected(migraphx::module& m2,
                           migraphx::instruction_ref x,
                           migraphx::instruction_ref w,
                           const std::vector<std::size_t>& w_lens,
                           const std::vector<std::size_t>& x_lens,
                           std::size_t num_spatial)
{
    auto ndim = 2 + num_spatial;

    // Compute kernel_elements
    std::size_t kernel_elements = 1;
    for(std::size_t d = 2; d < ndim; ++d)
        kernel_elements *= w_lens[d];

    // Scale weights
    auto scale_lit = m2.add_literal(
        migraphx::literal{migraphx::shape{migraphx::shape::float_type, {1}},
                          {static_cast<double>(kernel_elements)}});
    auto scale_bcast = m2.add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", w_lens}}), scale_lit);
    auto scaled_w = m2.add_instruction(migraphx::make_op("mul"), w, scale_bcast);

    // Build interleaved product shape: [N, C_out, k0, s0, k1, s1, ...]
    std::vector<std::size_t> prod_lens;
    prod_lens.push_back(x_lens[0]);
    prod_lens.push_back(w_lens[0]);
    for(std::size_t d = 0; d < num_spatial; ++d)
    {
        prod_lens.push_back(w_lens[2 + d]);
        prod_lens.push_back(x_lens[2 + d]);
    }

    // Unsqueeze input: insert kernel singleton dims at 2, 4, 6, ...
    std::vector<int64_t> input_unsq_axes;
    for(std::size_t d = 0; d < num_spatial; ++d)
        input_unsq_axes.push_back(static_cast<int64_t>(2 + 2 * d));
    auto unsq_x =
        m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", input_unsq_axes}}), x);
    auto bcast_x = m2.add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", prod_lens}}), unsq_x);

    // Squeeze weight axis 1, then unsqueeze for interleaved layout
    auto sq_w = m2.add_instruction(migraphx::make_op("squeeze", {{"axes", {1}}}), scaled_w);
    std::vector<int64_t> w_unsq_axes;
    w_unsq_axes.push_back(0);
    for(std::size_t d = 0; d < num_spatial; ++d)
        w_unsq_axes.push_back(static_cast<int64_t>(3 + 2 * d));
    auto unsq_w =
        m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", w_unsq_axes}}), sq_w);
    auto bcast_w = m2.add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", prod_lens}}), unsq_w);

    // Multiply
    auto product = m2.add_instruction(migraphx::make_op("mul"), bcast_x, bcast_w);

    // Reshape to flatten paired dims: [N, C_out, k0*s0, k1*s1, ...]
    std::vector<int64_t> flat_dims;
    flat_dims.push_back(static_cast<int64_t>(x_lens[0]));
    flat_dims.push_back(static_cast<int64_t>(w_lens[0]));
    for(std::size_t d = 0; d < num_spatial; ++d)
        flat_dims.push_back(static_cast<int64_t>(w_lens[2 + d] * x_lens[2 + d]));
    auto reshaped =
        m2.add_instruction(migraphx::make_op("reshape", {{"dims", flat_dims}}), product);

    // Dilated average pooling
    std::vector<std::size_t> pool_lengths;
    std::vector<std::size_t> pool_dilations;
    std::vector<std::size_t> pool_stride;
    std::vector<std::size_t> pool_padding;
    for(std::size_t d = 0; d < num_spatial; ++d)
    {
        pool_lengths.push_back(w_lens[2 + d]);
        pool_dilations.push_back(x_lens[2 + d] + 1);
        pool_stride.push_back(1);
        pool_padding.push_back(0);
    }

    return m2.add_instruction(
        migraphx::make_op("pooling",
                           {{"mode", migraphx::op::pooling_mode::average},
                            {"lengths", pool_lengths},
                            {"dilations", pool_dilations},
                            {"stride", pool_stride},
                            {"padding", pool_padding},
                            {"count_include_pad", true}}),
        reshaped);
}

TEST_CASE(nchw_depthwise_conv_3x3)
{
    migraphx::shape s1{migraphx::shape::float_type, {2, 4, 5, 5}};
    migraphx::shape s2{migraphx::shape::float_type, {4, 1, 3, 3}};
    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", s1);
        auto w    = m1.add_literal(migraphx::generate_literal(s2));
        auto conv = m1.add_instruction(
            migraphx::make_op("convolution", {{"group", 4}}), x, w);
        m1.add_return({conv});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x      = m2.add_parameter("x", s1);
        auto w      = m2.add_literal(migraphx::generate_literal(s2));
        auto result = build_channelwise_expected(m2, x, w, {4, 1, 3, 3}, {2, 4, 5, 5}, 2);
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
        auto conv = m1.add_instruction(
            migraphx::make_op("convolution", {{"group", 8}}), x, w);
        m1.add_return({conv});
    }
    run_pass(m1);
    migraphx::module m2;
    {
        auto x      = m2.add_parameter("x", s1);
        auto w      = m2.add_literal(migraphx::generate_literal(s2));
        auto result = build_channelwise_expected(m2, x, w, {8, 1, 1, 3}, {1, 8, 4, 6}, 2);
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
        auto conv = m1.add_instruction(
            migraphx::make_op("convolution", {{"group", 4}}), x, w);
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
        auto conv = m1.add_instruction(
            migraphx::make_op("convolution", {{"group", 4}}), x, w);
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
        auto squeeze = m2.add_instruction(
            migraphx::make_op("squeeze", {{"axes", {1, 2, 3}}}), w);
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
        auto result = build_channelwise_expected(m2, x, w, {3, 1, 3, 3}, {2, 1, 5, 5}, 2);
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
