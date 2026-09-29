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
#include <migraphx/op/pad.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/normalize_ops.hpp>
#include <migraphx/eliminate_pad.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/instruction.hpp>
#include <basic_ops.hpp>
#include <migraphx/op/common.hpp>
#include <migraphx/make_op.hpp>

#include <test.hpp>

static void run_pass(migraphx::module& m)
{
    migraphx::run_passes(
        m,
        {migraphx::normalize_ops{}, migraphx::eliminate_pad{}, migraphx::dead_code_elimination{}});
}

static migraphx::instruction_ref create_im2col(migraphx::instruction_ref l_img,
                                               size_t channels,
                                               migraphx::module& m,
                                               const std::vector<size_t>& padding = {0, 0})
{
    size_t f[2] = {1, 1};
    std::vector<int32_t> weights(channels * f[0] * f[1]);
    migraphx::shape s_weights{migraphx::shape::int32_type, {1, channels, f[0], f[1]}};
    auto l_weights = m.add_literal(migraphx::literal{s_weights, weights});
    return m.add_instruction(migraphx::make_op("im2col", {{"padding", padding}}), l_img, l_weights);
}

static migraphx::instruction_ref
create_conv(migraphx::instruction_ref l_img,
            size_t channels,
            migraphx::module& m,
            const std::vector<size_t>& padding        = {0, 0},
            migraphx::op::padding_mode_t padding_mode = migraphx::op::padding_mode_t::default_)
{
    migraphx::shape s_weights{migraphx::shape::int32_type, {4, channels, 3, 3}};
    std::vector<int32_t> weights(4 * channels * 3 * 3);
    auto l_weights = m.add_literal(migraphx::literal{s_weights, weights});
    return m.add_instruction(
        migraphx::make_op("convolution", {{"padding", padding}, {"padding_mode", padding_mode}}),
        l_img,
        l_weights);
}

static migraphx::instruction_ref create_max_pooling(migraphx::instruction_ref l_img,
                                                    migraphx::module& m,
                                                    const std::vector<size_t>& padding = {0, 0})
{
    return m.add_instruction(
        migraphx::make_op("pooling",
                          {{"mode", migraphx::op::pooling_mode::max}, {"padding", padding}}),
        l_img);
}

TEST_CASE(eliminate_pad)
{
    migraphx::module m;
    size_t img_dim[2] = {2, 2};
    size_t channels   = 1;
    std::vector<int32_t> input(channels * img_dim[0] * img_dim[1]);
    std::iota(input.begin(), input.end(), 0);

    migraphx::shape s_img{migraphx::shape::int32_type, {1, channels, img_dim[0], img_dim[1]}};
    auto l_img = m.add_literal(migraphx::literal{s_img, input});
    auto padded_img =
        m.add_instruction(migraphx::make_op("pad", {{"pads", {0, 0, 1, 1, 0, 0, 1, 1}}}), l_img);

    auto l0 = create_im2col(padded_img, channels, m);
    auto l1 = create_conv(padded_img, channels, m);
    auto l2 = m.add_instruction(
        migraphx::make_op("pooling", {{"mode", migraphx::op::pooling_mode::max}}), padded_img);
    m.add_instruction(migraphx::make_op("identity"), l0, l1, l2);

    auto s0 = l0->get_shape();
    auto s1 = l1->get_shape();
    auto s2 = l2->get_shape();
    run_pass(m);
    EXPECT(l0->get_shape() == s0);
    EXPECT(l1->get_shape() == s1);
    EXPECT(l2->get_shape() == s2);
    auto op0 = l0->get_operator().to_value();
    auto om1 = l1->get_operator().to_value();
    auto om2 = l2->get_operator().to_value();

    EXPECT(op0["padding"].to_vector<std::size_t>() == std::vector<std::size_t>{1, 1, 1, 1});
    EXPECT(om1["padding"].to_vector<std::size_t>() == std::vector<std::size_t>{1, 1, 1, 1});
    EXPECT(om2["padding"].to_vector<std::size_t>() == std::vector<std::size_t>{1, 1, 1, 1});

    EXPECT(std::none_of(
        m.begin(), m.end(), [](const migraphx::instruction& ins) { return ins.name() == "pad"; }));
}

TEST_CASE(eliminate_pad_im2col_asymmetric)
{
    migraphx::module m;

    size_t img_dim[2] = {2, 2};
    size_t channels   = 1;
    std::vector<int32_t> input(channels * img_dim[0] * img_dim[1]);
    std::iota(input.begin(), input.end(), 0);

    migraphx::shape s_img{migraphx::shape::int32_type, {1, channels, img_dim[0], img_dim[1]}};
    auto l_img = m.add_literal(migraphx::literal{s_img, input});
    auto padded_img =
        m.add_instruction(migraphx::make_op("pad", {{"pads", {0, 0, 0, 0, 0, 0, 2, 2}}}), l_img);

    auto l0 = create_im2col(padded_img, channels, m);

    auto s0 = l0->get_shape();
    run_pass(m);
    EXPECT(l0->get_shape() == s0);
    auto op0 = l0->get_operator().to_value();

    EXPECT(op0["padding"].to_vector<std::size_t>() == std::vector<std::size_t>{0, 0, 2, 2});

    run_pass(m);
    EXPECT(std::none_of(
        m.begin(), m.end(), [](const migraphx::instruction& ins) { return ins.name() == "pad"; }));
}

TEST_CASE(eliminate_pad_nonzero_pad)
{
    migraphx::module m1;

    size_t img_dim[2] = {2, 2};
    size_t channels   = 1;
    std::vector<int32_t> input(channels * img_dim[0] * img_dim[1]);
    std::iota(input.begin(), input.end(), 0);
    migraphx::shape s_img{migraphx::shape::int32_type, {1, channels, img_dim[0], img_dim[1]}};
    auto l_img = m1.add_literal(migraphx::literal{s_img, input});
    auto padded_img =
        m1.add_instruction(migraphx::make_op("pad",
                                             {{"pads", {0, 0, 1, 1, 0, 0, 1, 1}},
                                              {"mode", migraphx::op::pad::constant_pad},
                                              {"value", 5.5f}}),
                           l_img);

    auto im2col  = create_im2col(padded_img, channels, m1);
    auto conv    = create_conv(padded_img, channels, m1);
    auto pooling = m1.add_instruction(
        migraphx::make_op("pooling", {{"mode", migraphx::op::pooling_mode::max}}), padded_img);
    m1.add_return({im2col, conv, pooling});

    migraphx::module m2 = m1;
    run_pass(m1);
    migraphx::run_passes(m2, {migraphx::normalize_ops{}});

    EXPECT(m1 == m2);
}

TEST_CASE(eliminate_pad_skip_reflect)
{
    migraphx::module m1;

    size_t img_dim[2] = {2, 2};
    size_t channels   = 1;
    std::vector<int32_t> input(channels * img_dim[0] * img_dim[1]);
    std::iota(input.begin(), input.end(), 0);
    migraphx::shape s_img{migraphx::shape::int32_type, {1, channels, img_dim[0], img_dim[1]}};
    auto l_img      = m1.add_literal(migraphx::literal{s_img, input});
    auto padded_img = m1.add_instruction(
        migraphx::make_op(
            "pad", {{"pads", {0, 0, 1, 1, 0, 0, 1, 1}}, {"mode", migraphx::op::pad::reflect_pad}}),
        l_img);

    auto im2col  = create_im2col(padded_img, channels, m1);
    auto conv    = create_conv(padded_img, channels, m1);
    auto pooling = m1.add_instruction(
        migraphx::make_op("pooling", {{"mode", migraphx::op::pooling_mode::max}}), padded_img);
    m1.add_return({im2col, conv, pooling});

    migraphx::module m2 = m1;
    run_pass(m1);
    migraphx::run_passes(m2, {migraphx::normalize_ops{}});

    EXPECT(m1 == m2);
}

TEST_CASE(eliminate_pad_skip_edge)
{
    migraphx::module m1;

    size_t img_dim[2] = {2, 2};
    size_t channels   = 1;
    std::vector<int32_t> input(channels * img_dim[0] * img_dim[1]);
    std::iota(input.begin(), input.end(), 0);
    migraphx::shape s_img{migraphx::shape::int32_type, {1, channels, img_dim[0], img_dim[1]}};
    auto l_img      = m1.add_literal(migraphx::literal{s_img, input});
    auto padded_img = m1.add_instruction(
        migraphx::make_op(
            "pad", {{"pads", {0, 0, 1, 1, 0, 0, 1, 1}}, {"mode", migraphx::op::pad::edge_pad}}),
        l_img);

    auto im2col  = create_im2col(padded_img, channels, m1);
    auto conv    = create_conv(padded_img, channels, m1);
    auto pooling = m1.add_instruction(
        migraphx::make_op("pooling", {{"mode", migraphx::op::pooling_mode::max}}), padded_img);
    m1.add_return({im2col, conv, pooling});

    migraphx::module m2 = m1;
    run_pass(m1);
    migraphx::run_passes(m2, {migraphx::normalize_ops{}});

    EXPECT(m1 == m2);
}

TEST_CASE(eliminate_pad_asymmetric)
{
    migraphx::shape s_img{migraphx::shape::int32_type, {1, 1, 2, 2}};
    migraphx::module m1;
    {
        auto l_img      = m1.add_parameter("img", s_img);
        auto padded_img = m1.add_instruction(
            migraphx::make_op("pad", {{"pads", {0, 0, 1, 0, 0, 0, 0, 2}}}), l_img);
        auto im2col  = create_im2col(padded_img, 1, m1);
        auto conv    = create_conv(padded_img, 1, m1);
        auto pooling = create_max_pooling(padded_img, m1);
        m1.add_return({im2col, conv, pooling});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto l_img   = m2.add_parameter("img", s_img);
        auto im2col  = create_im2col(l_img, 1, m2, {1, 0, 0, 2});
        auto conv    = create_conv(l_img, 1, m2, {1, 0, 0, 2});
        auto pooling = create_max_pooling(l_img, m2, {1, 0, 0, 2});
        m2.add_return({im2col, conv, pooling});
    }
    migraphx::run_passes(m2, {migraphx::normalize_ops{}});

    EXPECT(m1 == m2);
}

TEST_CASE(eliminate_pad_asymmetric_disabled)
{
    migraphx::shape s_img{migraphx::shape::int32_type, {1, 1, 2, 2}};
    migraphx::module m1;
    {
        auto l_img   = m1.add_parameter("img", s_img);
        auto sym_pad = m1.add_instruction(
            migraphx::make_op("pad", {{"pads", {0, 0, 1, 1, 0, 0, 1, 1}}}), l_img);
        auto asym_pad = m1.add_instruction(
            migraphx::make_op("pad", {{"pads", {0, 0, 1, 0, 0, 0, 0, 2}}}), l_img);
        auto conv1 = create_conv(sym_pad, 1, m1);
        auto conv2 = create_conv(asym_pad, 1, m1);
        m1.add_return({conv1, conv2});
    }
    migraphx::run_passes(m1,
                         {migraphx::normalize_ops{},
                          migraphx::eliminate_pad{.asym_pad = false},
                          migraphx::dead_code_elimination{}});

    migraphx::module m2;
    {
        auto l_img    = m2.add_parameter("img", s_img);
        auto asym_pad = m2.add_instruction(
            migraphx::make_op("pad", {{"pads", {0, 0, 1, 0, 0, 0, 0, 2}}}), l_img);
        auto conv1 = create_conv(l_img, 1, m2, {1, 1, 1, 1});
        auto conv2 = create_conv(asym_pad, 1, m2);
        m2.add_return({conv1, conv2});
    }
    migraphx::run_passes(m2, {migraphx::normalize_ops{}});

    EXPECT(m1 == m2);
}

TEST_CASE(eliminate_pad_existing_padding)
{
    migraphx::shape s_img{migraphx::shape::int32_type, {1, 1, 2, 2}};
    migraphx::module m1;
    {
        auto l_img      = m1.add_parameter("img", s_img);
        auto padded_img = m1.add_instruction(
            migraphx::make_op("pad", {{"pads", {0, 0, 1, 0, 0, 0, 0, 1}}}), l_img);
        auto im2col  = create_im2col(padded_img, 1, m1, {1, 1});
        auto conv    = create_conv(padded_img, 1, m1, {1, 1});
        auto pooling = create_max_pooling(padded_img, m1, {1, 1});
        m1.add_return({im2col, conv, pooling});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto l_img   = m2.add_parameter("img", s_img);
        auto im2col  = create_im2col(l_img, 1, m2, {2, 1, 1, 2});
        auto conv    = create_conv(l_img, 1, m2, {2, 1, 1, 2});
        auto pooling = create_max_pooling(l_img, m2, {2, 1, 1, 2});
        m2.add_return({im2col, conv, pooling});
    }
    migraphx::run_passes(m2, {migraphx::normalize_ops{}});

    EXPECT(m1 == m2);
}

TEST_CASE(eliminate_pad_dynamic)
{
    migraphx::shape s_img{migraphx::shape::int32_type, {{1, 4}, {1, 1}, {2, 2}, {2, 2}}};
    migraphx::module m1;
    {
        auto l_img      = m1.add_parameter("img", s_img);
        auto padded_img = m1.add_instruction(
            migraphx::make_op("pad", {{"pads", {0, 0, 1, 0, 0, 0, 0, 2}}}), l_img);
        auto conv = create_conv(padded_img, 1, m1);
        m1.add_return({conv});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto l_img = m2.add_parameter("img", s_img);
        auto conv  = create_conv(l_img, 1, m2, {1, 0, 0, 2});
        m2.add_return({conv});
    }
    migraphx::run_passes(m2, {migraphx::normalize_ops{}});

    EXPECT(m1 == m2);
}

TEST_CASE(eliminate_pad_skip_nonspatial)
{
    migraphx::module m1;
    {
        auto l_img      = m1.add_parameter("img", {migraphx::shape::int32_type, {1, 1, 2, 2}});
        auto padded_img = m1.add_instruction(
            migraphx::make_op("pad", {{"pads", {0, 1, 1, 1, 0, 1, 1, 1}}}), l_img);
        auto im2col  = create_im2col(padded_img, 3, m1);
        auto conv    = create_conv(padded_img, 3, m1);
        auto pooling = create_max_pooling(padded_img, m1);
        m1.add_return({im2col, conv, pooling});
    }

    migraphx::module m2 = m1;
    run_pass(m1);
    migraphx::run_passes(m2, {migraphx::normalize_ops{}});

    EXPECT(m1 == m2);
}

TEST_CASE(eliminate_pad_auto_padding_static)
{
    migraphx::shape s_img{migraphx::shape::int32_type, {1, 1, 2, 2}};
    migraphx::module m1;
    {
        auto l_img      = m1.add_parameter("img", s_img);
        auto padded_img = m1.add_instruction(
            migraphx::make_op("pad", {{"pads", {0, 0, 1, 0, 0, 0, 0, 2}}}), l_img);
        auto conv =
            create_conv(padded_img, 1, m1, {0, 0}, migraphx::op::padding_mode_t::same_upper);
        auto pooling = m1.add_instruction(
            migraphx::make_op("pooling",
                              {{"mode", migraphx::op::pooling_mode::max},
                               {"padding_mode", migraphx::op::padding_mode_t::same_upper}}),
            padded_img);
        m1.add_return({conv, pooling});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto l_img   = m2.add_parameter("img", s_img);
        auto conv    = create_conv(l_img, 1, m2, {1, 0, 0, 2});
        auto pooling = create_max_pooling(l_img, m2, {1, 0, 0, 2});
        m2.add_return({conv, pooling});
    }
    migraphx::run_passes(m2, {migraphx::normalize_ops{}});

    EXPECT(m1 == m2);
}

TEST_CASE(eliminate_pad_skip_auto_padding_dynamic)
{
    migraphx::shape s_img{migraphx::shape::int32_type, {{1, 1}, {1, 1}, {2, 4}, {2, 4}}};
    migraphx::module m1;
    {
        auto l_img      = m1.add_parameter("img", s_img);
        auto padded_img = m1.add_instruction(
            migraphx::make_op("pad", {{"pads", {0, 0, 1, 1, 0, 0, 1, 1}}}), l_img);
        auto conv =
            create_conv(padded_img, 1, m1, {0, 0}, migraphx::op::padding_mode_t::same_upper);
        auto pooling = m1.add_instruction(
            migraphx::make_op("pooling",
                              {{"mode", migraphx::op::pooling_mode::max},
                               {"padding_mode", migraphx::op::padding_mode_t::same_upper}}),
            padded_img);
        m1.add_return({conv, pooling});
    }

    migraphx::module m2 = m1;
    run_pass(m1);
    migraphx::run_passes(m2, {migraphx::normalize_ops{}});

    EXPECT(m1 == m2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
