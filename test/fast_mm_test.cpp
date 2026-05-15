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
#include <migraphx/fast_mm.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/module.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/pass_manager.hpp>

#include <test.hpp>

static void run_pass(migraphx::module& m)
{
    migraphx::run_passes(m, {migraphx::fast_mm{}, migraphx::dead_code_elimination{}});
}

TEST_CASE(fp32_convolution_const_weights_rewritten)
{
    migraphx::shape xs{migraphx::shape::float_type, {1, 3, 8, 8}};
    migraphx::shape ws{migraphx::shape::float_type, {4, 3, 3, 3}};
    std::vector<float> w_data(ws.elements(), 0.5f);
    std::vector<std::size_t> out_lens      = {1, 4, 6, 6};
    std::vector<std::size_t> scale_lens    = {4, 1, 1, 1};
    std::vector<std::int64_t> reduce_axes  = {1, 2, 3};
    std::vector<std::int64_t> reshape_dims = {1, 4, 1, 1};

    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", xs);
        auto w    = m1.add_literal(migraphx::literal{ws, w_data});
        auto conv = m1.add_instruction(migraphx::make_op("convolution"), x, w);
        m1.add_return({conv});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", xs);
        auto w = m2.add_literal(migraphx::literal{ws, w_data});

        auto abs_w     = m2.add_instruction(migraphx::make_op("abs"), w);
        auto scale_max = m2.add_instruction(
            migraphx::make_op("reduce_max", {{"axes", reduce_axes}}), abs_w);

        auto eps = m2.add_literal(
            migraphx::literal{migraphx::shape{migraphx::shape::float_type}, {1e-30f}});
        auto eps_bc = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", scale_lens}}), eps);
        auto scale = m2.add_instruction(migraphx::make_op("max"), scale_max, eps_bc);

        auto scale_w_bc = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", ws.lens()}}), scale);
        auto w_scaled = m2.add_instruction(migraphx::make_op("div"), w, scale_w_bc);

        auto xh = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), x);
        auto wh = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), w_scaled);

        auto conv      = m2.add_instruction(migraphx::make_op("convolution"), xh, wh);
        auto converted = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), conv);

        auto scale_r = m2.add_instruction(
            migraphx::make_op("reshape", {{"dims", reshape_dims}}), scale);
        auto scale_out_bc = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", out_lens}}), scale_r);
        auto result = m2.add_instruction(migraphx::make_op("mul"), converted, scale_out_bc);
        m2.add_return({result});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(fp32_convolution_param_weights_unchanged)
{
    migraphx::shape xs{migraphx::shape::float_type, {1, 3, 8, 8}};
    migraphx::shape ws{migraphx::shape::float_type, {4, 3, 3, 3}};

    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", xs);
        auto w    = m1.add_parameter("w", ws);
        auto conv = m1.add_instruction(migraphx::make_op("convolution"), x, w);
        m1.add_return({conv});
    }
    auto m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(fp16_convolution_unchanged)
{
    migraphx::shape xs{migraphx::shape::half_type, {1, 3, 8, 8}};
    migraphx::shape ws{migraphx::shape::half_type, {4, 3, 3, 3}};

    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", xs);
        auto w    = m1.add_parameter("w", ws);
        auto conv = m1.add_instruction(migraphx::make_op("convolution"), x, w);
        m1.add_return({conv});
    }
    auto m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(non_convolution_unchanged)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};

    migraphx::module m1;
    {
        auto x   = m1.add_parameter("x", s);
        auto y   = m1.add_parameter("y", s);
        auto add = m1.add_instruction(migraphx::make_op("add"), x, y);
        m1.add_return({add});
    }
    auto m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
