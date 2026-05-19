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

static void run_pass(migraphx::module& m, migraphx::fast_mm fmm = {})
{
    migraphx::run_passes(m, {fmm, migraphx::dead_code_elimination{}});
}

TEST_CASE(fp32_convolution_const_weights_rewritten)
{
    migraphx::shape xs{migraphx::shape::float_type, {1, 3, 8, 8}};
    migraphx::shape ws{migraphx::shape::float_type, {4, 3, 3, 3}};
    std::vector<float> w_data(ws.elements(), 0.5f);

    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", xs);
        auto w    = m1.add_literal(migraphx::literal{ws, w_data});
        auto conv = m1.add_instruction(migraphx::make_op("convolution"), x, w);
        m1.add_return({conv});
    }
    run_pass(m1, {.skip_small_k = 0});

    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", xs);
        auto w = m2.add_literal(migraphx::literal{ws, w_data});

        auto w_hi_h = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), w);
        auto w_hi_f = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), w_hi_h);
        auto w_lo_f = m2.add_instruction(migraphx::make_op("sub"), w, w_hi_f);
        auto w_lo_h = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), w_lo_f);
        auto w_concat =
            m2.add_instruction(migraphx::make_op("concat", {{"axis", 1}}), w_hi_h, w_lo_h);

        auto x_h = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), x);
        auto x_unsq = m2.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), x_h);
        auto x_bc   = m2.add_instruction(
            migraphx::make_op("multibroadcast",
                                {{"out_lens", std::vector<std::size_t>{1, 2, 3, 8, 8}}}),
            x_unsq);
        auto x_doubled = m2.add_instruction(
            migraphx::make_op("reshape", {{"dims", std::vector<std::int64_t>{1, 6, 8, 8}}}), x_bc);

        auto conv = m2.add_instruction(migraphx::make_op("convolution"), x_doubled, w_concat);
        auto out  = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), conv);
        m2.add_return({out});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(fp32_convolution_tiny_unchanged)
{
    // 11 outputs * 8 reduction = 88 ops — too small to benefit from fp16
    // acceleration, and tiny conv outputs are precision-sensitive.
    migraphx::shape xs{migraphx::shape::float_type, {1, 8, 1, 1}};
    migraphx::shape ws{migraphx::shape::float_type, {11, 8, 1, 1}};
    std::vector<float> w_data(ws.elements(), 0.5f);

    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", xs);
        auto w    = m1.add_literal(migraphx::literal{ws, w_data});
        auto conv = m1.add_instruction(migraphx::make_op("convolution"), x, w);
        m1.add_return({conv});
    }
    auto m2 = m1;
    run_pass(m1);
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
