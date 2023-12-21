/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/simplify_qdq.hpp>
#include <migraphx/program.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/instruction.hpp>
#include <test.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/op/pooling.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/verify.hpp>
#include <migraphx/apply_alpha_beta.hpp>

bool is_convolution(const migraphx::instruction& ins) { return ins.name() == "convolution"; }
bool is_dot(const migraphx::instruction& ins) { return ins.name() == "dot"; }

void run_pass(migraphx::module& m)
{
    migraphx::simplify_qdq sqdq;
    sqdq.apply(m);
}

migraphx::instruction_ref broadcast_scale(migraphx::module& m,
                                          migraphx::instruction_ref scale,
                                          const std::vector<std::size_t>& out_lens,
                                          std::size_t axis)
{
    if(scale->get_shape().lens() == out_lens)
        return scale;

    migraphx::instruction_ref scale_mb;
    auto scale_lens = scale->get_shape().lens();
    if(scale_lens.front() == 1 and scale_lens.size() == 1)
        scale_mb =
            m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", out_lens}}), scale);
    else
        scale_mb = m.add_instruction(
            migraphx::make_op("broadcast", {{"axis", axis}, {"out_lens", out_lens}}), scale);
    return scale_mb;
}

migraphx::instruction_ref add_quantize_op(migraphx::module& m,
                                          const std::string& name,
                                          migraphx::instruction_ref x,
                                          migraphx::instruction_ref scale,
                                          migraphx::instruction_ref shift,
                                          std::size_t q_axis = 1)
{
    auto lens     = x->get_shape().lens();
    auto scale_mb = broadcast_scale(m, scale, lens, q_axis);
    auto shift_mb =
        m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", lens}}), shift);
    return m.add_instruction(migraphx::make_op(name), x, scale_mb, shift_mb);
}

migraphx::instruction_ref add_quantize_op(migraphx::module& m,
                                          const std::string& name,
                                          migraphx::instruction_ref x,
                                          migraphx::instruction_ref scale,
                                          std::size_t q_axis = 1)
{
    auto lens     = x->get_shape().lens();
    auto scale_mb = broadcast_scale(m, scale, lens, q_axis);
    return m.add_instruction(migraphx::make_op(name), x, scale_mb);
}

migraphx::instruction_ref add_scale_mul(migraphx::module& m,
                                        migraphx::instruction_ref scale1,
                                        migraphx::instruction_ref scale2,
                                        std::size_t axis1,
                                        std::size_t axis2,
                                        const std::vector<std::size_t>& out_lens)
{
    auto scale1_mb = broadcast_scale(m, scale1, out_lens, axis1);
    auto scale2_mb = broadcast_scale(m, scale2, out_lens, axis2);
    return m.add_instruction(migraphx::make_op("mul"), scale1_mb, scale2_mb);
}

TEST_CASE(remove_qdq)
{
    migraphx::shape sh1{migraphx::shape::float_type, {100, 100}};
    migraphx::shape sh2{migraphx::shape::float_type, {100, 100}};

    migraphx::module m1;
    {
        auto t1    = m1.add_parameter("t1", sh1);
        auto t2    = m1.add_parameter("t2", sh2);
        auto scale = m1.add_literal(0.5f);
        auto zero  = m1.add_literal(std::int8_t{0});

        auto q1  = add_quantize_op(m1, "quantizelinear", t1, scale, zero);
        auto d1  = add_quantize_op(m1, "dequantizelinear", q1, scale, zero);
        auto q2  = add_quantize_op(m1, "quantizelinear", t2, scale, zero);
        auto d2  = add_quantize_op(m1, "dequantizelinear", q2, scale, zero);
        auto add = m1.add_instruction(migraphx::make_op("add"), d1, d2);
        m1.add_return({add});
    }

    migraphx::module m2;
    {
        auto t1 = m2.add_parameter("t1", sh1);
        auto t2 = m2.add_parameter("t2", sh2);

        auto add = m2.add_instruction(migraphx::make_op("add"), t1, t2);
        m2.add_return({add});
    }

    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(qdq_different_scales)
{
    migraphx::shape sh1{migraphx::shape::float_type, {100, 100}};
    migraphx::shape sh2{migraphx::shape::float_type, {100, 100}};

    migraphx::module m1;
    {
        auto t1     = m1.add_parameter("t1", sh1);
        auto t2     = m1.add_parameter("t2", sh2);
        auto scale1 = m1.add_literal(0.5f);
        auto scale2 = m1.add_literal(0.4f);
        auto zero   = m1.add_literal(std::int8_t{0});

        auto q1  = add_quantize_op(m1, "quantizelinear", t1, scale1, zero);
        auto d1  = add_quantize_op(m1, "dequantizelinear", q1, scale2, zero);
        auto q2  = add_quantize_op(m1, "quantizelinear", t2, scale1, zero);
        auto d2  = add_quantize_op(m1, "dequantizelinear", q2, scale2, zero);
        auto add = m1.add_instruction(migraphx::make_op("add"), d1, d2);
        m1.add_return({add});
    }

    migraphx::module m2 = m1;

    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(dot)
{
    migraphx::shape sh1{migraphx::shape::float_type, {1280, 1000}};
    migraphx::shape sh2{migraphx::shape::float_type, {1000, 1024}};

    migraphx::module m1;
    {
        auto t1    = m1.add_parameter("t1", sh1);
        auto t2    = m1.add_parameter("t2", sh2);
        auto scale = m1.add_literal(0.5f);
        auto zero  = m1.add_literal(std::int8_t{0});

        auto q1  = add_quantize_op(m1, "quantizelinear", t1, scale, zero);
        auto d1  = add_quantize_op(m1, "dequantizelinear", q1, scale, zero);
        auto q2  = add_quantize_op(m1, "quantizelinear", t2, scale, zero);
        auto d2  = add_quantize_op(m1, "dequantizelinear", q2, scale, zero);
        auto dot = m1.add_instruction(migraphx::make_op("dot"), d1, d2);
        m1.add_return({dot});
    }

    migraphx::module m2;
    {
        auto t1    = m2.add_parameter("t1", sh1);
        auto t2    = m2.add_parameter("t2", sh2);
        auto scale = m2.add_literal(0.5f);
        auto zero  = m2.add_literal(std::int8_t{0});

        auto q1 = add_quantize_op(m2, "quantizelinear", t1, scale, zero);
        auto q2 = add_quantize_op(m2, "quantizelinear", t2, scale, zero);

        auto dot       = m2.add_instruction(migraphx::make_op("quant_dot"), q1, q2);
        auto out_scale = add_scale_mul(m2, scale, scale, 1, 1, dot->get_shape().lens());
        auto d3        = add_quantize_op(m2, "dequantizelinear", dot, out_scale);
        m2.add_return({d3});
    }

    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(dot_multi_scale)
{
    migraphx::shape sh1{migraphx::shape::float_type, {1280, 1000}};
    migraphx::shape sh2{migraphx::shape::float_type, {1000, 1024}};
    migraphx::shape sh3{migraphx::shape::float_type, {1280}};

    migraphx::module m1;
    {
        auto t1     = m1.add_parameter("t1", sh1);
        auto t2     = m1.add_parameter("t2", sh2);
        auto scale1 = m1.add_literal(migraphx::generate_literal(sh3, 0));
        auto scale2 = m1.add_literal(0.4f);
        auto zero   = m1.add_literal(std::int8_t{0});

        auto q1  = add_quantize_op(m1, "quantizelinear", t1, scale1, zero, 0);
        auto d1  = add_quantize_op(m1, "dequantizelinear", q1, scale1, zero, 0);
        auto q2  = add_quantize_op(m1, "quantizelinear", t2, scale2, zero, 1);
        auto d2  = add_quantize_op(m1, "dequantizelinear", q2, scale2, zero, 1);
        auto dot = m1.add_instruction(migraphx::make_op("dot"), d1, d2);
        m1.add_return({dot});
    }

    migraphx::module m2;
    {
        auto t1     = m2.add_parameter("t1", sh1);
        auto t2     = m2.add_parameter("t2", sh2);
        auto scale1 = m2.add_literal(migraphx::generate_literal(sh3, 0));
        auto scale2 = m2.add_literal(0.4f);
        auto zero   = m2.add_literal(std::int8_t{0});

        auto q1 = add_quantize_op(m2, "quantizelinear", t1, scale1, zero, 0);
        auto q2 = add_quantize_op(m2, "quantizelinear", t2, scale2, zero, 1);

        auto dot       = m2.add_instruction(migraphx::make_op("quant_dot"), q1, q2);
        auto out_scale = add_scale_mul(m2, scale1, scale2, 0, 1, dot->get_shape().lens());
        auto d3        = add_quantize_op(m2, "dequantizelinear", dot, out_scale);
        m2.add_return({d3});
    }

    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(dot_broadcasted)
{
    migraphx::shape sh1{migraphx::shape::float_type, {2, 1280, 1000}};
    migraphx::shape sh2{migraphx::shape::float_type, {1000, 1024}};

    migraphx::module m1;
    {
        auto t1    = m1.add_parameter("t1", sh1);
        auto t2    = m1.add_parameter("t2", sh2);
        auto scale = m1.add_literal(0.5f);
        auto zero  = m1.add_literal(std::int8_t{0});

        auto q1    = add_quantize_op(m1, "quantizelinear", t1, scale, zero);
        auto d1    = add_quantize_op(m1, "dequantizelinear", q1, scale, zero);
        auto q2    = add_quantize_op(m1, "quantizelinear", t2, scale, zero);
        auto d2    = add_quantize_op(m1, "dequantizelinear", q2, scale, zero);
        auto d2_mb = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 1000, 1024}}}), d2);
        auto dot = m1.add_instruction(migraphx::make_op("dot"), d1, d2_mb);
        m1.add_return({dot});
    }

    migraphx::module m2;
    {
        auto t1    = m2.add_parameter("t1", sh1);
        auto t2    = m2.add_parameter("t2", sh2);
        auto scale = m2.add_literal(0.5f);
        auto zero  = m2.add_literal(std::int8_t{0});

        auto q1    = add_quantize_op(m2, "quantizelinear", t1, scale, zero);
        auto q2    = add_quantize_op(m2, "quantizelinear", t2, scale, zero);
        auto q2_mb = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 1000, 1024}}}), q2);

        auto dot       = m2.add_instruction(migraphx::make_op("quant_dot"), q1, q2_mb);
        auto out_scale = add_scale_mul(m2, scale, scale, 1, 1, dot->get_shape().lens());
        auto d3        = add_quantize_op(m2, "dequantizelinear", dot, out_scale);
        m2.add_return({d3});
    }

    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(dot_transposed)
{
    migraphx::shape sh1{migraphx::shape::float_type, {1280, 1000}};
    migraphx::shape sh2{migraphx::shape::float_type, {1024, 1000}};

    migraphx::module m1;
    {
        auto t1    = m1.add_parameter("t1", sh1);
        auto t2    = m1.add_parameter("t2", sh2);
        auto scale = m1.add_literal(0.5f);
        auto zero  = m1.add_literal(std::int8_t{0});

        auto q1 = add_quantize_op(m1, "quantizelinear", t1, scale, zero);
        auto d1 = add_quantize_op(m1, "dequantizelinear", q1, scale, zero);
        auto q2 = add_quantize_op(m1, "quantizelinear", t2, scale, zero);
        auto d2 = add_quantize_op(m1, "dequantizelinear", q2, scale, zero);
        auto d2_t =
            m1.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), d2);
        auto dot = m1.add_instruction(migraphx::make_op("dot"), d1, d2_t);
        m1.add_return({dot});
    }

    migraphx::module m2;
    {
        auto t1    = m2.add_parameter("t1", sh1);
        auto t2    = m2.add_parameter("t2", sh2);
        auto scale = m2.add_literal(0.5f);
        auto zero  = m2.add_literal(std::int8_t{0});

        auto q1 = add_quantize_op(m2, "quantizelinear", t1, scale, zero);
        auto q2 = add_quantize_op(m2, "quantizelinear", t2, scale, zero);
        auto q2_t =
            m2.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), q2);

        auto dot       = m2.add_instruction(migraphx::make_op("quant_dot"), q1, q2_t);
        auto out_scale = add_scale_mul(m2, scale, scale, 1, 1, dot->get_shape().lens());
        auto d3        = add_quantize_op(m2, "dequantizelinear", dot, out_scale);
        m2.add_return({d3});
    }

    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(dot_reshaped)
{
    migraphx::shape sh1{migraphx::shape::float_type, {1280, 1000}};
    migraphx::shape sh2{migraphx::shape::float_type, {1024, 1000}};

    migraphx::module m1;
    {
        auto t1    = m1.add_parameter("t1", sh1);
        auto t2    = m1.add_parameter("t2", sh2);
        auto scale = m1.add_literal(0.5f);
        auto zero  = m1.add_literal(std::int8_t{0});

        auto q1   = add_quantize_op(m1, "quantizelinear", t1, scale, zero);
        auto d1   = add_quantize_op(m1, "dequantizelinear", q1, scale, zero);
        auto q2   = add_quantize_op(m1, "quantizelinear", t2, scale, zero);
        auto d2   = add_quantize_op(m1, "dequantizelinear", q2, scale, zero);
        auto d2_t = m1.add_instruction(migraphx::make_op("reshape", {{"dims", {1000, 1024}}}), d2);
        auto dot  = m1.add_instruction(migraphx::make_op("dot"), d1, d2_t);
        m1.add_return({dot});
    }

    migraphx::module m2;
    {
        auto t1    = m2.add_parameter("t1", sh1);
        auto t2    = m2.add_parameter("t2", sh2);
        auto scale = m2.add_literal(0.5f);
        auto zero  = m2.add_literal(std::int8_t{0});

        auto q1   = add_quantize_op(m2, "quantizelinear", t1, scale, zero);
        auto q2   = add_quantize_op(m2, "quantizelinear", t2, scale, zero);
        auto q2_t = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {1000, 1024}}}), q2);

        auto dot       = m2.add_instruction(migraphx::make_op("quant_dot"), q1, q2_t);
        auto out_scale = add_scale_mul(m2, scale, scale, 1, 1, dot->get_shape().lens());
        auto d3        = add_quantize_op(m2, "dequantizelinear", dot, out_scale);
        m2.add_return({d3});
    }

    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(dot_multi_scale_all_skip_post_dq_ops)
{
    migraphx::shape sh1{migraphx::shape::float_type, {2, 3, 1280, 1000}};
    migraphx::shape sh2{migraphx::shape::float_type, {1024, 10, 100}};
    migraphx::shape sh3{migraphx::shape::float_type, {1280}};
    migraphx::shape sh4{migraphx::shape::float_type, {1024}};

    migraphx::module m1;
    {
        auto t1     = m1.add_parameter("t1", sh1);
        auto t2     = m1.add_parameter("t2", sh2);
        auto scale1 = m1.add_literal(migraphx::generate_literal(sh3, 0));
        auto scale2 = m1.add_literal(migraphx::generate_literal(sh4, 0));
        auto zero   = m1.add_literal(std::int8_t{0});

        auto q1   = add_quantize_op(m1, "quantizelinear", t1, scale1, zero, 2);
        auto d1   = add_quantize_op(m1, "dequantizelinear", q1, scale1, zero, 2);
        auto q2   = add_quantize_op(m1, "quantizelinear", t2, scale2, zero, 0);
        auto d2   = add_quantize_op(m1, "dequantizelinear", q2, scale2, zero, 0);
        auto d2_r = m1.add_instruction(migraphx::make_op("reshape", {{"dims", {1024, 1000}}}), d2);
        auto d2_t =
            m1.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), d2_r);
        auto d2_mb = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 3, 1000, 1024}}}), d2_t);
        auto dot = m1.add_instruction(migraphx::make_op("dot"), d1, d2_mb);
        m1.add_return({dot});
    }

    migraphx::module m2;
    {
        auto t1     = m2.add_parameter("t1", sh1);
        auto t2     = m2.add_parameter("t2", sh2);
        auto scale1 = m2.add_literal(migraphx::generate_literal(sh3, 0));
        auto scale2 = m2.add_literal(migraphx::generate_literal(sh4, 0));
        auto zero   = m2.add_literal(std::int8_t{0});

        auto q1   = add_quantize_op(m2, "quantizelinear", t1, scale1, zero, 2);
        auto q2   = add_quantize_op(m2, "quantizelinear", t2, scale2, zero, 0);
        auto q2_r = m2.add_instruction(migraphx::make_op("reshape", {{"dims", {1024, 1000}}}), q2);
        auto q2_t =
            m2.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), q2_r);
        auto q2_mb = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 3, 1000, 1024}}}), q2_t);

        auto dot       = m2.add_instruction(migraphx::make_op("quant_dot"), q1, q2_mb);
        auto out_scale = add_scale_mul(m2, scale1, scale2, 2, 3, dot->get_shape().lens());
        auto d3        = add_quantize_op(m2, "dequantizelinear", dot, out_scale);
        m2.add_return({d3});
    }

    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(dot_multi_scale_unsupported_axis)
{
    migraphx::shape sh1{migraphx::shape::float_type, {1280, 1000}};
    migraphx::shape sh2{migraphx::shape::float_type, {1000, 1024}};
    migraphx::shape sh3{migraphx::shape::float_type, {1000}};

    migraphx::module m1;
    {
        auto t1     = m1.add_parameter("t1", sh1);
        auto t2     = m1.add_parameter("t2", sh2);
        auto scale1 = m1.add_literal(migraphx::generate_literal(sh3, 0));
        auto scale2 = m1.add_literal(0.4f);
        auto zero   = m1.add_literal(std::int8_t{0});

        auto q1  = add_quantize_op(m1, "quantizelinear", t1, scale1, zero, 1);
        auto d1  = add_quantize_op(m1, "dequantizelinear", q1, scale1, zero, 1);
        auto q2  = add_quantize_op(m1, "quantizelinear", t2, scale2, zero, 1);
        auto d2  = add_quantize_op(m1, "dequantizelinear", q2, scale2, zero, 1);
        auto dot = m1.add_instruction(migraphx::make_op("dot"), d1, d2);
        m1.add_return({dot});
    }

    migraphx::module m2;
    {
        auto t1  = m2.add_parameter("t1", sh1);
        auto t2  = m2.add_parameter("t2", sh2);
        auto dot = m2.add_instruction(migraphx::make_op("dot"), t1, t2);
        m2.add_return({dot});
    }

    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(dot_non_zero_point)
{
    migraphx::shape sh1{migraphx::shape::float_type, {1280, 1000}};
    migraphx::shape sh2{migraphx::shape::float_type, {1000, 1024}};

    migraphx::module m1;
    {
        auto t1    = m1.add_parameter("t1", sh1);
        auto t2    = m1.add_parameter("t2", sh2);
        auto scale = m1.add_literal(0.5f);
        auto zero  = m1.add_literal(std::int8_t{1});

        auto q1  = add_quantize_op(m1, "quantizelinear", t1, scale, zero);
        auto d1  = add_quantize_op(m1, "dequantizelinear", q1, scale, zero);
        auto q2  = add_quantize_op(m1, "quantizelinear", t2, scale, zero);
        auto d2  = add_quantize_op(m1, "dequantizelinear", q2, scale, zero);
        auto dot = m1.add_instruction(migraphx::make_op("dot"), d1, d2);
        m1.add_return({dot});
    }

    migraphx::module m2;
    {
        auto t1  = m2.add_parameter("t1", sh1);
        auto t2  = m2.add_parameter("t2", sh2);
        auto dot = m2.add_instruction(migraphx::make_op("dot"), t1, t2);
        m2.add_return({dot});
    }

    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(dot_uint8)
{
    migraphx::shape sh1{migraphx::shape::float_type, {1280, 1000}};
    migraphx::shape sh2{migraphx::shape::float_type, {1000, 1024}};

    migraphx::module m1;
    {
        auto t1    = m1.add_parameter("t1", sh1);
        auto t2    = m1.add_parameter("t2", sh2);
        auto scale = m1.add_literal(0.5f);
        auto zero  = m1.add_literal(std::uint8_t{0});

        auto q1  = add_quantize_op(m1, "quantizelinear", t1, scale, zero);
        auto d1  = add_quantize_op(m1, "dequantizelinear", q1, scale, zero);
        auto q2  = add_quantize_op(m1, "quantizelinear", t2, scale, zero);
        auto d2  = add_quantize_op(m1, "dequantizelinear", q2, scale, zero);
        auto dot = m1.add_instruction(migraphx::make_op("dot"), d1, d2);
        m1.add_return({dot});
    }

    migraphx::module m2;
    {
        auto t1  = m2.add_parameter("t1", sh1);
        auto t2  = m2.add_parameter("t2", sh2);
        auto dot = m2.add_instruction(migraphx::make_op("dot"), t1, t2);
        m2.add_return({dot});
    }

    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(dot_add)
{
    migraphx::shape sh1{migraphx::shape::float_type, {1280, 1000}};
    migraphx::shape sh2{migraphx::shape::float_type, {1000, 1024}};
    migraphx::shape sh3{migraphx::shape::float_type, {1280, 1024}};

    migraphx::module m1;
    {
        auto t1    = m1.add_parameter("t1", sh1);
        auto t2    = m1.add_parameter("t2", sh2);
        auto ab    = m1.add_parameter("ab", sh3);
        auto scale = m1.add_literal(0.5f);
        auto zero  = m1.add_literal(std::int8_t{0});

        auto q1  = add_quantize_op(m1, "quantizelinear", t1, scale, zero);
        auto d1  = add_quantize_op(m1, "dequantizelinear", q1, scale, zero);
        auto q2  = add_quantize_op(m1, "quantizelinear", t2, scale, zero);
        auto d2  = add_quantize_op(m1, "dequantizelinear", q2, scale, zero);
        auto dot = m1.add_instruction(migraphx::make_op("dot"), d1, d2);
        auto q3  = add_quantize_op(m1, "quantizelinear", dot, scale, zero);
        auto d3  = add_quantize_op(m1, "dequantizelinear", q3, scale, zero);
        auto add = m1.add_instruction(migraphx::make_op("add"), d3, ab);
        m1.add_return({add});
    }

    migraphx::module m2;
    {
        auto t1    = m2.add_parameter("t1", sh1);
        auto t2    = m2.add_parameter("t2", sh2);
        auto ab    = m2.add_parameter("ab", sh3);
        auto scale = m2.add_literal(0.5f);
        auto zero  = m2.add_literal(std::int8_t{0});

        auto q1        = add_quantize_op(m2, "quantizelinear", t1, scale, zero);
        auto q2        = add_quantize_op(m2, "quantizelinear", t2, scale, zero);
        auto dot       = m2.add_instruction(migraphx::make_op("quant_dot"), q1, q2);
        auto out_scale = add_scale_mul(m2, scale, scale, 1, 1, dot->get_shape().lens());
        auto d3        = add_quantize_op(m2, "dequantizelinear", dot, out_scale);
        auto add       = m2.add_instruction(migraphx::make_op("add"), d3, ab);
        m2.add_return({add});
    }

    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(dot_add_multiple_dq_use)
{
    migraphx::shape sh1{migraphx::shape::float_type, {32, 1}};
    migraphx::shape sh2{migraphx::shape::float_type, {32, 32}};
    migraphx::module m1;
    {
        auto t1    = m1.add_parameter("t1", sh1);
        auto t2    = m1.add_parameter("t2", sh2);
        auto scale = m1.add_literal(0.5f);
        auto zero  = m1.add_literal(std::int8_t{0});

        auto q1 = add_quantize_op(m1, "quantizelinear", t1, scale, zero);
        auto d1 = add_quantize_op(m1, "dequantizelinear", q1, scale, zero);
        auto d1_t =
            m1.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), d1);
        auto d1_tmb =
            m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {32, 32}}}), d1_t);
        auto d1_tmbc = m1.add_instruction(migraphx::make_op("contiguous"), d1_tmb);
        auto q2      = add_quantize_op(m1, "quantizelinear", t2, scale, zero);
        auto d2      = add_quantize_op(m1, "dequantizelinear", q2, scale, zero);
        auto dot_1   = m1.add_instruction(migraphx::make_op("dot"), d1_tmbc, d2);
        auto q3      = add_quantize_op(m1, "quantizelinear", dot_1, scale, zero);
        auto d3      = add_quantize_op(m1, "dequantizelinear", q3, scale, zero);
        auto dot_2   = m1.add_instruction(migraphx::make_op("dot"), d3, d1);
        auto add     = m1.add_instruction(migraphx::make_op("add"), {dot_2, d1});
        m1.add_return({add});
    }

    migraphx::module m2;
    {
        auto t1    = m2.add_parameter("t1", sh1);
        auto t2    = m2.add_parameter("t2", sh2);
        auto scale = m2.add_literal(0.5f);
        auto zero  = m2.add_literal(std::int8_t{0});

        auto q1 = add_quantize_op(m2, "quantizelinear", t1, scale, zero);
        auto q1_t =
            m2.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), q1);
        auto q1_tmb =
            m2.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {32, 32}}}), q1_t);
        auto q1_tmbc     = m2.add_instruction(migraphx::make_op("contiguous"), q1_tmb);
        auto q2          = add_quantize_op(m2, "quantizelinear", t2, scale, zero);
        auto dot_1       = m2.add_instruction(migraphx::make_op("quant_dot"), q1_tmbc, q2);
        auto out_scale   = add_scale_mul(m2, scale, scale, 1, 1, dot_1->get_shape().lens());
        auto d3          = add_quantize_op(m2, "dequantizelinear", dot_1, out_scale);
        auto d3_q        = add_quantize_op(m2, "quantizelinear", d3, scale, zero);
        auto dot_2       = m2.add_instruction(migraphx::make_op("quant_dot"), d3_q, q1);
        auto out_scale_2 = add_scale_mul(m2, scale, scale, 1, 1, dot_2->get_shape().lens());
        auto d4          = add_quantize_op(m2, "dequantizelinear", dot_2, out_scale_2);
        auto add         = m2.add_instruction(migraphx::make_op("add"), d4, t1);
        m2.add_return({add});
    }
    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(conv)
{
    migraphx::shape s4{migraphx::shape::int8_type, {1280, 320, 1, 1}};
    migraphx::shape s7{migraphx::shape::float_type, {1, 320, 7, 7}};

    migraphx::module m1;
    {
        auto input   = m1.add_parameter("input", s7);
        auto weights = m1.add_parameter("weights", s4);
        auto scale   = m1.add_literal(0.5f);
        auto zero    = m1.add_literal(std::int8_t{0});

        auto d1 = add_quantize_op(m1, "dequantizelinear", weights, scale, zero);
        auto q1 = add_quantize_op(m1, "quantizelinear", input, scale, zero);
        auto d5 = add_quantize_op(m1, "dequantizelinear", q1, scale, zero);
        auto c1 = m1.add_instruction(migraphx::make_op("convolution",
                                                       {{"padding", {0, 0, 0, 0}},
                                                        {"stride", {1, 1}},
                                                        {"dilation", {1, 1}},
                                                        {"group", 1},
                                                        {"padding_mode", 0}}),
                                     d5,
                                     d1);
        m1.add_return({c1});
    }

    migraphx::module m2;
    {
        auto input   = m2.add_parameter("input", s7);
        auto weights = m2.add_parameter("weights", s4);
        auto scale   = m2.add_literal(0.5f);
        auto zero    = m2.add_literal(std::int8_t{0});

        auto q1        = add_quantize_op(m2, "quantizelinear", input, scale, zero);
        auto c1        = m2.add_instruction(migraphx::make_op("quant_convolution",
                                                              {{"padding", {0, 0, 0, 0}},
                                                               {"stride", {1, 1}},
                                                               {"dilation", {1, 1}},
                                                               {"group", 1},
                                                               {"padding_mode", 0}}),
                                     q1,
                                     weights);
        auto out_scale = add_scale_mul(m2, scale, scale, 1, 1, c1->get_shape().lens());
        auto d6        = add_quantize_op(m2, "dequantizelinear", c1, out_scale);
        m2.add_return({d6});
    }

    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(conv_multi_scale)
{
    migraphx::shape s4{migraphx::shape::int8_type, {1280, 320, 1, 1}};
    migraphx::shape s7{migraphx::shape::float_type, {1, 320, 7, 7}};
    migraphx::shape s8{migraphx::shape::float_type, {1280}};

    migraphx::module m1;
    {
        auto input     = m1.add_parameter("input", s7);
        auto weights   = m1.add_parameter("weights", s4);
        auto w_scale   = m1.add_literal(migraphx::generate_literal(s8, 0));
        auto inp_scale = m1.add_literal(0.5f);
        auto zero      = m1.add_literal(std::int8_t{0});

        auto d1 = add_quantize_op(m1, "dequantizelinear", weights, w_scale, zero, 0);
        auto q1 = add_quantize_op(m1, "quantizelinear", input, inp_scale, zero);
        auto d5 = add_quantize_op(m1, "dequantizelinear", q1, inp_scale, zero);
        auto c1 = m1.add_instruction(migraphx::make_op("convolution",
                                                       {{"padding", {0, 0, 0, 0}},
                                                        {"stride", {1, 1}},
                                                        {"dilation", {1, 1}},
                                                        {"group", 1},
                                                        {"padding_mode", 0}}),
                                     d5,
                                     d1);
        m1.add_return({c1});
    }

    migraphx::module m2;
    {
        auto input     = m2.add_parameter("input", s7);
        auto weights   = m2.add_parameter("weights", s4);
        auto w_scale   = m2.add_literal(migraphx::generate_literal(s8, 0));
        auto inp_scale = m2.add_literal(0.5f);
        auto zero      = m2.add_literal(std::int8_t{0});

        auto q_inp     = add_quantize_op(m2, "quantizelinear", input, inp_scale, zero);
        auto c1        = m2.add_instruction(migraphx::make_op("quant_convolution",
                                                              {{"padding", {0, 0, 0, 0}},
                                                               {"stride", {1, 1}},
                                                               {"dilation", {1, 1}},
                                                               {"group", 1},
                                                               {"padding_mode", 0}}),
                                     q_inp,
                                     weights);
        auto out_scale = add_scale_mul(m2, inp_scale, w_scale, 1, 1, c1->get_shape().lens());
        auto d1        = add_quantize_op(m2, "dequantizelinear", c1, out_scale);
        m2.add_return({d1});
    }

    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(conv_multi_scale_unsupported_axis)
{
    migraphx::shape s4{migraphx::shape::int8_type, {1280, 320, 1, 1}};
    migraphx::shape s7{migraphx::shape::float_type, {1, 320, 7, 7}};
    migraphx::shape s8{migraphx::shape::float_type, {320}};

    migraphx::module m1;
    {
        auto input   = m1.add_parameter("input", s7);
        auto weights = m1.add_parameter("weights", s4);
        auto scale   = m1.add_literal(migraphx::generate_literal(s8, 0));
        auto zero    = m1.add_literal(std::int8_t{0});

        auto d1 = add_quantize_op(m1, "dequantizelinear", weights, scale, zero);
        auto q1 = add_quantize_op(m1, "quantizelinear", input, scale, zero);
        auto d5 = add_quantize_op(m1, "dequantizelinear", q1, scale, zero);
        auto c1 = m1.add_instruction(migraphx::make_op("convolution",
                                                       {{"padding", {0, 0, 0, 0}},
                                                        {"stride", {1, 1}},
                                                        {"dilation", {1, 1}},
                                                        {"group", 1},
                                                        {"padding_mode", 0}}),
                                     d5,
                                     d1);
        m1.add_return({c1});
    }

    migraphx::module m2;
    {
        auto input   = m2.add_parameter("input", s7);
        auto weights = m2.add_parameter("weights", s4);
        auto scale   = m2.add_literal(migraphx::generate_literal(s8, 0));
        auto zero    = m2.add_literal(std::int8_t{0});

        auto d1 = add_quantize_op(m2, "dequantizelinear", weights, scale, zero);
        auto c1 = m2.add_instruction(migraphx::make_op("convolution",
                                                       {{"padding", {0, 0, 0, 0}},
                                                        {"stride", {1, 1}},
                                                        {"dilation", {1, 1}},
                                                        {"group", 1},
                                                        {"padding_mode", 0}}),
                                     input,
                                     d1);
        m2.add_return({c1});
    }

    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(conv_bias_add)
{
    migraphx::shape s4{migraphx::shape::int8_type, {1280, 320, 1, 1}};
    migraphx::shape s6{migraphx::shape::int32_type, {1280}};
    migraphx::shape s7{migraphx::shape::float_type, {1, 320, 7, 7}};

    migraphx::module m1;
    {
        auto input   = m1.add_parameter("input", s7);
        auto weights = m1.add_parameter("weights", s4);
        auto bias    = m1.add_parameter("bias", s6);
        auto scale   = m1.add_literal(0.5f);
        auto zero    = m1.add_literal(std::int8_t{0});
        auto zero32  = m1.add_literal(std::int32_t{0});

        auto d1 = add_quantize_op(m1, "dequantizelinear", weights, scale, zero);
        auto d2 = add_quantize_op(m1, "dequantizelinear", bias, scale, zero32);
        auto q1 = add_quantize_op(m1, "quantizelinear", input, scale, zero);
        auto d5 = add_quantize_op(m1, "dequantizelinear", q1, scale, zero);
        auto c1 = m1.add_instruction(migraphx::make_op("convolution",
                                                       {{"padding", {0, 0, 0, 0}},
                                                        {"stride", {1, 1}},
                                                        {"dilation", {1, 1}},
                                                        {"group", 1},
                                                        {"padding_mode", 0}}),
                                     d5,
                                     d1);
        auto b1 = m1.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {1, 1280, 7, 7}}}), d2);
        auto a1 = m1.add_instruction(migraphx::make_op("add"), c1, b1);
        m1.add_return({a1});
    }

    migraphx::module m2;
    {
        auto input   = m2.add_parameter("input", s7);
        auto weights = m2.add_parameter("weights", s4);
        auto bias    = m2.add_parameter("bias", s6);
        auto scale   = m2.add_literal(0.5f);
        auto zero    = m2.add_literal(std::int8_t{0});
        auto zero32  = m2.add_literal(std::int32_t{0});

        auto d2        = add_quantize_op(m2, "dequantizelinear", bias, scale, zero32);
        auto q1        = add_quantize_op(m2, "quantizelinear", input, scale, zero);
        auto c1        = m2.add_instruction(migraphx::make_op("quant_convolution",
                                                              {{"padding", {0, 0, 0, 0}},
                                                               {"stride", {1, 1}},
                                                               {"dilation", {1, 1}},
                                                               {"group", 1},
                                                               {"padding_mode", 0}}),
                                     q1,
                                     weights);
        auto out_scale = add_scale_mul(m2, scale, scale, 1, 1, c1->get_shape().lens());
        auto d6        = add_quantize_op(m2, "dequantizelinear", c1, out_scale);
        auto b1        = m2.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {1, 1280, 7, 7}}}), d2);
        auto a1 = m2.add_instruction(migraphx::make_op("add"), d6, b1);
        m2.add_return({a1});
    }

    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(conv_pooling_dot)
{
    migraphx::shape s2{migraphx::shape::int8_type, {1280, 1000}};
    migraphx::shape s3{migraphx::shape::int8_type, {1000}};
    migraphx::shape s4{migraphx::shape::int8_type, {1280, 320, 1, 1}};
    migraphx::shape s6{migraphx::shape::int32_type, {1280}};
    migraphx::shape s7{migraphx::shape::float_type, {1, 320, 7, 7}};

    migraphx::module m1;
    {
        auto db      = m1.add_parameter("db", s2); // dot input b
        auto ab      = m1.add_parameter("ab", s3); // add input b
        auto weights = m1.add_parameter("weights", s4);
        auto bias    = m1.add_parameter("bias", s6);
        auto input   = m1.add_parameter("input", s7);
        auto scale   = m1.add_literal(0.5f);
        auto zero    = m1.add_literal(std::int8_t{0});
        auto zero32  = m1.add_literal(std::int32_t{0});

        auto d1  = add_quantize_op(m1, "dequantizelinear", weights, scale, zero);
        auto d2  = add_quantize_op(m1, "dequantizelinear", bias, scale, zero32);
        auto d3  = add_quantize_op(m1, "dequantizelinear", ab, scale, zero);
        auto d4  = add_quantize_op(m1, "dequantizelinear", db, scale, zero);
        auto q1  = add_quantize_op(m1, "quantizelinear", input, scale, zero);
        auto d5  = add_quantize_op(m1, "dequantizelinear", q1, scale, zero);
        auto c1  = m1.add_instruction(migraphx::make_op("convolution",
                                                        {{"padding", {0, 0, 0, 0}},
                                                         {"stride", {1, 1}},
                                                         {"dilation", {1, 1}},
                                                         {"group", 1},
                                                         {"padding_mode", 0}}),
                                     d5,
                                     d1);
        auto bc1 = m1.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {1, 1280, 7, 7}}}), d2);
        auto a1 = m1.add_instruction(migraphx::make_op("add"), c1, bc1);
        auto ap =
            m1.add_instruction(migraphx::make_op("pooling",
                                                 {{"mode", migraphx::op::pooling_mode::average},
                                                  {"padding", {0, 0, 0, 0}},
                                                  {"stride", {1, 1}},
                                                  {"lengths", {7, 7}},
                                                  {"dilations", {1, 1}},
                                                  {"ceil_mode", 0}}),
                               a1);
        auto fl  = m1.add_instruction(migraphx::make_op("flatten", {{"axis", 1}}), ap);
        auto q4  = add_quantize_op(m1, "quantizelinear", fl, scale, zero);
        auto d8  = add_quantize_op(m1, "dequantizelinear", q4, scale, zero);
        auto dot = m1.add_instruction(migraphx::make_op("dot"), d8, d4);
        auto q5  = add_quantize_op(m1, "quantizelinear", dot, scale, zero);
        auto d9  = add_quantize_op(m1, "dequantizelinear", q5, scale, zero);
        auto mb1 =
            m1.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {1, 1000}}}), d3);
        auto a2 = m1.add_instruction(migraphx::make_op("add"), d9, mb1);
        m1.add_return({a2});
    }

    migraphx::module m2;
    {
        auto db      = m2.add_parameter("db", s2); // dot input b
        auto ab      = m2.add_parameter("ab", s3); // add input b
        auto weights = m2.add_parameter("weights", s4);
        auto bias    = m2.add_parameter("bias", s6);
        auto input   = m2.add_parameter("input", s7);
        auto scale   = m2.add_literal(0.5f);
        auto zero    = m2.add_literal(std::int8_t{0});
        auto zero32  = m2.add_literal(std::int32_t{0});

        auto d2         = add_quantize_op(m2, "dequantizelinear", bias, scale, zero32);
        auto d3         = add_quantize_op(m2, "dequantizelinear", ab, scale, zero);
        auto q1         = add_quantize_op(m2, "quantizelinear", input, scale, zero);
        auto c1         = m2.add_instruction(migraphx::make_op("quant_convolution",
                                                               {{"padding", {0, 0, 0, 0}},
                                                                {"stride", {1, 1}},
                                                                {"dilation", {1, 1}},
                                                                {"group", 1},
                                                                {"padding_mode", 0}}),
                                     q1,
                                     weights);
        auto out_scale1 = add_scale_mul(m2, scale, scale, 1, 1, c1->get_shape().lens());
        auto d5         = add_quantize_op(m2, "dequantizelinear", c1, out_scale1);
        auto bc1        = m2.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {1, 1280, 7, 7}}}), d2);
        auto a1 = m2.add_instruction(migraphx::make_op("add"), d5, bc1);
        auto ap =
            m2.add_instruction(migraphx::make_op("pooling",
                                                 {{"mode", migraphx::op::pooling_mode::average},
                                                  {"padding", {0, 0, 0, 0}},
                                                  {"stride", {1, 1}},
                                                  {"lengths", {7, 7}},
                                                  {"dilations", {1, 1}},
                                                  {"ceil_mode", 0}}),
                               a1);
        auto fl         = m2.add_instruction(migraphx::make_op("flatten", {{"axis", 1}}), ap);
        auto q4         = add_quantize_op(m2, "quantizelinear", fl, scale, zero);
        auto dot        = m2.add_instruction(migraphx::make_op("quant_dot"), q4, db);
        auto out_scale2 = add_scale_mul(m2, scale, scale, 1, 0, dot->get_shape().lens());
        auto d9         = add_quantize_op(m2, "dequantizelinear", dot, out_scale2);
        auto mb1 =
            m2.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {1, 1000}}}), d3);
        auto a2 = m2.add_instruction(migraphx::make_op("add"), d9, mb1);
        m2.add_return({a2});
    }

    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(mobilenet_snippet)
{
    migraphx::shape s2{migraphx::shape::int8_type, {1280, 1000}};
    migraphx::shape s3{migraphx::shape::int8_type, {1000}};
    migraphx::shape s4{migraphx::shape::int8_type, {1280, 320, 1, 1}};
    migraphx::shape s6{migraphx::shape::int32_type, {1280}};
    migraphx::shape s7{migraphx::shape::float_type, {1, 320, 7, 7}};

    auto create_module = [&]() {
        migraphx::module mm;
        auto db      = mm.add_parameter("db", s2); // dot input b
        auto ab      = mm.add_parameter("ab", s3); // add input b
        auto weights = mm.add_parameter("weights", s4);
        auto bias    = mm.add_parameter("bias", s6);
        auto input   = mm.add_parameter("input", s7);
        auto scale   = mm.add_literal(0.5f);
        auto zero    = mm.add_literal(std::int8_t{0});
        auto zero32  = mm.add_literal(std::int32_t{0});

        auto d1  = add_quantize_op(mm, "dequantizelinear", weights, scale, zero);
        auto d2  = add_quantize_op(mm, "dequantizelinear", bias, scale, zero32);
        auto d3  = add_quantize_op(mm, "dequantizelinear", ab, scale, zero);
        auto d4  = add_quantize_op(mm, "dequantizelinear", db, scale, zero);
        auto q1  = add_quantize_op(mm, "quantizelinear", input, scale, zero);
        auto d5  = add_quantize_op(mm, "dequantizelinear", q1, scale, zero);
        auto c1  = mm.add_instruction(migraphx::make_op("convolution",
                                                        {{"padding", {0, 0, 0, 0}},
                                                         {"stride", {1, 1}},
                                                         {"dilation", {1, 1}},
                                                         {"group", 1},
                                                         {"padding_mode", 0}}),
                                     d5,
                                     d1);
        auto bc1 = mm.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {1, 1280, 7, 7}}}), d2);
        auto a1 = mm.add_instruction(migraphx::make_op("add"), c1, bc1);
        auto q2 = add_quantize_op(mm, "quantizelinear", a1, scale, zero);
        auto d6 = add_quantize_op(mm, "dequantizelinear", q2, scale, zero);
        auto ap =
            mm.add_instruction(migraphx::make_op("pooling",
                                                 {{"mode", migraphx::op::pooling_mode::average},
                                                  {"padding", {0, 0, 0, 0}},
                                                  {"stride", {1, 1}},
                                                  {"lengths", {7, 7}},
                                                  {"dilations", {1, 1}},
                                                  {"ceil_mode", 0}}),
                               d6);
        auto q3  = add_quantize_op(mm, "quantizelinear", ap, scale, zero);
        auto d7  = add_quantize_op(mm, "dequantizelinear", q3, scale, zero);
        auto rs  = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {1, -1}}}), d7);
        auto q4  = add_quantize_op(mm, "quantizelinear", rs, scale, zero);
        auto d8  = add_quantize_op(mm, "dequantizelinear", q4, scale, zero);
        auto dot = mm.add_instruction(migraphx::make_op("dot"), d8, d4);
        auto q5  = add_quantize_op(mm, "quantizelinear", dot, scale, zero);
        auto d9  = add_quantize_op(mm, "dequantizelinear", q5, scale, zero);
        auto mb1 =
            mm.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {1, 1000}}}), d3);
        auto a2 = mm.add_instruction(migraphx::make_op("add"), d9, mb1);
        mm.add_return({a2});

        return mm;
    };

    auto mod1 = create_module();
    auto mod2 = create_module();
    run_pass(mod2);

    auto match_qdq = migraphx::match::name("dequantizelinear")(
        migraphx::match::arg(0)(migraphx::match::name("quantizelinear")));
    auto ins1 = migraphx::match::find_match(mod1, match_qdq);
    auto ins2 = migraphx::match::find_match(mod2, match_qdq);

    EXPECT((ins1.result != mod1.end()) and (ins2.result == mod2.end()));
    EXPECT(any_of(mod1, &is_convolution));
    EXPECT(none_of(mod2, &is_convolution));
    EXPECT(any_of(mod1, &is_dot));
    EXPECT(none_of(mod2, &is_dot));
}

TEST_CASE(conv_correctness)
{
    migraphx::shape si{migraphx::shape::float_type, {2, 3, 4, 4}};
    migraphx::shape sw{migraphx::shape::int8_type, {2, 3, 3, 3}};

    migraphx::program p1;
    {
        auto* m1     = p1.get_main_module();
        auto input   = m1->add_parameter("input", si);
        auto weights = m1->add_parameter("weights", sw);
        auto scale_i = m1->add_literal(0.5f);
        auto scale_w = m1->add_literal(0.1f);
        auto zero    = m1->add_literal(std::int8_t{0});

        auto d1 = add_quantize_op(*m1, "dequantizelinear", weights, scale_w, zero);
        auto q1 = add_quantize_op(*m1, "quantizelinear", input, scale_i, zero);
        auto d5 = add_quantize_op(*m1, "dequantizelinear", q1, scale_i, zero);
        auto c1 = m1->add_instruction(migraphx::make_op("convolution",
                                                        {{"padding", {0, 0, 0, 0}},
                                                         {"stride", {1, 1}},
                                                         {"dilation", {1, 1}},
                                                         {"group", 1},
                                                         {"padding_mode", 0}}),
                                      d5,
                                      d1);
        m1->add_return({c1});
        run_pass(*m1);
    }

    migraphx::program p2;
    {
        auto* m2     = p2.get_main_module();
        auto input   = m2->add_parameter("input", si);
        auto weights = m2->add_parameter("weights", sw);
        auto scale   = m2->add_literal(0.1f);
        auto zero    = m2->add_literal(std::int8_t{0});

        auto d1 = add_quantize_op(*m2, "dequantizelinear", weights, scale, zero);
        auto c1 = m2->add_instruction(migraphx::make_op("convolution",
                                                        {{"padding", {0, 0, 0, 0}},
                                                         {"stride", {1, 1}},
                                                         {"dilation", {1, 1}},
                                                         {"group", 1},
                                                         {"padding_mode", 0}}),
                                      input,
                                      d1);
        m2->add_return({c1});
    }

    std::vector<float> iv(si.elements(), 4);
    auto input = migraphx::argument(si, iv.data());
    std::vector<float> wv(sw.elements(), 10);
    auto weights = migraphx::argument(sw, wv.data());
    p1.compile(migraphx::target(migraphx::make_target("ref")));
    p2.compile(migraphx::target(migraphx::make_target("ref")));

    auto result1 = p1.eval({{"input", input}, {"weights", weights}}).back();
    std::vector<float> rv1(16);
    result1.visit([&](auto output) { rv1.assign(output.begin(), output.end()); });
    auto result2 = p2.eval({{"input", input}, {"weights", weights}}).back();
    std::vector<float> rv2(16);
    result2.visit([&](auto output) { rv2.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(rv1, rv2));
}

TEST_CASE(dot_correctness)
{
    migraphx::shape sh1{migraphx::shape::float_type, {10, 4}};
    migraphx::shape sh2{migraphx::shape::float_type, {4, 12}};
    migraphx::shape sh3{migraphx::shape::float_type, {10, 12}};

    migraphx::program p1;
    {
        auto* m1     = p1.get_main_module();
        auto a       = m1->add_parameter("a", sh1);
        auto b       = m1->add_parameter("b", sh2);
        auto scale_a = m1->add_literal(0.4f);
        auto scale_b = m1->add_literal(0.5f);
        auto zero    = m1->add_literal(std::int8_t{0});

        auto q1  = add_quantize_op(*m1, "quantizelinear", a, scale_a, zero);
        auto d1  = add_quantize_op(*m1, "dequantizelinear", q1, scale_a, zero);
        auto q2  = add_quantize_op(*m1, "quantizelinear", b, scale_b, zero);
        auto d2  = add_quantize_op(*m1, "dequantizelinear", q2, scale_b, zero);
        auto dot = m1->add_instruction(migraphx::make_op("dot"), d1, d2);
        m1->add_return({dot});

        run_pass(*m1);
    }

    migraphx::program p2;
    {
        auto* m2 = p2.get_main_module();
        auto a   = m2->add_parameter("a", sh1);
        auto b   = m2->add_parameter("b", sh2);
        auto dot = m2->add_instruction(migraphx::make_op("dot"), a, b);
        m2->add_return({dot});
    }

    std::vector<float> av(sh1.elements(), 10);
    auto a = migraphx::argument(sh1, av.data());
    std::vector<float> bv(sh2.elements(), 10);
    auto b = migraphx::argument(sh2, bv.data());
    p1.compile(migraphx::target(migraphx::make_target("ref")));
    p2.compile(migraphx::target(migraphx::make_target("ref")));

    auto result1 = p1.eval({{"a", a}, {"b", b}}).back();
    std::vector<float> rv1(sh3.elements());
    result1.visit([&](auto output) { rv1.assign(output.begin(), output.end()); });
    auto result2 = p2.eval({{"a", a}, {"b", b}}).back();
    std::vector<float> rv2(sh3.elements());
    result2.visit([&](auto output) { rv2.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(rv1, rv2));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
