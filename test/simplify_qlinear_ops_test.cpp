/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/simplify_qlinear_ops.hpp>
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

void run_pass(migraphx::module& m) { run_passes(m, {migraphx::simplify_qlinear_ops{}}); }

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

migraphx::instruction_ref broadcast_shift(migraphx::module& m,
                                          migraphx::instruction_ref shift,
                                          const std::vector<std::size_t>& out_lens)
{
    if(shift->get_shape().lens() == out_lens)
        return shift;
    return m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", out_lens}}), shift);
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
    auto shift_mb = broadcast_shift(m, shift, lens);
    return m.add_instruction(migraphx::make_op(name), x, scale_mb, shift_mb);
}

migraphx::instruction_ref add_quantize_op(migraphx::module& m,
                                          const std::string& name,
                                          migraphx::instruction_ref x,
                                          migraphx::instruction_ref scale,
                                          migraphx::shape output_shape = migraphx::shape{},
                                          std::size_t q_axis           = 1)
{
    auto lens     = x->get_shape().lens();
    auto scale_mb = broadcast_scale(m, scale, lens, q_axis);
    auto op       = migraphx::make_op(name);
    auto op_val   = op.to_value();
    if(name == "quantizelinear")
    {
        op_val["output_shape"] = to_value(output_shape);
    }
    return m.add_instruction(migraphx::make_op(name, op_val), x, scale_mb);
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

migraphx::instruction_ref init_zero_point(migraphx::module& m, migraphx::instruction_ref q_ins)
{
    auto zp = m.add_literal(migraphx::literal{migraphx::shape{q_ins->get_shape().type()}, {0}});
    return m.add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", q_ins->get_shape().lens()}}), zp);
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
        auto zero1 = m1.add_literal(std::int8_t{0});
        auto zero2 = m1.add_literal(std::int8_t{0});

        auto q1 = add_quantize_op(m1, "quantizelinear", t1, scale, zero1);
        auto q2 = add_quantize_op(m1, "quantizelinear", t2, scale, zero2);

        auto dot       = m1.add_instruction(migraphx::make_op("quant_dot"), q1, q2);
        auto out_scale = add_scale_mul(m1, scale, scale, 1, 1, dot->get_shape().lens());
        auto out_zp    = init_zero_point(m1, dot);
        auto d3        = add_quantize_op(m1, "dequantizelinear", dot, out_scale, out_zp);
        m1.add_return({d3});
    }
    migraphx::module m2;
    {
        auto t1    = m2.add_parameter("t1", sh1);
        auto t2    = m2.add_parameter("t2", sh2);
        auto scale = m2.add_literal(0.5f);

        auto q1 = add_quantize_op(m2,
                                  "quantizelinear",
                                  t1,
                                  scale,
                                  migraphx::shape{migraphx::shape::int8_type, sh1.lens()});
        auto q2 = add_quantize_op(m2,
                                  "quantizelinear",
                                  t2,
                                  scale,
                                  migraphx::shape{migraphx::shape::int8_type, sh2.lens()});

        auto dot       = m2.add_instruction(migraphx::make_op("quant_dot"), q1, q2);
        auto out_scale = add_scale_mul(m2, scale, scale, 1, 1, dot->get_shape().lens());
        auto d3        = add_quantize_op(m2, "dequantizelinear", dot, out_scale);
        m2.add_return({d3});
    }
    run_pass(m1);
    EXPECT(m1 == m2);
}

TEST_CASE(dot_asymmetric_first_arg)
{
    migraphx::shape sh1{migraphx::shape::float_type, {1280, 1000}};
    migraphx::shape sh2{migraphx::shape::float_type, {1000, 1024}};

    migraphx::module m1;
    {
        auto t1    = m1.add_parameter("t1", sh1);
        auto t2    = m1.add_parameter("t2", sh2);
        auto scale = m1.add_literal(0.5f);
        auto zp1   = m1.add_literal(std::int8_t{1});
        auto zp2   = m1.add_literal(std::int8_t{0});

        auto q1  = add_quantize_op(m1, "quantizelinear", t1, scale, zp1);
        auto q2  = add_quantize_op(m1, "quantizelinear", t2, scale, zp2);
        auto dot = m1.add_instruction(migraphx::make_op("quant_dot"), q1, q2);

        auto out_scale = add_scale_mul(m1, scale, scale, 1, 1, dot->get_shape().lens());

        auto out_zp  = init_zero_point(m1, dot);
        auto zp1_bc  = broadcast_shift(m1, zp1, t1->get_shape().lens());
        auto zp_term = m1.add_instruction(migraphx::make_op("quant_dot"), zp1_bc, q2);
        out_zp       = m1.add_instruction(migraphx::make_op("add"), out_zp, zp_term);

        auto d3 = add_quantize_op(m1, "dequantizelinear", dot, out_scale, out_zp);
        m1.add_return({d3});
    }

    run_pass(m1);

    migraphx::module m2;
    {
        auto t1    = m2.add_parameter("t1", sh1);
        auto t2    = m2.add_parameter("t2", sh2);
        auto scale = m2.add_literal(0.5f);
        auto zp1   = m2.add_literal(std::int8_t{1});

        auto q1  = add_quantize_op(m2, "quantizelinear", t1, scale, zp1);
        auto q2  = add_quantize_op(m2,
                                  "quantizelinear",
                                  t2,
                                  scale,
                                  migraphx::shape{migraphx::shape::int8_type, sh2.lens()});
        auto dot = m2.add_instruction(migraphx::make_op("quant_dot"), q1, q2);

        auto out_scale = add_scale_mul(m2, scale, scale, 1, 1, dot->get_shape().lens());

        auto out_zp  = init_zero_point(m2, dot);
        auto zp1_bc  = broadcast_shift(m2, zp1, t1->get_shape().lens());
        auto zp_term = m2.add_instruction(migraphx::make_op("quant_dot"), zp1_bc, q2);
        out_zp       = m2.add_instruction(migraphx::make_op("add"), out_zp, zp_term);

        auto d3 = add_quantize_op(m2, "dequantizelinear", dot, out_scale, out_zp);
        m2.add_return({d3});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(dot_asymmetric_both_args)
{
    migraphx::shape sh1{migraphx::shape::float_type, {1280, 1000}};
    migraphx::shape sh2{migraphx::shape::float_type, {1000, 1024}};

    migraphx::module m1;
    {
        auto t1    = m1.add_parameter("t1", sh1);
        auto t2    = m1.add_parameter("t2", sh2);
        auto scale = m1.add_literal(0.5f);
        auto zp1   = m1.add_literal(std::int8_t{2});
        auto zp2   = m1.add_literal(std::int8_t{1});

        auto q1  = add_quantize_op(m1, "quantizelinear", t1, scale, zp1);
        auto q2  = add_quantize_op(m1, "quantizelinear", t2, scale, zp2);
        auto dot = m1.add_instruction(migraphx::make_op("quant_dot"), q1, q2);

        auto out_scale = add_scale_mul(m1, scale, scale, 1, 1, dot->get_shape().lens());

        auto out_zp   = init_zero_point(m1, dot);
        auto zp1_bc   = broadcast_shift(m1, zp1, t1->get_shape().lens());
        auto zp2_bc   = broadcast_shift(m1, zp2, t2->get_shape().lens());
        auto zp_term1 = m1.add_instruction(migraphx::make_op("quant_dot"), zp1_bc, q2);
        out_zp        = m1.add_instruction(migraphx::make_op("add"), out_zp, zp_term1);
        auto zp_term2 = m1.add_instruction(migraphx::make_op("quant_dot"), q1, zp2_bc);
        out_zp        = m1.add_instruction(migraphx::make_op("add"), out_zp, zp_term2);
        auto zp_term3 = m1.add_instruction(migraphx::make_op("quant_dot"), zp1_bc, zp2_bc);
        out_zp        = m1.add_instruction(migraphx::make_op("sub"), out_zp, zp_term3);

        auto d3 = add_quantize_op(m1, "dequantizelinear", dot, out_scale, out_zp);
        m1.add_return({d3});
    }
    migraphx::module m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
