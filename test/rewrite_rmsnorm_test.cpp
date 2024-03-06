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
#include <migraphx/rewrite_rmsnorm.hpp>
#include <migraphx/program.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/common.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/verify.hpp>

#include <test.hpp>

void create_rmsnorm_fp16_div_sqrt(migraphx::module& m, const std::vector<std::size_t>& input_lens)
{
    migraphx::shape s_input{migraphx::shape::half_type, input_lens};
    migraphx::shape s_lit{migraphx::shape::half_type, {1}};
    auto l_pow_2     = m.add_literal(migraphx::literal{s_lit, {2.0f}});
    auto l_eps       = m.add_literal(migraphx::literal{s_lit, {1e-05}});
    auto l_div_1     = m.add_literal(migraphx::literal{s_lit, {1.0f}});
    auto input       = m.add_parameter("input", s_input);
    auto pow         = add_common_op(m, migraphx::make_op("pow"), {input, l_pow_2});
    auto reduce_mean = m.add_instruction(migraphx::make_op("reduce_mean", {{"axes", {-1}}}), pow);
    auto add         = add_common_op(m, migraphx::make_op("add"), {reduce_mean, l_eps});
    auto sqrt        = m.add_instruction(migraphx::make_op("sqrt"), add);
    auto div         = add_common_op(m, migraphx::make_op("div"), {l_div_1, sqrt});
    auto mul         = add_common_op(m, migraphx::make_op("mul"), {input, div});
    m.add_return({mul});
}

void create_rmsnorm_fp16_mul_rsqrt(migraphx::module& m, const std::vector<std::size_t>& input_lens)
{
    migraphx::shape s_input{migraphx::shape::half_type, input_lens};
    migraphx::shape s_lit{migraphx::shape::half_type, {1}};
    auto l_pow_2     = m.add_literal(migraphx::literal{s_lit, {2.0f}});
    auto l_eps       = m.add_literal(migraphx::literal{s_lit, {1e-05}});
    auto input       = m.add_parameter("input", s_input);
    auto pow         = add_common_op(m, migraphx::make_op("pow"), {input, l_pow_2});
    auto reduce_mean = m.add_instruction(migraphx::make_op("reduce_mean", {{"axes", {-1}}}), pow);
    auto add         = add_common_op(m, migraphx::make_op("add"), {reduce_mean, l_eps});
    auto rsqrt       = m.add_instruction(migraphx::make_op("rsqrt"), add);
    auto mul         = add_common_op(m, migraphx::make_op("mul"), {input, rsqrt});
    m.add_return({mul});
}

TEST_CASE(rewrite_rmsnorm_fp16_to_fp32_div_sqrt_test)
{
    migraphx::shape s_input{migraphx::shape::half_type, {1, 256, 4096}};
    migraphx::shape s_lit{migraphx::shape::half_type, {1}};
    migraphx::module m1;
    create_rmsnorm_fp16_div_sqrt(m1, s_input.lens());
    migraphx::rewrite_rmsnorm pass{};
    pass.apply(m1);

    migraphx::module m2;
    {
        auto l_pow_2      = m2.add_literal(migraphx::literal{s_lit, {2.0f}});
        auto l_eps        = m2.add_literal(migraphx::literal{s_lit, {1e-05}});
        auto l_div_1      = m2.add_literal(migraphx::literal{s_lit, {1.0f}});
        auto input        = m2.add_parameter("input", s_input);
        auto pow_2_mbcast = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s_input.lens()}}), l_pow_2);
        auto input_f = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), input);
        auto pow_2_mbcast_f = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}),
            pow_2_mbcast);
        auto pow = m2.add_instruction(migraphx::make_op("pow"), {input_f, pow_2_mbcast_f});
        auto reduce_mean =
            m2.add_instruction(migraphx::make_op("reduce_mean", {{"axes", {-1}}}), pow);
        auto add   = add_common_op(m2, migraphx::make_op("add"), {reduce_mean, l_eps});
        auto sqrt  = m2.add_instruction(migraphx::make_op("sqrt"), add);
        auto div   = add_common_op(m2, migraphx::make_op("div"), {l_div_1, sqrt});
        auto div_h = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), div);
        auto mul = add_common_op(m2, migraphx::make_op("mul"), {input, div_h});
        m2.add_return({mul});
    }

    EXPECT(m1 == m2);
}

TEST_CASE(rewrite_rmsnorm_fp16_to_fp32_mul_rsqrt_test)
{
    migraphx::shape s_input{migraphx::shape::half_type, {1, 256, 4096}};
    migraphx::shape s_lit{migraphx::shape::half_type, {1}};
    migraphx::module m1;
    create_rmsnorm_fp16_mul_rsqrt(m1, s_input.lens());
    migraphx::rewrite_rmsnorm pass{};
    pass.apply(m1);

    migraphx::module m2;
    {
        auto l_pow_2      = m2.add_literal(migraphx::literal{s_lit, {2.0f}});
        auto l_eps        = m2.add_literal(migraphx::literal{s_lit, {1e-05}});
        auto input        = m2.add_parameter("input", s_input);
        auto pow_2_mbcast = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s_input.lens()}}), l_pow_2);
        auto input_f = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), input);
        auto pow_2_mbcast_f = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}),
            pow_2_mbcast);
        auto pow = m2.add_instruction(migraphx::make_op("pow"), {input_f, pow_2_mbcast_f});
        auto reduce_mean =
            m2.add_instruction(migraphx::make_op("reduce_mean", {{"axes", {-1}}}), pow);
        auto add          = add_common_op(m2, migraphx::make_op("add"), {reduce_mean, l_eps});
        auto rsqrt        = add_common_op(m2, migraphx::make_op("rsqrt"), {add});
        auto rsqrt_mbcast = m2.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s_input.lens()}}), rsqrt);
        auto rsqrt_mbcast_h = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}),
            rsqrt_mbcast);
        auto mul = m2.add_instruction(migraphx::make_op("mul"), input, rsqrt_mbcast_h);
        m2.add_return({mul});
    }

    EXPECT(m1 == m2);
}

TEST_CASE(rewrite_rmsnorm_fp16_to_fp32_div_sqrt_accuracy_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape s_input{migraphx::shape::half_type, {1, 3, 9}};
    create_rmsnorm_fp16_div_sqrt(*mm, s_input.lens());
    migraphx::rewrite_rmsnorm pass{};
    pass.apply(*mm);
    p.compile(migraphx::make_target("ref"));

    // Note: converting the calculation to fp32 can handle half::max numbers
    std::vector<migraphx::half> data{
        migraphx::half{-2446.96233826},  migraphx::half{-37347.03145467},
        migraphx::half{-1829.02050925},  migraphx::half{10155.3566038},
        migraphx::half{-11032.43548165}, migraphx::half{-215.74688557},
        migraphx::half{5642.95552563},   migraphx::half{31193.5484022},
        migraphx::half{-12379.14968852}, migraphx::half{3876.8102353},
        migraphx::half{5228.56319534},   migraphx::half{4017.52134421},
        migraphx::half{-41109.51403055}, migraphx::half{-13427.75230111},
        migraphx::half{17637.85685999},  migraphx::half{13903.10580781},
        migraphx::half{-6630.92344508},  migraphx::half{-36526.86578131},
        migraphx::half{-18907.53509807}, migraphx::half{-7211.00914282},
        migraphx::half{-33006.71807566}, migraphx::half{8259.18826548},
        migraphx::half{-9487.80837355},  migraphx::half{-2824.52207497},
        migraphx::half{24448.37690028},  migraphx::half{-770.63316253},
        migraphx::half{310.0554823}};
    migraphx::parameter_map params;
    params["input"] = migraphx::argument(s_input, data.data());
    auto result     = p.eval(params).back();

    std::vector<migraphx::half> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<migraphx::half> gold{
        migraphx::half{-0.13904916}, migraphx::half{-2.12225311}, migraphx::half{-0.10393448},
        migraphx::half{0.57708033},  migraphx::half{-0.62692052}, migraphx::half{-0.01225986},
        migraphx::half{0.32066216},  migraphx::half{1.77258011},  migraphx::half{-0.70344785},
        migraphx::half{0.18838872},  migraphx::half{0.25407546},  migraphx::half{0.1952264},
        migraphx::half{-1.99766516}, migraphx::half{-0.65250474}, migraphx::half{0.85708948},
        migraphx::half{0.67560395},  migraphx::half{-0.32222139}, migraphx::half{-1.77497712},
        migraphx::half{-1.19223013}, migraphx::half{-0.4546961},  migraphx::half{-2.08126568},
        migraphx::half{0.52078989},  migraphx::half{-0.59826154}, migraphx::half{-0.17810256},
        migraphx::half{1.54161245},  migraphx::half{-0.04859291}, migraphx::half{0.0195508}};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(rewrite_rmsnorm_fp16_to_fp32_mul_rsqrt_accuracy_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape s_input{migraphx::shape::half_type, {1, 3, 9}};
    create_rmsnorm_fp16_mul_rsqrt(*mm, s_input.lens());
    migraphx::rewrite_rmsnorm pass{};
    pass.apply(*mm);
    p.compile(migraphx::make_target("ref"));

    // Note: converting the calculation to fp32 can handle half::max numbers
    std::vector<migraphx::half> data{
        migraphx::half{-2446.96233826},  migraphx::half{-37347.03145467},
        migraphx::half{-1829.02050925},  migraphx::half{10155.3566038},
        migraphx::half{-11032.43548165}, migraphx::half{-215.74688557},
        migraphx::half{5642.95552563},   migraphx::half{31193.5484022},
        migraphx::half{-12379.14968852}, migraphx::half{3876.8102353},
        migraphx::half{5228.56319534},   migraphx::half{4017.52134421},
        migraphx::half{-41109.51403055}, migraphx::half{-13427.75230111},
        migraphx::half{17637.85685999},  migraphx::half{13903.10580781},
        migraphx::half{-6630.92344508},  migraphx::half{-36526.86578131},
        migraphx::half{-18907.53509807}, migraphx::half{-7211.00914282},
        migraphx::half{-33006.71807566}, migraphx::half{8259.18826548},
        migraphx::half{-9487.80837355},  migraphx::half{-2824.52207497},
        migraphx::half{24448.37690028},  migraphx::half{-770.63316253},
        migraphx::half{310.0554823}};
    migraphx::parameter_map params;
    params["input"] = migraphx::argument(s_input, data.data());
    auto result     = p.eval(params).back();

    std::vector<migraphx::half> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<migraphx::half> gold{
        migraphx::half{-0.13904916}, migraphx::half{-2.12225311}, migraphx::half{-0.10393448},
        migraphx::half{0.57708033},  migraphx::half{-0.62692052}, migraphx::half{-0.01225986},
        migraphx::half{0.32066216},  migraphx::half{1.77258011},  migraphx::half{-0.70344785},
        migraphx::half{0.18838872},  migraphx::half{0.25407546},  migraphx::half{0.1952264},
        migraphx::half{-1.99766516}, migraphx::half{-0.65250474}, migraphx::half{0.85708948},
        migraphx::half{0.67560395},  migraphx::half{-0.32222139}, migraphx::half{-1.77497712},
        migraphx::half{-1.19223013}, migraphx::half{-0.4546961},  migraphx::half{-2.08126568},
        migraphx::half{0.52078989},  migraphx::half{-0.59826154}, migraphx::half{-0.17810256},
        migraphx::half{1.54161245},  migraphx::half{-0.04859291}, migraphx::half{0.0195508}};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
