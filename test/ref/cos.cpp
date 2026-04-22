/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
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
#include "migraphx/module.hpp"
#include <migraphx/instruction.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/program.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <migraphx/load_save.hpp>
#include <random>

#include <test.hpp>

TEST_CASE(cos_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3}};
    std::vector<float> data{-1, 0, 1};
    auto l = mm->add_literal(migraphx::literal{s, data});
    mm->add_instruction(migraphx::make_op("cos"), l);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = data;
    std::transform(
        gold.begin(), gold.end(), gold.begin(), [](float n) -> float { return cosf(n); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(cos_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape::dynamic_dimension dd{3, 8};
    migraphx::shape s{migraphx::shape::float_type, {dd}};
    auto input = mm->add_parameter("X", s);
    mm->add_instruction(migraphx::make_op("cos"), input);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> input_data{-1, 0, 1};
    migraphx::parameter_map params0;
    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {3}};
    params0["X"] = migraphx::argument(input_fixed_shape0, input_data.data());
    auto result  = p.eval(params0).back();
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = input_data;
    std::transform(
        gold.begin(), gold.end(), gold.begin(), [](float n) -> float { return cosf(n); });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(bla)
{
    // migraphx::program p1;
    // auto* mm   = p1.get_main_module();
    // migraphx::shape a_shape{migraphx::shape::half_type, {1, 2, 240, 256}};
    // migraphx::shape b_shape{migraphx::shape::half_type, {1, 2, 256, 240}};
    // migraphx::shape b1_shape{migraphx::shape::half_type, {1, 2, 240, 256}};
    // auto a     = mm->add_parameter("q", a_shape);  // [1, 256, 240]
    // auto b     = mm->add_parameter("k", b_shape);  // [1, 256, 240]
    // auto b1    = mm->add_parameter("v", b1_shape); // [1, 240, 256]
    // auto gemm1 = mm->add_instruction(
    //     migraphx::make_op("dot"), a, b); // [1, 240, 256] x [1, 256, 240] = [1, 240, 240]
    // auto rmax = mm->add_instruction(migraphx::make_op("reduce_max", {{"axes", {2}}}),
    //                                 gemm1); // [1, 240, 1]
    // rmax      = mm->add_instruction(
    //     migraphx::make_op("multibroadcast", {{"out_lens", gemm1->get_shape().lens()}}),
    //     rmax);                                                         // [1, 240, 240]
    // auto sub  = mm->add_instruction(migraphx::make_op("sub"), gemm1, rmax); // [1, 240, 240]
    // auto exp  = mm->add_instruction(migraphx::make_op("exp"), sub);         // [1, 240, 240]
    // auto rsum = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}),
    //                                 exp); // [1, 240, 1]
    // rsum      = mm->add_instruction(
    //     migraphx::make_op("multibroadcast", {{"out_lens", gemm1->get_shape().lens()}}),
    //     rsum);                                                        // [1, 240, 240]
    // auto div   = mm->add_instruction(migraphx::make_op("div"), exp, rsum); // [1, 240, 240]
    // auto gemm2 = mm->add_instruction(
    //     migraphx::make_op("dot"), div, b1); // [1, 240, 240] x [1, 240, 256] = [1, 240, 256]
    // mm->add_return({gemm2});

    // std::cout << p1 << std::endl;
    // p1.compile(migraphx::make_target("gpu"));
    // std::cout << p1 << std::endl;

    migraphx::shape s_3d{migraphx::shape::half_type, {1, 256, 240}};
    migraphx::shape st_3d{migraphx::shape::half_type, {1, 240, 256}};
    migraphx::program p1;
    auto* mm   = p1.get_main_module();
    auto a     = mm->add_parameter("q", s_3d);  // [1, 256, 240]
    auto b     = mm->add_parameter("k", s_3d);  // [1, 256, 240]
    auto b1    = mm->add_parameter("v", st_3d); // [1, 240, 256]
    a          = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}),
                            a); // [1, 240, 256]
    auto gemm1 = mm->add_instruction(
        migraphx::make_op("dot"), a, b); // [1, 240, 256] x [1, 256, 240] = [1, 240, 240]
    auto rmax = mm->add_instruction(migraphx::make_op("reduce_max", {{"axes", {2}}}),
                                    gemm1); // [1, 240, 1]
    rmax      = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", gemm1->get_shape().lens()}}),
        rmax);                                                         // [1, 240, 240]
    auto sub  = mm->add_instruction(migraphx::make_op("sub"), gemm1, rmax); // [1, 240, 240]
    auto exp  = mm->add_instruction(migraphx::make_op("exp"), sub);         // [1, 240, 240]
    auto rsum = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}),
                                    exp); // [1, 240, 1]
    rsum      = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", gemm1->get_shape().lens()}}),
        rsum);                                                        // [1, 240, 240]
    auto div   = mm->add_instruction(migraphx::make_op("div"), exp, rsum); // [1, 240, 240]
    auto gemm2 = mm->add_instruction(
        migraphx::make_op("dot"), div, b1); // [1, 240, 240] x [1, 240, 256] = [1, 240, 256]
    mm->add_return({gemm2});

    std::cout << p1 << std::endl;
    p1.compile(migraphx::make_target("gpu"));
    std::cout << p1 << std::endl;
}

TEST_CASE(bla2)
{
    migraphx::shape q_shape{migraphx::shape::half_type, {1, 64, 16}};
    migraphx::shape k_shape{migraphx::shape::half_type, {1, 256, 16}};
    migraphx::shape v_shape{migraphx::shape::half_type, {1, 256, 32}};

    migraphx::program p1;
    auto* mm = p1.get_main_module();
    auto a   = mm->add_parameter("q", q_shape);
    auto b   = mm->add_parameter("k", k_shape);
    b        = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}),
                            b); // {1, 16, 256}
    auto b1  = mm->add_parameter("v", v_shape);

    auto gemm1 = mm->add_instruction(migraphx::make_op("dot"), a, b); // {1, 64, 256}
    auto rmax  = mm->add_instruction(migraphx::make_op("reduce_max", {{"axes", {2}}}),
                                    gemm1); // {1, 64, 1}
    rmax = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {1, 64, 256}}}),
                               rmax);
    auto sub  = mm->add_instruction(migraphx::make_op("sub"), gemm1, rmax); // {1, 64, 256}
    auto exp  = mm->add_instruction(migraphx::make_op("exp"), sub);         // {1, 64, 256}
    auto rsum = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}),
                                    exp); // {1, 64, 1}
    rsum = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {1, 64, 256}}}),
                               rsum);                                      // {1, 64, 256}
    auto div   = mm->add_instruction(migraphx::make_op("div"), exp, rsum); // {1, 64, 256}
    auto gemm2 = mm->add_instruction(migraphx::make_op("dot"), div, b1);   // {1, 64, 32}
    mm->add_return({gemm2});

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    migraphx::parameter_map pm;
    std::vector<migraphx::half> q_data(q_shape.elements());
    std::generate(q_data.begin(), q_data.end(), [&]() { return dist(rng); });
    std::vector<migraphx::half> k_data(k_shape.elements());
    std::generate(k_data.begin(), k_data.end(), [&]() { return dist(rng); });
    std::vector<migraphx::half> v_data(v_shape.elements());
    std::generate(v_data.begin(), v_data.end(), [&]() { return dist(rng); });
    pm["q"]    = migraphx::argument(q_shape, q_data.data());
    pm["k"]    = migraphx::argument(k_shape, k_data.data());
    pm["v"]    = migraphx::argument(v_shape, v_data.data());
    auto ref_p = p1;
    auto gpu_p = p1;
    // std::cout << p1 << std::endl;

    ref_p.compile(migraphx::make_target("ref"));
    auto ref_out = ref_p.eval(pm).back();
    std::vector<migraphx::half> ref_out_data(ref_out.get_shape().elements());
    ref_out.visit([&](auto output) { ref_out_data.assign(output.begin(), output.end()); });
    std::cout << "ref_out_data: \n";
    for(auto i = 0; i < 20; i++)
    {
        std::cout << static_cast<float>(ref_out_data[i]) << " ";
    }
    std::cout << std::endl;

    migraphx::compile_options options;
    options.offload_copy = true;
    options.exhaustive_tune = true;
    gpu_p.compile(migraphx::make_target("gpu"), options);
    // std::cout << gpu_p << std::endl;
    auto gpu_out = gpu_p.eval(pm).back();
    std::vector<migraphx::half> gpu_out_data(gpu_out.get_shape().elements());
    gpu_out.visit([&](auto output) { gpu_out_data.assign(output.begin(), output.end()); });
    std::cout << "gpu_out_data: \n";
    for(auto i = 0; i < 20; i++)
    {
        std::cout << static_cast<float>(gpu_out_data[i]) << " ";
    }
    std::cout << std::endl;
    EXPECT(migraphx::verify::verify_rms_range(gpu_out_data, ref_out_data));
}

TEST_CASE(bla3)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape a_shape{migraphx::shape::half_type, {1, 12, 128, 256}};
    migraphx::shape b_shape{migraphx::shape::half_type, {1, 12, 512, 256}};
    migraphx::shape b1_shape{migraphx::shape::half_type, {1, 12, 512, 32}};

    auto a  = mm->add_parameter("1", a_shape);
    auto b  = mm->add_parameter("2", b_shape);
    auto b1 = mm->add_parameter("3", b1_shape);
    b = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), b);
    auto gemm1 = mm->add_instruction(migraphx::make_op("dot"), a, b);
    auto rmax  = mm->add_instruction(migraphx::make_op("reduce_max", {{"axes", {3}}}), gemm1);
    rmax = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", gemm1->get_shape().lens()}}),
                               rmax);
    auto sub  = mm->add_instruction(migraphx::make_op("sub"), gemm1, rmax);
    auto exp  = mm->add_instruction(migraphx::make_op("exp"), sub);
    auto rsum = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {3}}}), exp);
    rsum = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", exp->get_shape().lens()}}),
                               rsum);
    auto div   = mm->add_instruction(migraphx::make_op("div"), exp, rsum);
    auto gemm2 = mm->add_instruction(migraphx::make_op("dot"), div, b1);
    mm->add_return({gemm2});

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    migraphx::parameter_map pm;
    std::vector<migraphx::half> q_data(a_shape.elements());
    std::generate(q_data.begin(), q_data.end(), [&]() { return dist(rng); });
    std::vector<migraphx::half> k_data(b_shape.elements());
    std::generate(k_data.begin(), k_data.end(), [&]() { return dist(rng); });
    std::vector<migraphx::half> v_data(b1_shape.elements());
    std::generate(v_data.begin(), v_data.end(), [&]() { return dist(rng); });
    pm["1"]    = migraphx::argument(a_shape, q_data.data());
    pm["2"]    = migraphx::argument(b_shape, k_data.data());
    pm["3"]    = migraphx::argument(b1_shape, v_data.data());
    auto ref_p = p;
    auto gpu_p = p;
    // std::cout << p1 << std::endl;

    ref_p.compile(migraphx::make_target("ref"));
    auto ref_out = ref_p.eval(pm).back();
    std::vector<migraphx::half> ref_out_data(ref_out.get_shape().elements());
    ref_out.visit([&](auto output) { ref_out_data.assign(output.begin(), output.end()); });
    std::cout << "ref_out_data: \n";
    for(auto i = 0; i < 20; i++)
    {
        std::cout << static_cast<float>(ref_out_data[i]) << " ";
    }
    std::cout << std::endl;

    migraphx::compile_options options;
    options.offload_copy = true;
    gpu_p.compile(migraphx::make_target("gpu"), options);
    // std::cout << gpu_p << std::endl;
    auto gpu_out = gpu_p.eval(pm).back();
    std::vector<migraphx::half> gpu_out_data(gpu_out.get_shape().elements());
    gpu_out.visit([&](auto output) { gpu_out_data.assign(output.begin(), output.end()); });
    std::cout << "gpu_out_data: \n";
    for(auto i = 0; i < 20; i++)
    {
        std::cout << static_cast<float>(gpu_out_data[i]) << " ";
    }
    std::cout << std::endl;
    EXPECT(migraphx::verify::verify_rms_range(gpu_out_data, ref_out_data));
}

TEST_CASE(combine_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape o_shape{migraphx::shape::half_type, {1, 2, 240, 256}};
    migraphx::shape lse_shape{migraphx::shape::float_type, {1, 2, 240, 1}};

    auto o                = mm->add_parameter("o", o_shape);
    auto lse              = mm->add_parameter("lse", lse_shape);
    constexpr auto g_axis = 1;

    auto lse_max = mm->add_instruction(migraphx::make_op("reduce_max", {{"axes", {g_axis}}}), lse);
    auto lse_max_bcast = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", lse->get_shape().lens()}}), lse_max);
    auto lse_sub = mm->add_instruction(migraphx::make_op("sub"), lse, lse_max_bcast);
    auto lse_exp = mm->add_instruction(migraphx::make_op("exp"), lse_sub);
    auto lse_sum =
        mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {g_axis}}}), lse_exp);
    auto lse_sum_bcast = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", lse_exp->get_shape().lens()}}), lse_sum);

    auto scale       = mm->add_instruction(migraphx::make_op("div"), lse_exp, lse_sum_bcast);
    auto scale_bcast = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", o->get_shape().lens()}}), scale);
    auto scale_converted = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", o->get_shape().type()}}), scale_bcast);

    auto scaled_r = mm->add_instruction(migraphx::make_op("mul"), o, scale_converted);
    auto final_output_o =
        mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {g_axis}}}), scaled_r);
    auto final_squeezed_o =
        mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {g_axis}}}), final_output_o);

    mm->add_return({final_squeezed_o});
    std::cout << p << std::endl;
    p.compile(migraphx::make_target("gpu"));
    std::cout << p << std::endl;
}
