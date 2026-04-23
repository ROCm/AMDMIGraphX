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
#include <filesystem>

#include <test.hpp>

#include <cmath>
#include <iostream>
#include <limits>

struct vector_stats
{
    float min_val;
    float max_val;
    float avg;
    float variance;
    float stddev;
};

vector_stats compute_half_vector_stats(const std::vector<migraphx::half>& data)
{
    if(data.empty())
        return {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    double sum  = 0.0;
    double sum2 = 0.0;
    float lo    = std::numeric_limits<float>::max();
    float hi    = std::numeric_limits<float>::lowest();

    for(const auto& h : data)
    {
        float v = static_cast<float>(h);
        if(v < lo)
            lo = v;
        if(v > hi)
            hi = v;
        sum += v;
        sum2 += static_cast<double>(v) * v;
    }

    auto n    = static_cast<double>(data.size());
    auto mean = static_cast<float>(sum / n);
    auto var  = static_cast<float>(sum2 / n - (sum / n) * (sum / n));
    auto sd   = std::sqrt(var);

    return {lo, hi, mean, var, sd};
}

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
    options.offload_copy    = true;
    options.exhaustive_tune = false;
    gpu_p.compile(migraphx::make_target("gpu"), options);
    std::cout << gpu_p << std::endl;
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
    rmax       = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", gemm1->get_shape().lens()}}), rmax);
    auto sub  = mm->add_instruction(migraphx::make_op("sub"), gemm1, rmax);
    auto exp  = mm->add_instruction(migraphx::make_op("exp"), sub);
    auto rsum = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {3}}}), exp);
    rsum      = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", exp->get_shape().lens()}}), rsum);
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

TEST_CASE(make_combinations)
{
    const std::size_t batch = 2;
    const std::size_t nhead = 4;

    std::vector<size_t> seqlens_q{1};
    std::vector<size_t> seqlens_k{512, 1024, 2048, 4096};
    std::vector<size_t> hdims_q{32, 48, 64, 80, 96, 128, 192, 256};
    std::vector<size_t> hdims_v{32, 48, 64, 80, 96, 128, 192, 256};

    for(const auto& seqlen_q : seqlens_q)
    {
        for(const auto& seqlen_k : seqlens_k)
        {
            for(const auto& hdim_q : hdims_q)
            {
                for(const auto& hdim_v : hdims_v)
                {
                    // if(seqlen_q != 512 or seqlen_k != 512 or hdim_q != 32 or hdim_v != 32) {
                    //     continue;
                    // }
                    const std::size_t M = seqlen_q; // seqlen_q
                    const std::size_t N = seqlen_k; // seqlen_k
                    const std::size_t K = hdim_q;   // hdim_q
                    const std::size_t O = hdim_v;   // hdim_v
                    migraphx::program p;
                    auto* mm = p.get_main_module();
                    migraphx::shape a_shape{migraphx::shape::half_type, {batch, nhead, M, K}};
                    migraphx::shape b_shape{migraphx::shape::half_type, {batch, nhead, N, K}};
                    migraphx::shape b1_shape{migraphx::shape::half_type, {batch, nhead, N, O}};

                    auto a  = mm->add_parameter("1", a_shape);
                    auto b  = mm->add_parameter("2", b_shape);
                    auto b1 = mm->add_parameter("3", b1_shape);
                    b       = mm->add_instruction(
                        migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), b);
                    auto gemm1 = mm->add_instruction(migraphx::make_op("dot"), a, b);
                    auto rmax  = mm->add_instruction(
                        migraphx::make_op("reduce_max", {{"axes", {3}}}), gemm1);
                    rmax = mm->add_instruction(
                        migraphx::make_op("multibroadcast",
                                          {{"out_lens", gemm1->get_shape().lens()}}),
                        rmax);
                    auto sub = mm->add_instruction(migraphx::make_op("sub"), gemm1, rmax);
                    auto exp = mm->add_instruction(migraphx::make_op("exp"), sub);
                    auto rsum =
                        mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {3}}}), exp);
                    rsum = mm->add_instruction(
                        migraphx::make_op("multibroadcast",
                                          {{"out_lens", exp->get_shape().lens()}}),
                        rsum);
                    auto div   = mm->add_instruction(migraphx::make_op("div"), exp, rsum);
                    auto gemm2 = mm->add_instruction(migraphx::make_op("dot"), div, b1);
                    mm->add_return({gemm2});

                    std::string backend = "ck";
                    migraphx::compile_options options;
                    options.exhaustive_tune = backend == "mlir" ? false : true;

                    std::stringstream ss;
                    ss << backend << "_" << batch << "_" << nhead << "_" << M << "_" << N << "_"
                       << K << "_" << O << ".mxr";
                    std::string check_filename = "saved_models/" + backend + "_models/" + ss.str();
                    if(std::filesystem::exists(check_filename))
                    {
                        std::cout << "Skipping, file already exists: " << check_filename
                                  << std::endl;
                        continue;
                    }
                    std::string output_filename = "saved_models/" + backend + "_models/" + ss.str();
                    std::cout << "Compiling " << output_filename << std::endl;
                    auto start_time = std::chrono::high_resolution_clock::now();
                    p.compile(migraphx::make_target("gpu"), options);
                    auto end_time = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> elapsed = end_time - start_time;
                    std::cout << "Finished compiling " << output_filename << " in "
                              << elapsed.count() << " seconds" << std::endl;
                    std::cout << p << std::endl;
                    migraphx::save(p, output_filename);
                }
            }
        }
    }
}

TEST_CASE(test_combinations)
{
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    const std::size_t batch = 2;
    const std::size_t nhead = 4;
    std::vector<size_t> seqlens_q{1, 512, 1024, 2048, 4096};
    std::vector<size_t> seqlens_k{512, 1024, 2048, 4096};
    std::vector<size_t> hdims_q{32, 48, 64, 80, 96, 128, 192, 256};
    std::vector<size_t> hdims_v{32, 48, 64, 80, 96, 128, 192, 256};

    auto test_body = [&](size_t seqlen_q, size_t seqlen_k, size_t hdim_q, size_t hdim_v) {
        try
        {
            const std::size_t M = seqlen_q; // seqlen_q
            const std::size_t N = seqlen_k; // seqlen_k
            const std::size_t K = hdim_q;   // hdim_q
            const std::size_t O = hdim_v;   // hdim_v
            migraphx::program p;
            auto* mm = p.get_main_module();
            migraphx::shape a_shape{migraphx::shape::half_type, {batch, nhead, M, K}};
            migraphx::shape b_shape{migraphx::shape::half_type, {batch, nhead, N, K}};
            migraphx::shape b1_shape{migraphx::shape::half_type, {batch, nhead, N, O}};

            auto a  = mm->add_parameter("1", a_shape);
            auto b  = mm->add_parameter("2", b_shape);
            auto b1 = mm->add_parameter("3", b1_shape);
            b = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}),
                                    b);
            auto gemm1 = mm->add_instruction(migraphx::make_op("dot"), a, b);
            auto rmax =
                mm->add_instruction(migraphx::make_op("reduce_max", {{"axes", {3}}}), gemm1);
            rmax = mm->add_instruction(
                migraphx::make_op("multibroadcast", {{"out_lens", gemm1->get_shape().lens()}}),
                rmax);
            auto sub  = mm->add_instruction(migraphx::make_op("sub"), gemm1, rmax);
            auto exp  = mm->add_instruction(migraphx::make_op("exp"), sub);
            auto rsum = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {3}}}), exp);
            rsum      = mm->add_instruction(
                migraphx::make_op("multibroadcast", {{"out_lens", exp->get_shape().lens()}}), rsum);
            auto div   = mm->add_instruction(migraphx::make_op("div"), exp, rsum);
            auto gemm2 = mm->add_instruction(migraphx::make_op("dot"), div, b1);
            mm->add_return({gemm2});

            auto gpu_p = p;
            migraphx::compile_options options;
            options.exhaustive_tune = false;
            options.offload_copy    = true;
            gpu_p.compile(migraphx::make_target("gpu"), options);
            p.compile(migraphx::make_target("ref"));

            migraphx::parameter_map pm;
            std::vector<migraphx::half> q_data(a_shape.elements());
            std::generate(q_data.begin(), q_data.end(), [&]() { return dist(rng); });
            std::vector<migraphx::half> k_data(b_shape.elements());
            std::generate(k_data.begin(), k_data.end(), [&]() { return dist(rng); });
            std::vector<migraphx::half> v_data(b1_shape.elements());
            std::generate(v_data.begin(), v_data.end(), [&]() { return dist(rng); });
            pm["1"] = migraphx::argument(a_shape, q_data.data());
            pm["2"] = migraphx::argument(b_shape, k_data.data());
            pm["3"] = migraphx::argument(b1_shape, v_data.data());

            auto ref_out = p.eval(pm).back();
            std::vector<migraphx::half> ref_out_data(ref_out.get_shape().elements());
            ref_out.visit([&](auto output) { ref_out_data.assign(output.begin(), output.end()); });
            std::cout << "Ref out data: ";
            for(auto i = 0; i < 35; ++i) {
                std::cout << ref_out_data[i] << " ";
            }
            std::cout << std::endl;

            auto gpu_out = gpu_p.eval(pm).back();
            std::vector<migraphx::half> gpu_out_data(gpu_out.get_shape().elements());
            gpu_out.visit([&](auto output) { gpu_out_data.assign(output.begin(), output.end()); });
            std::cout << "GPU out data: ";
            for(auto i = 0; i < 35; ++i) {
                std::cout << gpu_out_data[i] << " ";
            }
            std::cout << std::endl;
            bool passed = migraphx::verify::verify_rms_range(gpu_out_data, ref_out_data);
            if(!passed)
            {
                std::cout << "Failed for seqlen_q: " << seqlen_q << ", seqlen_k: " << seqlen_k
                          << ", hdim_q: " << hdim_q << ", hdim_v: " << hdim_v << std::endl;
                auto ref_stats = compute_half_vector_stats(ref_out_data);
                auto gpu_stats = compute_half_vector_stats(gpu_out_data);
                std::cout << "Ref stats: "
                          << "min_val: " << ref_stats.min_val << ", max_val: " << ref_stats.max_val
                          << ", avg: " << ref_stats.avg << ", variance: " << ref_stats.variance
                          << ", stddev: " << ref_stats.stddev << std::endl;
                std::cout << "Gpu stats: "
                          << "min_val: " << gpu_stats.min_val << ", max_val: " << gpu_stats.max_val
                          << ", avg: " << gpu_stats.avg << ", variance: " << gpu_stats.variance
                          << ", stddev: " << gpu_stats.stddev << std::endl;
            }
            CHECK(passed);
        }
        catch(...)
        {
            std::cout << "Error for seqlen_q: " << seqlen_q << ", seqlen_k: " << seqlen_k
                      << ", hdim_q: " << hdim_q << ", hdim_v: " << hdim_v << std::endl;
        }
    };

    auto num_combinations = seqlens_q.size() * seqlens_k.size() * hdims_q.size() * hdims_v.size();
    auto iteration        = 1ul;
    // for(const auto& seqlen_q : seqlens_q)
    // {
    //     for(const auto& seqlen_k : seqlens_k)
    //     {
    //         for(const auto& hdim_q : hdims_q)
    //         {
    //             for(const auto& hdim_v : hdims_v)
    //             {
    //                 std::cout << "Iteration " << iteration++ << "/" << num_combinations <<
    //                 std::endl; test_body(seqlen_q, seqlen_k, hdim_q, hdim_v);
    //             }
    //         }
    //     }
    // }
    struct test_dims
    {
        size_t seqlen_q;
        size_t seqlen_k;
        size_t hdim_q;
        size_t hdim_v;
    };
    const std::vector<test_dims> test_dims_list = {
        {1, 512, 64, 64},      {1, 512, 80, 32},      {1, 512, 80, 64},      {1, 512, 80, 96},
        {1, 512, 96, 48},      {1, 512, 96, 80},      {1, 512, 96, 96},      {1, 512, 96, 256},
        {1, 512, 128, 32},     {1, 512, 128, 48},     {1, 512, 128, 64},     {1, 512, 128, 80},
        {1, 512, 128, 96},     {1, 512, 128, 128},    {1, 512, 128, 192},    {1, 512, 128, 256},
        {1, 512, 192, 32},     {1, 512, 192, 48},     {1, 512, 192, 64},     {1, 512, 192, 80},
        {1, 512, 192, 96},     {1, 512, 192, 128},    {1, 512, 192, 192},    {1, 512, 192, 256},
        {1, 512, 256, 32},     {1, 512, 256, 48},     {1, 512, 256, 64},     {1, 512, 256, 80},
        {1, 512, 256, 96},     {1, 512, 256, 128},    {1, 512, 256, 192},    {1, 512, 256, 256},
        {1, 1024, 80, 32},     {1, 1024, 96, 32},     {1, 1024, 96, 64},     {1, 1024, 96, 80},
        {1, 1024, 96, 192},    {1, 1024, 96, 256},    {1, 1024, 128, 32},    {1, 1024, 128, 48},
        {1, 1024, 128, 64},    {1, 1024, 128, 80},    {1, 1024, 128, 128},   {1, 1024, 128, 192},
        {1, 1024, 128, 256},   {1, 1024, 192, 32},    {1, 1024, 192, 48},    {1, 1024, 192, 64},
        {1, 1024, 192, 80},    {1, 1024, 192, 96},    {1, 1024, 192, 128},   {1, 1024, 192, 192},
        {1, 1024, 192, 256},   {1, 1024, 256, 32},    {1, 1024, 256, 48},    {1, 1024, 256, 64},
        {1, 1024, 256, 80},    {1, 1024, 256, 96},    {1, 1024, 256, 128},   {1, 1024, 256, 192},
        {1, 1024, 256, 256},   {1, 2048, 80, 32},     {1, 2048, 80, 64},     {1, 2048, 80, 80},
        {1, 2048, 80, 96},     {1, 2048, 96, 32},     {1, 2048, 96, 48},     {1, 2048, 96, 64},
        {1, 2048, 96, 80},     {1, 2048, 96, 96},     {1, 2048, 96, 192},    {1, 2048, 128, 48},
        {1, 2048, 128, 64},    {1, 2048, 128, 80},    {1, 2048, 128, 96},    {1, 2048, 128, 128},
        {1, 2048, 128, 256},   {1, 2048, 192, 32},    {1, 2048, 192, 48},    {1, 2048, 192, 64},
        {1, 2048, 192, 80},    {1, 2048, 192, 96},    {1, 2048, 192, 128},   {1, 2048, 192, 192},
        {1, 2048, 192, 256},   {1, 2048, 256, 32},    {1, 2048, 256, 48},    {1, 2048, 256, 64},
        {1, 2048, 256, 80},    {1, 2048, 256, 96},    {1, 2048, 256, 128},   {1, 2048, 256, 192},
        {1, 2048, 256, 256},   {1, 4096, 64, 80},     {1, 4096, 80, 32},     {1, 4096, 80, 96},
        {1, 4096, 96, 32},     {1, 4096, 96, 48},     {1, 4096, 96, 64},     {1, 4096, 96, 96},
        {1, 4096, 96, 256},    {1, 4096, 128, 32},    {1, 4096, 128, 48},    {1, 4096, 128, 64},
        {1, 4096, 128, 80},    {1, 4096, 128, 96},    {1, 4096, 128, 128},   {1, 4096, 128, 192},
        {1, 4096, 128, 256},   {1, 4096, 192, 32},    {1, 4096, 192, 48},    {1, 4096, 192, 64},
        {1, 4096, 192, 80},    {1, 4096, 192, 96},    {1, 4096, 192, 128},   {1, 4096, 192, 192},
        {1, 4096, 192, 256},   {1, 4096, 256, 32},    {1, 4096, 256, 48},    {1, 4096, 256, 64},
        {1, 4096, 256, 80},    {1, 4096, 256, 96},    {1, 4096, 256, 128},   {1, 4096, 256, 192},
        {1, 4096, 256, 256},   {512, 512, 192, 48},   {512, 1024, 192, 96},  {512, 2048, 192, 32},
        {512, 2048, 192, 48},  {512, 2048, 192, 80},  {512, 2048, 192, 192}, {512, 4096, 192, 64},
        {512, 4096, 256, 48},  {512, 4096, 256, 80},  {512, 4096, 256, 128}, {1024, 1024, 192, 80},
        {1024, 2048, 256, 32}, {1024, 4096, 192, 32}, {1024, 4096, 192, 48}, {1024, 4096, 192, 64},
        {1024, 4096, 192, 80}, {1024, 4096, 192, 96}, {1024, 4096, 192, 128}};

    for(const auto& test_dims : test_dims_list)
    {
        std::cout << "Iteration " << iteration++ << "/" << test_dims_list.size() << std::endl;
        test_body(test_dims.seqlen_q, test_dims.seqlen_k, test_dims.hdim_q, test_dims.hdim_v);
    }
}
