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
#include "migraphx/op/builder/insert.hpp"
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
#include <fstream>
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
    gemm2      = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), gemm2);
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
    // auto ref_out = ref_p.eval(pm).back();
    // std::vector<migraphx::half> ref_out_data(ref_out.get_shape().elements());
    // ref_out.visit([&](auto output) { ref_out_data.assign(output.begin(), output.end()); });
    // std::cout << "ref_out_data: \n";
    // for(auto i = 0; i < 20; i++)
    // {
    //     std::cout << static_cast<float>(ref_out_data[i]) << " ";
    // }
    // std::cout << std::endl;

    migraphx::compile_options options;
    options.offload_copy = true;
    gpu_p.compile(migraphx::make_target("gpu"), options);
    std::cout << gpu_p << std::endl;
    // auto gpu_out = gpu_p.eval(pm).back();
    // std::vector<migraphx::half> gpu_out_data(gpu_out.get_shape().elements());
    // gpu_out.visit([&](auto output) { gpu_out_data.assign(output.begin(), output.end()); });
    // std::cout << "gpu_out_data: \n";
    // for(auto i = 0; i < 20; i++)
    // {
    //     std::cout << static_cast<float>(gpu_out_data[i]) << " ";
    // }
    // std::cout << std::endl;
    // EXPECT(migraphx::verify::verify_rms_range(gpu_out_data, ref_out_data));
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
    setenv("MIGRAPHX_FLASH_DECODING_ENABLED", "1", 1);
    std::string backend = "mlir";
    if(backend == "ck")
    {
        setenv("MIGRAPHX_ENABLE_CK", "1", 1);
    }
    const std::size_t batch = 2;
    const std::size_t nhead = 4;

    const char* num_split_chars = std::getenv("MIGRAPHX_FLASH_DECODING_NUM_SPLITS");
    if(num_split_chars == nullptr)
    {
        throw std::runtime_error("MIGRAPHX_FLASH_DECODING_NUM_SPLITS is not set");
    }
    std::size_t num_split = std::stoull(num_split_chars);
    std::vector<size_t> seqlens_q{1, 16, 32};
    std::vector<size_t> seqlens_k{1024, 2048, 4096};
    std::vector<size_t> hdims_q{32, 48, 64, 80, 96, 128, 192, 256};
    std::vector<size_t> hdims_v{32, 48, 64, 80, 96, 128, 192, 256};
    auto num_combinations = seqlens_q.size() * seqlens_k.size() * hdims_q.size() * hdims_v.size();
    auto iteration        = 1ul;

    for(const auto& seqlen_q : seqlens_q)
    {
        for(const auto& seqlen_k : seqlens_k)
        {
            if(seqlen_k / num_split < 32)
            {
                std::cout << "Skipping seqlen_k = " << seqlen_k << std::endl;
                iteration += hdims_q.size() * hdims_v.size();
                continue;
            }
            for(const auto& hdim_q : hdims_q)
            {
                for(const auto& hdim_v : hdims_v)
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

                    migraphx::compile_options options;
                    options.exhaustive_tune = backend == "mlir" ? false : true;
                    options.exhaustive_tune = false;

                    std::cout << "Iteration " << iteration++ << "/" << num_combinations
                              << std::endl;

                    std::stringstream ss;
                    ss << num_split << "_" << batch << "_" << nhead << "_" << M << "_" << N << "_"
                       << K << "_" << O << ".mxr";
                    std::string check_filename =
                        "saved_models/mlir_main_mgx_combine_models/" + ss.str();
                    if(std::filesystem::exists(check_filename))
                    {
                        std::cout << "Skipping, file already exists: " << check_filename
                                  << std::endl;
                        continue;
                    }
                    std::string output_filename =
                        "saved_models/mlir_main_mgx_combine_models/" + ss.str();
                    std::cout << "Compiling " << output_filename << std::endl;
                    auto start_time = std::chrono::high_resolution_clock::now();
                    p.compile(migraphx::make_target("gpu"), options);
                    auto end_time = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> elapsed = end_time - start_time;
                    std::cout << p << std::endl;
                    std::cout << "Finished compiling " << output_filename << " in "
                              << elapsed.count() << " seconds" << std::endl;
                    migraphx::save(p, output_filename);
                }
            }
        }
    }
}

TEST_CASE(test_combinations)
{
    std::ofstream fail_log("test_combinations_failures.log", std::ios::app);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    const std::size_t batch = 2;
    const std::size_t nhead = 4;
    std::vector<size_t> seqlens_q{1, 16, 32};
    std::vector<size_t> seqlens_k{1024, 2048, 4096};
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
            std::cout << "GPU program: " << std::endl;
            std::cout << gpu_p << std::endl;
            p.compile(migraphx::make_target("ref"));
            std::cout << "Ref program: " << std::endl;
            std::cout << p << std::endl;

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
            for(auto i = 0; i < 35; ++i)
            {
                std::cout << ref_out_data[i] << " ";
            }
            std::cout << std::endl;

            auto gpu_out = gpu_p.eval(pm).back();
            std::vector<migraphx::half> gpu_out_data(gpu_out.get_shape().elements());
            gpu_out.visit([&](auto output) { gpu_out_data.assign(output.begin(), output.end()); });
            std::cout << "GPU out data: ";
            for(auto i = 0; i < 35; ++i)
            {
                std::cout << gpu_out_data[i] << " ";
            }
            std::cout << std::endl;
            bool passed = migraphx::verify::verify_rms_range(gpu_out_data, ref_out_data);
            if(!passed)
            {
                auto ref_stats     = compute_half_vector_stats(ref_out_data);
                auto gpu_stats     = compute_half_vector_stats(gpu_out_data);
                auto write_failure = [&](std::ostream& os) {
                    os << "Failed for seqlen_q: " << seqlen_q << ", seqlen_k: " << seqlen_k
                       << ", hdim_q: " << hdim_q << ", hdim_v: " << hdim_v << "\n";
                    os << "Ref stats: "
                       << "min_val: " << ref_stats.min_val << ", max_val: " << ref_stats.max_val
                       << ", avg: " << ref_stats.avg << ", variance: " << ref_stats.variance
                       << ", stddev: " << ref_stats.stddev << "\n";
                    os << "Gpu stats: "
                       << "min_val: " << gpu_stats.min_val << ", max_val: " << gpu_stats.max_val
                       << ", avg: " << gpu_stats.avg << ", variance: " << gpu_stats.variance
                       << ", stddev: " << gpu_stats.stddev << "\n";
                };
                write_failure(std::cout);
                write_failure(fail_log);
            }
            else
            {
                std::cout << "Passed\n";
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
    for(const auto& seqlen_q : seqlens_q)
    {
        for(const auto& seqlen_k : seqlens_k)
        {
            for(const auto& hdim_q : hdims_q)
            {
                for(const auto& hdim_v : hdims_v)
                {
                    std::cout << "Iteration " << iteration++ << "/" << num_combinations
                              << std::endl;
                    test_body(seqlen_q, seqlen_k, hdim_q, hdim_v);
                }
            }
        }
    }
}

struct gqa_program_params
{
    size_t batch;
    size_t nhead;
    size_t nhead_kv;
    size_t seqlen;
    size_t seqlen_old;
    size_t head_size;
    int local_window_size;
    float scale;
    bool do_rotary;
    bool rotary_interleaved;
};
migraphx::program make_gqa_program(gqa_program_params p, bool ck)
{
    using namespace migraphx;
    program prog;
    auto* mm = prog.get_main_module();

    auto qkv_hidden_size = p.head_size * (p.nhead + 2 * p.nhead_kv);
    shape qkv_shape{shape::half_type, {p.batch, p.seqlen, qkv_hidden_size}};
    shape kv_old_shape{shape::half_type, {p.batch, p.nhead_kv, p.seqlen_old, p.head_size}};
    shape slk_shape{shape::int32_type, {p.batch}};
    shape trig_cache_shape{shape::half_type, {p.seqlen_old, p.head_size / 2}};

    auto qkv       = mm->add_parameter("qkv", qkv_shape);
    auto k_old     = mm->add_parameter("k_old", kv_old_shape);
    auto v_old     = mm->add_parameter("v_old", kv_old_shape);
    auto slk       = mm->add_parameter("slk", slk_shape);
    auto cos_cache = mm->add_parameter("cos_cache", trig_cache_shape);
    auto sin_cache = mm->add_parameter("sin_cache", trig_cache_shape);

    std::vector<size_t> bsnh{p.batch, p.seqlen, p.nhead + 2 * p.nhead_kv, p.head_size};
    auto transposed_qkv = mm->add_instruction(make_op("reshape", {{"dims", bsnh}}), qkv);
    transposed_qkv =
        mm->add_instruction(make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), transposed_qkv);
    auto qk = mm->add_instruction(
        make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {p.nhead + p.nhead_kv}}}),
        transposed_qkv);
    auto v = mm->add_instruction(make_op("slice",
                                         {{"axes", {1}},
                                          {"starts", {p.nhead + p.nhead_kv}},
                                          {"ends", {p.nhead + 2 * p.nhead_kv}}}),
                                 transposed_qkv);

    if(p.do_rotary)
    {
        auto pos_ids = slk;
        if(not ck)
        {
            if(p.seqlen > 1)
            {
                pos_ids = mm->add_literal(literal{shape{pos_ids->get_shape().type(), {1}}, {0}});
            }
            qk = op::builder::add("rotary_embedding",
                                  *mm,
                                  {qk, pos_ids, cos_cache, sin_cache},
                                  {{"interleaved", p.rotary_interleaved}})
                     .at(0);
        }
        else
        {
            if(p.seqlen > 1)
            {
                pos_ids =
                    mm->add_literal(literal{pos_ids->get_shape(), std::vector<int>(p.batch, 0)});
            }
            qk = mm->add_instruction(
                make_op("rotary_embedding", {{"interleaved", p.rotary_interleaved}}),
                qk,
                pos_ids,
                cos_cache,
                sin_cache);
        }
    }

    auto q = mm->add_instruction(
        make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {p.nhead}}}), qk);
    auto k = mm->add_instruction(
        make_op("slice", {{"axes", {1}}, {"starts", {p.nhead}}, {"ends", {p.nhead + p.nhead_kv}}}),
        qk);
    std::vector<instruction_ref> concat_k_inputs{k, slk, k_old};
    std::vector<instruction_ref> concat_v_inputs{v, slk, v_old};

    k = mm->add_instruction(make_op("concat_past_present", {{"kv_num_heads", p.nhead_kv}}),
                            concat_k_inputs);
    v = mm->add_instruction(make_op("concat_past_present", {{"kv_num_heads", p.nhead_kv}}),
                            concat_v_inputs);

    auto k_out = k;
    auto v_out = v;

    auto kv_num_heads_factor = p.nhead / p.nhead_kv;
    auto max_seq_len         = k->get_shape().lens()[2];
    auto past_sl =
        mm->add_instruction(make_op("multibroadcast", {{"out_lens", {p.batch, p.nhead}}}), slk);

    if(kv_num_heads_factor != 1)
    {
        auto kv_new_lens         = k->get_shape().lens();
        kv_new_lens.at(1)        = p.nhead;
        k                        = mm->add_instruction(make_op("unsqueeze", {{"axes", {2}}}), k);
        v                        = mm->add_instruction(make_op("unsqueeze", {{"axes", {2}}}), v);
        auto kv_unsqueezed_lens  = k->get_shape().lens();
        kv_unsqueezed_lens.at(2) = kv_num_heads_factor;
        k = mm->add_instruction(make_op("multibroadcast", {{"out_lens", kv_unsqueezed_lens}}), k);
        v = mm->add_instruction(make_op("multibroadcast", {{"out_lens", kv_unsqueezed_lens}}), v);
        k = mm->add_instruction(make_op("reshape", {{"dims", kv_new_lens}}), k);
        v = mm->add_instruction(make_op("reshape", {{"dims", kv_new_lens}}), v);
    }
    auto kt    = mm->add_instruction(make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), k);
    auto gemm1 = mm->add_instruction(make_op("dot"), q, kt);

    std::vector<int> range_vec(max_seq_len);
    std::iota(range_vec.begin(), range_vec.end(), 0);
    shape range_s{past_sl->get_shape().type(), {max_seq_len}};
    auto range = mm->add_literal(range_s, range_vec);
    std::vector<std::size_t> bnsm{p.batch, p.nhead, p.seqlen, max_seq_len};
    auto bc_range = mm->add_instruction(make_op("multibroadcast", {{"out_lens", bnsm}}), range);

    auto scalar_s = shape{transposed_qkv->get_shape().type(), {1}};
    auto ninf     = mm->add_literal(literal{scalar_s, {-std::numeric_limits<float>::infinity()}});
    ninf          = mm->add_instruction(make_op("multibroadcast", {{"out_lens", bnsm}}), ninf);

    auto scale = p.scale;
    if(float_equal(scale, 0.0))
    {
        scale = 1.0f / std::sqrt(static_cast<float>(p.head_size));
    }
    auto scale_ins = mm->add_literal(literal{scalar_s, {scale}});
    scale_ins = mm->add_instruction(make_op("multibroadcast", {{"out_lens", bnsm}}), scale_ins);
    auto mul  = mm->add_instruction(make_op("mul"), gemm1, scale_ins);

    instruction_ref seq_range;
    if(p.seqlen > 1)
    {
        std::vector<int> seq_range_vec(p.seqlen);
        std::iota(seq_range_vec.begin(), seq_range_vec.end(), 0);
        shape seq_range_s{past_sl->get_shape().type(), {p.seqlen}};
        seq_range = mm->add_literal(seq_range_s, seq_range_vec);
        seq_range = mm->add_instruction(make_op("reshape", {{"dims", {p.seqlen, 1}}}), seq_range);
        seq_range = mm->add_instruction(make_op("multibroadcast", {{"out_lens", bnsm}}), seq_range);
        auto causal_mask = mm->add_instruction(make_op("greater"), bc_range, seq_range);
        causal_mask = mm->add_instruction(make_op("convert", {{"target_type", shape::bool_type}}),
                                          causal_mask);
        mul         = mm->add_instruction(make_op("where"), causal_mask, ninf, mul);
    }

    auto bc_past_sl =
        mm->add_instruction(make_op("reshape", {{"dims", {p.batch, p.nhead, 1, 1}}}), past_sl);
    auto mask_comp =
        mm->add_instruction(make_op("multibroadcast", {{"out_lens", bnsm}}), bc_past_sl);
    if(p.local_window_size > 0)
    {
        bool is_prompt       = p.seqlen > 1;
        auto window_size_lit = mm->add_literal(
            migraphx::literal{migraphx::shape{past_sl->get_shape().type(), {1}},
                              {is_prompt ? -p.local_window_size : -(p.local_window_size + 1)}});
        window_size_lit = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", bnsm}}), window_size_lit);
        auto window_comp = mm->add_instruction(
            migraphx::make_op("add"), is_prompt ? seq_range : mask_comp, window_size_lit);
        auto window_mask = mm->add_instruction(migraphx::make_op("greater"), window_comp, bc_range);
        window_mask      = mm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::bool_type}}),
            window_mask);
        mul = mm->add_instruction(migraphx::make_op("where"), window_mask, ninf, mul);
    }
    auto mask  = mm->add_instruction(make_op("greater"), bc_range, mask_comp);
    mask       = mm->add_instruction(make_op("convert", {{"target_type", shape::bool_type}}), mask);
    auto where = mm->add_instruction(make_op("where"), mask, ninf, mul);
    auto softmax = mm->add_instruction(make_op("softmax", {{"axis", 3}}), where);
    auto scores  = mm->add_instruction(make_op("dot"), softmax, v);
    auto out = mm->add_instruction(make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), scores);
    out      = mm->add_instruction(
        make_op("reshape", {{"dims", {p.batch, p.seqlen, p.head_size * p.nhead}}}), out);
    mm->add_return({out, k_out, v_out});

    return prog;
}

TEST_CASE(gqa_decode_local)
{
    gqa_program_params params;
    params.batch              = 1;
    params.nhead              = 2;
    params.nhead_kv           = 2;
    params.seqlen             = 1;
    params.seqlen_old         = 10;
    params.head_size          = 16;
    params.local_window_size  = 4;
    params.scale              = 1.0;
    params.do_rotary          = true;
    params.rotary_interleaved = false;
    auto prog                 = make_gqa_program(params, true);

    std::cout << prog << std::endl;
    migraphx::compile_options opts;
    opts.offload_copy = true;
    prog.compile(migraphx::make_target("gpu"), opts);
    std::cout << prog << std::endl;

    migraphx::shape qkv_shape{migraphx::shape::half_type, {1, 1, 96}};
    std::vector<float> qkv_data = {
        6.153,  7.545,  -0.804, -6.835, -6.139, 2.956,  6.716,  6.893,  7.473,  -9.716, -3.429,
        -4.660, 7.343,  -3.562, 9.919,  -7.503, 4.383,  -2.274, 1.762,  -6.985, 4.702,  -5.070,
        5.009,  6.450,  -2.873, 0.363,  0.052,  -9.278, 1.541,  -2.714, 0.147,  -5.890, -5.202,
        -9.477, 1.640,  9.588,  8.967,  -2.795, -8.801, 7.888,  -7.699, -6.706, -7.154, 6.284,
        -5.744, -5.343, 4.492,  -8.902, 1.595,  0.696,  5.202,  1.360,  -0.066, -1.406, -5.225,
        -4.940, 6.140,  2.266,  -6.849, -7.607, 0.914,  0.885,  9.477,  -7.357, 8.032,  9.065,
        -5.225, 6.465,  0.300,  -9.999, -0.089, 6.549,  -8.623, -7.224, 7.020,  -5.164, -8.470,
        -9.049, 0.766,  -8.397, 2.805,  7.043,  2.467,  8.405,  0.738,  -3.961, -4.948, 7.460,
        2.534,  -4.354, -5.608, 2.411,  -7.487, -0.264, -7.888, 0.128};

    migraphx::shape past_key_values_shape{migraphx::shape::half_type, {1, 2, 10, 16}};
    std::vector<float> past_key_values_data(past_key_values_shape.elements(), 1);

    migraphx::shape slk_shape{migraphx::shape::int32_type, {1}};
    std::vector<int> slk_data = {8};

    migraphx::shape trig_cache_shape{migraphx::shape::half_type, {10, 8}};
    std::vector<float> trig_cache_data(trig_cache_shape.elements(), 1);

    migraphx::literal qkv{qkv_shape, qkv_data};
    migraphx::literal past_key_values_key{past_key_values_shape, past_key_values_data};
    migraphx::literal past_key_values_value{past_key_values_shape, past_key_values_data};
    migraphx::literal seqlens_k{slk_shape, slk_data};
    migraphx::literal cos_cache{trig_cache_shape, trig_cache_data};
    migraphx::literal sin_cache{trig_cache_shape, trig_cache_data};

    migraphx::parameter_map pp;
    pp["qkv"]       = qkv.get_argument();
    pp["k_old"]     = past_key_values_key.get_argument();
    pp["v_old"]     = past_key_values_value.get_argument();
    pp["slk"]       = seqlens_k.get_argument();
    pp["cos_cache"] = cos_cache.get_argument();
    pp["sin_cache"] = sin_cache.get_argument();

    auto outputs       = prog.eval(pp);
    const auto& result = outputs.front();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    const auto& pres_key = outputs.at(1);
    std::vector<float> pres_key_vector;
    pres_key.visit([&](auto output) { pres_key_vector.assign(output.begin(), output.end()); });
    const auto& pres_val = outputs.back();
    std::vector<float> pres_val_vector;
    pres_val.visit([&](auto output) { pres_val_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold   = {1,        1,         1,        1,       1,        1,        1,
                                 1,        1,         1,        1,       1,        1,        1,
                                 1,        1,         2.80469,  7.04297, 2.4668,   8.39844,  0.737793,
                                 -3.96094, -4.94531,  7.45703,  2.5332,  -4.35156, -5.60547, 2.41016,
                                 -7.48438, -0.263916, -7.88672, 0.12793};
    std::vector<float> gold_k = {
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        2.49609,  -2.77344, 8.78906,   3.30469,  14.7031,  2.54492,  -13.2812,
        16.7812,  -12.8906, -16.1719, -5.51172, 15.8672,   3.21875,  -8.13281, -4.30859, -1.01172,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        -4.53906, -1.56934, 12.0469,  8.96094,  -0.979492, -2.28906, -14.6953, 2.41797,  7.73047,
        2.96094,  -1.64844, -6.24609, 0.847168, -0.520508, 4.25391,  -12.2891, 1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1};
    std::vector<float> gold_v = {
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        8.03125,  9.0625,   -5.22266,  6.46484,  0.299805, -9.99219, -0.0889893,
        6.54688,  -8.61719, -7.22266, 7.01953,  -5.16016,  -8.46875, -9.04688, 0.765625, -8.39062,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        2.80469,  7.04297,  2.4668,   8.39844,  0.737793,  -3.96094, -4.94531, 7.45703,  2.5332,
        -4.35156, -5.60547, 2.41016,  -7.48438, -0.263916, -7.88672, 0.12793,  1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1};

    EXPECT(migraphx::verify::verify_range_with_tolerance(result_vector,
                                                         migraphx::verify::expected{gold}));
    EXPECT(migraphx::verify::verify_range_with_tolerance(pres_key_vector,
                                                         migraphx::verify::expected{gold_k}));
    EXPECT(migraphx::verify::verify_range_with_tolerance(pres_val_vector,
                                                         migraphx::verify::expected{gold_v}));
}

TEST_CASE(gqa_decode)
{
    gqa_program_params params;
    params.batch              = 1;
    params.nhead              = 2;
    params.nhead_kv           = 2;
    params.seqlen             = 1;
    params.seqlen_old         = 10;
    params.head_size          = 16;
    params.local_window_size  = -1;
    params.scale              = 0.25;
    params.do_rotary          = true;
    params.rotary_interleaved = false;
    auto prog                 = make_gqa_program(params, true);

    std::cout << prog << std::endl;
    migraphx::compile_options opts;
    opts.offload_copy = true;
    prog.compile(migraphx::make_target("gpu"), opts);
    std::cout << prog << std::endl;

    migraphx::shape qkv_shape{migraphx::shape::half_type, {1, 1, 96}};
    std::vector<float> qkv_data = {
        6.153,  7.545,  -0.804, -6.835, -6.139, 2.956,  6.716,  6.893,  7.473,  -9.716, -3.429,
        -4.660, 7.343,  -3.562, 9.919,  -7.503, 4.383,  -2.274, 1.762,  -6.985, 4.702,  -5.070,
        5.009,  6.450,  -2.873, 0.363,  0.052,  -9.278, 1.541,  -2.714, 0.147,  -5.890, -5.202,
        -9.477, 1.640,  9.588,  8.967,  -2.795, -8.801, 7.888,  -7.699, -6.706, -7.154, 6.284,
        -5.744, -5.343, 4.492,  -8.902, 1.595,  0.696,  5.202,  1.360,  -0.066, -1.406, -5.225,
        -4.940, 6.140,  2.266,  -6.849, -7.607, 0.914,  0.885,  9.477,  -7.357, 8.032,  9.065,
        -5.225, 6.465,  0.300,  -9.999, -0.089, 6.549,  -8.623, -7.224, 7.020,  -5.164, -8.470,
        -9.049, 0.766,  -8.397, 2.805,  7.043,  2.467,  8.405,  0.738,  -3.961, -4.948, 7.460,
        2.534,  -4.354, -5.608, 2.411,  -7.487, -0.264, -7.888, 0.128};

    migraphx::shape past_key_values_shape{migraphx::shape::half_type, {1, 2, 10, 16}};
    std::vector<float> past_key_values_data(past_key_values_shape.elements(), 1);

    migraphx::shape slk_shape{migraphx::shape::int32_type, {1}};
    std::vector<int> slk_data = {8};

    migraphx::shape trig_cache_shape{migraphx::shape::half_type, {10, 8}};
    std::vector<float> trig_cache_data(trig_cache_shape.elements(), 1);

    migraphx::literal qkv{qkv_shape, qkv_data};
    migraphx::literal past_key_values_key{past_key_values_shape, past_key_values_data};
    migraphx::literal past_key_values_value{past_key_values_shape, past_key_values_data};
    migraphx::literal seqlens_k{slk_shape, slk_data};
    migraphx::literal cos_cache{trig_cache_shape, trig_cache_data};
    migraphx::literal sin_cache{trig_cache_shape, trig_cache_data};

    migraphx::parameter_map pp;
    pp["qkv"]       = qkv.get_argument();
    pp["k_old"]     = past_key_values_key.get_argument();
    pp["v_old"]     = past_key_values_value.get_argument();
    pp["slk"]       = seqlens_k.get_argument();
    pp["cos_cache"] = cos_cache.get_argument();
    pp["sin_cache"] = sin_cache.get_argument();

    auto outputs       = prog.eval(pp);
    const auto& result = outputs.front();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    const auto& pres_key = outputs.at(1);
    std::vector<float> pres_key_vector;
    pres_key.visit([&](auto output) { pres_key_vector.assign(output.begin(), output.end()); });
    const auto& pres_val = outputs.back();
    std::vector<float> pres_val_vector;
    pres_val.visit([&](auto output) { pres_val_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold   = {1,        1,         1,        1,       1,        1,        1,
                                 1,        1,         1,        1,       1,        1,        1,
                                 1,        1,         2.80469,  7.04297, 2.4668,   8.39844,  0.737793,
                                 -3.96094, -4.94531,  7.45703,  2.5332,  -4.35156, -5.60547, 2.41016,
                                 -7.48438, -0.263916, -7.88672, 0.12793};
    std::vector<float> gold_k = {
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        2.49609,  -2.77344, 8.78906,   3.30469,  14.7031,  2.54492,  -13.2812,
        16.7812,  -12.8906, -16.1719, -5.51172, 15.8672,   3.21875,  -8.13281, -4.30859, -1.01172,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        -4.53906, -1.56934, 12.0469,  8.96094,  -0.979492, -2.28906, -14.6953, 2.41797,  7.73047,
        2.96094,  -1.64844, -6.24609, 0.847168, -0.520508, 4.25391,  -12.2891, 1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1};
    std::vector<float> gold_v = {
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        8.03125,  9.0625,   -5.22266,  6.46484,  0.299805, -9.99219, -0.0889893,
        6.54688,  -8.61719, -7.22266, 7.01953,  -5.16016,  -8.46875, -9.04688, 0.765625, -8.39062,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        2.80469,  7.04297,  2.4668,   8.39844,  0.737793,  -3.96094, -4.94531, 7.45703,  2.5332,
        -4.35156, -5.60547, 2.41016,  -7.48438, -0.263916, -7.88672, 0.12793,  1,        1,
        1,        1,        1,        1,        1,         1,        1,        1,        1,
        1,        1,        1,        1,        1};

    EXPECT(migraphx::verify::verify_range_with_tolerance(result_vector,
                                                         migraphx::verify::expected{gold}));
    EXPECT(migraphx::verify::verify_range_with_tolerance(pres_key_vector,
                                                         migraphx::verify::expected{gold_k}));
    EXPECT(migraphx::verify::verify_range_with_tolerance(pres_val_vector,
                                                         migraphx::verify::expected{gold_v}));
}

TEST_CASE(gqa_prefill_local)
{
    gqa_program_params params;
    params.batch              = 1;
    params.nhead              = 2;
    params.nhead_kv           = 2;
    params.seqlen             = 8;
    params.seqlen_old         = 10;
    params.head_size          = 16;
    params.local_window_size  = 4;
    params.scale              = 1.0;
    params.do_rotary          = true;
    params.rotary_interleaved = false;
    auto prog                 = make_gqa_program(params, true);

    std::cout << prog << std::endl;
    migraphx::compile_options opts;
    opts.offload_copy = true;
    prog.compile(migraphx::make_target("gpu"), opts);
    std::cout << prog << std::endl;

    migraphx::shape qkv_shape{migraphx::shape::half_type, {1, 8, 96}};
    std::vector<float> qkv_data = {
        -2.707, -9.217, 3.604,  -3.169, -7.648, -0.631, 4.736,  -4.145, -5.164, 9.856,  7.113,
        -0.016, -1.276, 7.265,  -0.914, -8.918, -4.952, 0.462,  -5.337, 4.891,  9.857,  5.038,
        2.986,  -4.757, -1.091, 4.046,  6.013,  6.098,  3.373,  -2.033, -8.047, -9.521, 7.813,
        -3.437, 7.428,  -6.121, 9.238,  -5.061, -2.192, -5.968, 0.548,  2.166,  9.094,  3.307,
        7.071,  -1.826, -7.451, 0.603,  -8.055, -4.086, -8.365, 3.251,  9.895,  -6.341, -2.914,
        1.132,  5.589,  6.264,  -5.479, 8.920,  4.931,  7.371,  5.822,  8.066,  -5.280, -8.744,
        -9.929, 7.546,  -7.343, 2.731,  -2.635, -9.712, 5.641,  -6.348, 5.393,  -0.089, -9.904,
        1.394,  9.283,  1.883,  3.094,  -4.556, -7.857, -8.703, -9.219, 7.730,  -5.009, -4.866,
        7.475,  5.562,  7.201,  -2.014, 3.634,  -1.830, -7.319, -2.499, 3.372,  -9.584, 0.925,
        0.434,  8.353,  4.521,  -0.350, -1.656, -5.599, 2.280,  1.699,  6.249,  -6.387, 5.099,
        1.757,  9.203,  6.494,  -2.987, -2.495, 1.338,  8.907,  7.232,  4.376,  -7.904, 9.765,
        -9.688, -9.706, -4.688, -2.828, 0.508,  2.638,  -9.166, -1.640, -5.619, -8.270, -2.222,
        8.149,  -7.529, 0.142,  -4.503, -4.180, -7.148, -9.294, 0.267,  5.392,  5.752,  -4.127,
        -6.299, -8.599, -3.539, -5.793, -6.177, 2.489,  8.176,  -5.034, 0.405,  -5.426, 1.554,
        -8.540, 9.316,  -4.098, 6.140,  -9.922, 1.598,  -1.452, 2.244,  -4.847, 4.874,  8.239,
        6.175,  -7.873, -8.535, 4.070,  4.103,  8.056,  -1.885, -1.551, -7.098, 2.213,  -2.610,
        4.253,  6.512,  -0.361, 8.135,  3.667,  8.747,  -1.656, -9.427, -0.268, 4.739,  -7.581,
        6.935,  5.196,  7.102,  0.735,  -3.179, 7.180,  -8.168, -0.760, 6.111,  9.000,  -2.693,
        -0.643, 5.644,  8.603,  1.522,  9.508,  7.690,  -5.484, -9.172, -3.051, 9.003,  -9.445,
        -8.449, 1.191,  -0.515, 1.571,  -6.036, 0.673,  8.919,  -8.547, 6.691,  8.855,  7.517,
        5.560,  2.345,  -4.032, 9.375,  -6.827, -9.711, 3.658,  -7.130, 5.405,  7.769,  5.266,
        0.684,  -1.354, 5.411,  6.191,  -2.695, 9.052,  1.049,  5.516,  -5.571, -1.176, -7.180,
        0.719,  -1.372, -4.986, -4.852, 1.104,  -3.382, -6.809, -0.034, -4.718, -3.621, 5.799,
        6.828,  -4.359, -0.123, -4.502, -0.933, 3.399,  -7.633, -5.410, -5.718, 1.630,  -3.689,
        -5.703, -0.235, -5.001, 7.421,  -0.958, -8.775, 0.354,  -2.508, 1.126,  0.173,  -8.197,
        -0.074, -2.248, 6.708,  -1.006, 1.312,  -2.736, -2.821, 5.691,  -9.502, 1.738,  -6.845,
        -3.883, -5.388, -5.451, 3.385,  -2.914, -4.535, -7.023, 9.257,  0.785,  0.875,  -0.504,
        4.715,  -5.588, -6.764, 1.855,  -8.275, 1.559,  3.612,  -8.691, -0.468, 7.569,  9.601,
        9.055,  -8.358, 3.614,  5.003,  -9.313, 6.412,  -7.283, -9.730, 3.663,  -9.483, -4.799,
        -9.138, 0.205,  9.026,  5.978,  5.475,  -6.973, -9.074, 1.540,  1.737,  8.624,  -1.555,
        -2.075, 5.661,  6.908,  -7.930, 3.978,  -0.618, -0.905, -3.416, -8.463, 5.438,  -6.267,
        -5.317, 8.852,  9.000,  3.680,  -0.640, 3.683,  8.243,  -4.587, 2.403,  -5.576, 8.174,
        -9.982, -1.729, -0.696, -6.938, -9.163, -8.007, 9.840,  -9.835, -5.867, 4.368,  -7.267,
        8.786,  6.352,  -9.456, 5.952,  4.503,  -8.209, 0.343,  -4.738, -9.670, 2.162,  3.678,
        9.154,  6.820,  -0.892, 1.661,  -9.081, -6.337, -3.107, 1.651,  4.055,  -6.301, 2.418,
        -5.933, 4.144,  9.895,  1.156,  -8.081, -3.528, 9.768,  0.483,  1.298,  -2.931, 8.827,
        1.165,  -6.051, -2.965, -1.264, 8.529,  -5.677, 7.944,  6.342,  -0.239, 8.987,  2.951,
        3.416,  -6.629, -5.323, 1.390,  8.639,  -8.434, 9.970,  -2.866, -2.263, -6.153, 5.636,
        -2.043, -3.472, -9.734, 4.655,  -7.232, 0.846,  -3.584, 8.488,  -4.681, -5.627, -2.729,
        6.770,  2.796,  -0.180, -1.664, 9.035,  1.725,  7.415,  -8.953, -0.940, 8.638,  9.477,
        2.824,  -9.540, -9.162, -1.884, 9.558,  -2.283, 0.512,  -6.405, 5.503,  9.424,  -5.669,
        -4.580, -9.641, 5.800,  -1.705, -3.077, -4.671, -3.444, 2.331,  -0.195, 0.474,  -9.232,
        2.589,  8.462,  3.872,  -5.712, -6.627, -4.677, 8.425,  -7.382, -7.707, -2.488, 6.735,
        2.150,  4.466,  -0.837, 7.132,  8.136,  -6.477, 8.708,  -7.757, 5.772,  -2.523, -0.897,
        -7.703, 3.661,  4.968,  -1.326, -0.846, -6.884, -5.274, 7.457,  -7.555, -4.273, 7.191,
        1.489,  8.630,  -1.270, -4.059, 3.887,  -5.932, 8.531,  4.057,  -2.472, -8.513, 4.731,
        6.814,  -0.008, -3.060, 6.529,  1.532,  7.095,  -3.110, 9.843,  -8.941, 1.129,  3.779,
        0.082,  -2.473, -3.190, -6.463, 9.658,  7.538,  -6.143, 9.423,  9.233,  -5.731, -1.320,
        -9.228, -6.455, -4.296, 6.263,  -4.169, 7.245,  2.983,  2.765,  4.597,  -5.823, 5.130,
        8.140,  0.623,  5.960,  1.728,  4.201,  6.599,  3.884,  -3.067, -7.194, 0.401,  4.215,
        -0.841, 6.757,  8.023,  7.044,  -5.321, -2.362, 8.459,  6.712,  2.918,  4.862,  6.957,
        -5.234, -0.550, 5.537,  -2.537, 4.416,  -7.745, -4.740, -3.520, -5.459, 7.875,  -8.023,
        -4.016, -9.048, 5.533,  -8.619, -7.614, -1.452, -6.988, -7.505, -6.085, 1.798,  -9.648,
        -8.903, 5.042,  -9.601, 4.289,  -4.397, -6.635, 6.882,  1.451,  6.506,  1.565,  -0.529,
        -7.154, 1.606,  -7.394, 8.064,  1.269,  8.908,  -5.531, -5.293, 1.658,  -0.301, -2.814,
        4.509,  7.937,  0.459,  5.714,  -8.383, 7.540,  1.137,  -0.118, -1.319, -3.052, 7.871,
        3.566,  5.789,  -3.266, -9.810, 1.681,  2.242,  7.868,  -4.146, -3.221, -8.030, 8.651,
        -6.748, 0.277,  -0.627, 1.380,  8.226,  -6.454, -2.499, -3.430, -9.451, -3.828, -8.902,
        -3.198, -5.245, 1.525,  -9.971, -8.961, -8.049, 4.364,  -0.326, 5.662,  -6.424, 9.126,
        -0.828, 7.106,  -5.813, 7.622,  7.198,  -2.044, -0.942, 2.991,  5.672,  -3.264, 1.870,
        4.974,  1.450,  1.992,  -3.817, -1.086, 2.200,  0.219,  -3.009, -2.247, -5.495, 6.933,
        -1.711, -8.377, 3.968,  -1.958, -5.523, -3.127, 7.829,  5.818,  1.695,  -3.425, 8.928,
        -3.025, 4.507,  -4.567, -5.167, -2.225, -5.601, 8.581,  6.219,  -4.023, -1.921, 8.816,
        -7.227, -2.608, 9.676,  2.334,  3.524,  -0.892, 0.628,  -7.928, 5.060,  -2.705, -5.276,
        0.638,  -4.713, -3.290, -9.432, -5.974, 6.965,  5.995,  -5.425, 3.966,  1.468,  7.970,
        1.582,  -7.724, 3.168,  -1.222, 7.728,  -3.536, -6.907, 4.146,  -3.046, 8.990,  4.917,
        5.019,  -1.291, -0.490, 3.891,  -6.154, 6.963,  3.121,  2.338,  9.394,  8.832,  -5.484,
        -3.277, 2.075,  5.174,  2.390,  7.313,  8.872,  -0.613, 6.507,  -0.298, -1.448, 2.750,
        4.410,  -1.614, -8.528, -4.927, -2.355, 7.592,  9.312,  -3.493, 2.129,  -1.156, 1.035,
        6.225,  6.896,  -5.849, -0.693, -9.459, -4.728, -3.931, 0.930,  -1.244};

    migraphx::shape past_key_values_shape{migraphx::shape::half_type, {1, 2, 10, 16}};
    std::vector<float> past_key_values_data(past_key_values_shape.elements(), 1);

    migraphx::shape slk_shape{migraphx::shape::int32_type, {1}};
    std::vector<int> slk_data = {8};

    migraphx::shape trig_cache_shape{migraphx::shape::half_type, {10, 8}};
    std::vector<float> trig_cache_data(trig_cache_shape.elements(), 1);

    migraphx::literal qkv{qkv_shape, qkv_data};
    migraphx::literal past_key_values_key{past_key_values_shape, past_key_values_data};
    migraphx::literal past_key_values_value{past_key_values_shape, past_key_values_data};
    migraphx::literal seqlens_k{slk_shape, slk_data};
    migraphx::literal cos_cache{trig_cache_shape, trig_cache_data};
    migraphx::literal sin_cache{trig_cache_shape, trig_cache_data};

    migraphx::parameter_map pp;
    pp["qkv"]       = qkv.get_argument();
    pp["k_old"]     = past_key_values_key.get_argument();
    pp["v_old"]     = past_key_values_value.get_argument();
    pp["slk"]       = seqlens_k.get_argument();
    pp["cos_cache"] = cos_cache.get_argument();
    pp["sin_cache"] = sin_cache.get_argument();

    auto outputs       = prog.eval(pp);
    const auto& result = outputs.front();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    const auto& pres_key = outputs.at(1);
    std::vector<float> pres_key_vector;
    pres_key.visit([&](auto output) { pres_key_vector.assign(output.begin(), output.end()); });
    const auto& pres_val = outputs.back();
    std::vector<float> pres_val_vector;
    pres_val.visit([&](auto output) { pres_val_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {
        -5.27734,  -8.74219,  -9.92188,  7.54297,    -7.33984, 2.73047,  -2.63477, -9.71094,
        5.64062,   -6.34766,  5.39062,   -0.0889893, -9.89844, 1.39355,  9.28125,  1.88281,
        3.09375,   -4.55469,  -7.85547,  -8.69531,   -9.21875, 7.72656,  -5.00781, -4.86328,
        7.47266,   5.55859,   7.19922,   -2.01367,   3.63281,  -1.8291,  -7.31641, -2.49805,
        -5.27734,  -8.74219,  -9.92188,  7.54297,    -7.33984, 2.73047,  -2.63477, -9.71094,
        5.64062,   -6.34766,  5.39062,   -0.0889893, -9.89844, 1.39355,  9.28125,  1.88281,
        4.25,      6.51172,   -0.36084,  8.13281,    3.66602,  8.74219,  -1.65527, -9.42188,
        -0.267822, 4.73828,   -7.57812,  6.93359,    5.19531,  7.10156,  0.734863, -3.17773,
        -5.27734,  -8.74219,  -9.92188,  7.54297,    -7.33984, 2.73047,  -2.63477, -9.71094,
        5.64062,   -6.34766,  5.39062,   -0.0889893, -9.89844, 1.39355,  9.28125,  1.88281,
        3.09375,   -4.55469,  -7.85547,  -8.69531,   -9.21875, 7.72656,  -5.00781, -4.86328,
        7.47266,   5.55859,   7.19922,   -2.01367,   3.63281,  -1.8291,  -7.31641, -2.49805,
        -4.5,      -0.932617, 3.39844,   -7.63281,   -5.40625, -5.71484, 1.62988,  -3.6875,
        -5.69922,  -0.234985, -5,        7.41797,    -0.95752, -8.77344, 0.35376,  -2.50781,
        1.12598,   0.172974,  -8.19531,  -0.0739746, -2.24609, 6.70703,  -1.00586, 1.31152,
        -2.73438,  -2.82031,  5.6875,    -9.5,       1.7373,   -6.84375, -3.88281, -5.38672,
        -9.97656,  -1.72852,  -0.695801, -6.9375,    -9.15625, -8,       9.83594,  -9.82812,
        -5.86328,  4.36719,   -7.26562,  8.78125,    6.35156,  -9.45312, 5.94922,  4.5,
        4.25,      6.51172,   -0.36084,  8.13281,    3.66602,  8.74219,  -1.65527, -9.42188,
        -0.267822, 4.73828,   -7.57812,  6.93359,    5.19531,  7.10156,  0.734863, -3.17773,
        -9.97656,  -1.72852,  -0.695801, -6.9375,    -9.15625, -8,       9.83594,  -9.82812,
        -5.86328,  4.36719,   -7.26562,  8.78125,    6.35156,  -9.45312, 5.94922,  4.5,
        -8.20312,  0.342773,  -4.73438,  -9.66406,   2.16016,  3.67773,  9.14844,  6.81641,
        -0.891602, 1.66016,   -9.07812,  -6.33594,   -3.10547, 1.65039,  4.05469,  -6.30078,
        5.5,       9.42188,   -5.66797,  -4.57812,   -9.64062, 5.79688,  -1.7041,  -3.07617,
        -4.66797,  -3.44336,  2.33008,   -0.194946,  0.473877, -9.22656, 2.58789,  8.46094,
        6.95312,   -5.23047,  -0.549805, 5.53516,    -2.53516, 4.41406,  -7.74219, -4.73828,
        -3.51953,  -5.45703,  7.875,     -8.01562,   -4.01562, -9.04688, 5.53125,  -8.61719,
        5.5,       9.42188,   -5.66797,  -4.57812,   -9.64062, 5.79688,  -1.7041,  -3.07617,
        -4.66797,  -3.44336,  2.33008,   -0.194946,  0.473877, -9.22656, 2.58789,  8.46094,
        -8.20312,  0.342773,  -4.73438,  -9.66406,   2.16016,  3.67773,  9.14844,  6.81641,
        -0.891602, 1.66016,   -9.07812,  -6.33594,   -3.10547, 1.65039,  4.05469,  -6.30078};
    std::vector<float> gold_k = {
        7.26172,  -5.59766,  -1.66797, -9.42188, 2.16406,    -3.23242, 5.25781,  -6.56641,
        8.35938,  -1.27148,  16.5156,  -2.81055, 16.2969,    -6.88281, -9.64062, -5.35938,
        2.53906,  1.52734,   1.02344,  -2.48633, 2.75781,    -13.2734, 4.26562,  1.79688,
        -5.81641, -12.7578,  -17.5469, -1.95312, 13.5391,    -1.77734, -3.98242, -10.7969,
        -5.46875, -15.1172,  -2.53125, -4.43359, -3.64453,   6.71484,  -0.25,    6.25391,
        -8.17188, -4.30078,  9.84375,  -9.82031, 14.4453,    8.8125,   10.7812,  -4.88672,
        -8.40625, 10.5781,   8.04688,  -0.1875,  -13.875,    -1.14062, -2.4375,  2.35352,
        8.82031,  7.46875,   3.90234,  11.1328,  -0.0664062, -17,      5.51562,  1.11816,
        -2.56836, -2.85156,  2.63672,  2.1543,   -6.99609,   -2.11719, -10.0234, 1.02539,
        -9.73438, 14.1172,   -6.72266, -9.09375, -12.4531,   11.4219,  -4.43359, 0.665527,
        3.35156,  0.179688,  -16.4688, 7.26953,  -5.64453,   -9.14062, 3.25781,  -1.87012,
        -9.57031, 19.4844,   -1.40234, -5.01172, 13.1953,    9.30469,  -8.20312, -4.50781,
        2.14648,  -14.1641,  10.8047,  10.9375,  -1.79785,   -3.55859, -10.9141, 12.0078,
        9.27344,  -2.59766,  4.27344,  -8.66406, 1.5625,     0.921875, 4.81641,  3.72266,
        -3.32617, -6.17578,  -11.2578, -11.0078, 1.75,       3.79688,  7.21094,  -13.1484,
        4.60156,  -3.24219,  4.67969,  -7.84766, -13.6953,   10.1328,  4.76953,  2.30469,
        1,        1,         1,        1,        1,          1,        1,        1,
        1,        1,         1,        1,        1,          1,        1,        1,
        1,        1,         1,        1,        1,          1,        1,        1,
        1,        1,         1,        1,        1,          1,        1,        1,
        -13.6406, -10.3438,  -2.88281, -5.66406, 4.96094,    -13.7031, -8.72656, -6.92969,
        -2.46875, 2.17578,   -13.8359, 12.1641,  14.8203,    1.02734,  2.9082,   9.1875,
        -3.16797, -5.08984,  2.74609,  -15.4844, 6.58594,    2.03516,  4.89062,  -1.19238,
        -14.0156, -1.9834,   -14.3281, 3.13672,  -1.60938,   14.3047,  -14.9531, 2.00195,
        5.63281,  -7.14453,  5.43359,  2.24805,  -10.7812,   -11.6719, 5.45703,  -3.25781,
        -7.98438, -7.21094,  -3.99609, -4.98828, 0.8125,     1.97266,  -3.25195, -3.50195,
        -4.58203, -2.77344,  -12.1406, -2.80469, -1.67969,   -7.71875, 14.4219,  0.828125,
        2.77344,  -4.05078,  -4.77734, 13.6797,  -10.8516,   -2.91406, 3.27734,  17.1719,
        -4.48438, 18.5625,   10.875,   9.29688,  -18.5,      1.34082,  8.11719,  15.875,
        1.15918,  -0.507812, -7.42969, 5.52734,  0.609375,   -3.2207,  9.14062,  3.07422,
        -13.8203, -0.632812, -9.42188, -1.87109, -4.78906,   1.28516,  1.25488,  -1.43555,
        -4.63281, -12.2734,  0.835938, 14.3906,  -3.54492,   13.1953,  4.70703,  6.96094,
        3.23242,  -5.52344,  12.0781,  2.69922,  4.10156,    8.26562,  4.57422,  13.4609,
        -9.67188, -10.5156,  5.21875,  -16.1875, -3.54883,   -9.52344, -1.81738, 2.97656,
        -3.04492, -10.7969,  10.2969,  -10,      5.86328,    2.57617,  -4.375,   -10.1172,
        -4.02344, -3.01562,  -2.00781, 3.91602,  12.1016,    7.25,     14.4062,  7.53516,
        1,        1,         1,        1,        1,          1,        1,        1,
        1,        1,         1,        1,        1,          1,        1,        1,
        1,        1,         1,        1,        1,          1,        1,        1,
        1,        1,         1,        1,        1,          1,        1,        1};
    std::vector<float> gold_v = {
        -5.27734,  -8.74219,  -9.92188,  7.54297,    -7.33984, 2.73047,   -2.63477, -9.71094,
        5.64062,   -6.34766,  5.39062,   -0.0889893, -9.89844, 1.39355,   9.28125,  1.88281,
        -1.45117,  2.24219,   -4.84375,  4.87109,    8.23438,  6.17188,   -7.87109, -8.53125,
        4.06641,   4.10156,   8.05469,   -1.88477,   -1.55078, -7.09766,  2.21289,  -2.60938,
        -4.5,      -0.932617, 3.39844,   -7.63281,   -5.40625, -5.71484,  1.62988,  -3.6875,
        -5.69922,  -0.234985, -5,        7.41797,    -0.95752, -8.77344,  0.35376,  -2.50781,
        -9.97656,  -1.72852,  -0.695801, -6.9375,    -9.15625, -8,        9.83594,  -9.82812,
        -5.86328,  4.36719,   -7.26562,  8.78125,    6.35156,  -9.45312,  5.94922,  4.5,
        5.5,       9.42188,   -5.66797,  -4.57812,   -9.64062, 5.79688,   -1.7041,  -3.07617,
        -4.66797,  -3.44336,  2.33008,   -0.194946,  0.473877, -9.22656,  2.58789,  8.46094,
        6.59766,   3.88281,   -3.06641,  -7.19141,   0.400879, 4.21484,   -0.84082, 6.75391,
        8.01562,   7.04297,   -5.32031,  -2.36133,   8.45312,  6.71094,   2.91797,  4.85938,
        1.52441,   -9.96875,  -8.96094,  -8.04688,   4.36328,  -0.325928, 5.66016,  -6.42188,
        9.125,     -0.827637, 7.10547,   -5.8125,    7.62109,  7.19531,   -2.04297, -0.941895,
        -5.48047,  -3.27539,  2.07422,   5.17188,    2.38867,  7.3125,    8.86719,  -0.612793,
        6.50391,   -0.297852, -1.44727,  2.75,       4.40625,  -1.61328,  -8.52344, -4.92578,
        1,         1,         1,         1,          1,        1,         1,        1,
        1,         1,         1,         1,          1,        1,         1,        1,
        1,         1,         1,         1,          1,        1,         1,        1,
        1,         1,         1,         1,          1,        1,         1,        1,
        3.09375,   -4.55469,  -7.85547,  -8.69531,   -9.21875, 7.72656,   -5.00781, -4.86328,
        7.47266,   5.55859,   7.19922,   -2.01367,   3.63281,  -1.8291,   -7.31641, -2.49805,
        4.25,      6.51172,   -0.36084,  8.13281,    3.66602,  8.74219,   -1.65527, -9.42188,
        -0.267822, 4.73828,   -7.57812,  6.93359,    5.19531,  7.10156,   0.734863, -3.17773,
        1.12598,   0.172974,  -8.19531,  -0.0739746, -2.24609, 6.70703,   -1.00586, 1.31152,
        -2.73438,  -2.82031,  5.6875,    -9.5,       1.7373,   -6.84375,  -3.88281, -5.38672,
        -8.20312,  0.342773,  -4.73438,  -9.66406,   2.16016,  3.67773,   9.14844,  6.81641,
        -0.891602, 1.66016,   -9.07812,  -6.33594,   -3.10547, 1.65039,   4.05469,  -6.30078,
        3.87109,   -5.71094,  -6.625,    -4.67578,   8.42188,  -7.37891,  -7.70312, -2.48633,
        6.73438,   2.14844,   4.46484,   -0.836914,  7.12891,  8.13281,   -6.47656, 8.70312,
        6.95312,   -5.23047,  -0.549805, 5.53516,    -2.53516, 4.41406,   -7.74219, -4.73828,
        -3.51953,  -5.45703,  7.875,     -8.01562,   -4.01562, -9.04688,  5.53125,  -8.61719,
        2.99023,   5.67188,   -3.26367,  1.86914,    4.97266,  1.44922,   1.99121,  -3.81641,
        -1.08594,  2.19922,   0.218994,  -3.00781,   -2.24609, -5.49219,  6.92969,  -1.71094,
        -2.35352,  7.58984,   9.30469,   -3.49219,   2.12891,  -1.15527,  1.03418,  6.22266,
        6.89453,   -5.84766,  -0.692871, -9.45312,   -4.72656, -3.92969,  0.929688, -1.24316,
        1,         1,         1,         1,          1,        1,         1,        1,
        1,         1,         1,         1,          1,        1,         1,        1,
        1,         1,         1,         1,          1,        1,         1,        1,
        1,         1,         1,         1,          1,        1,         1,        1};

    CHECK(migraphx::verify::verify_range_with_tolerance(result_vector,
                                                        migraphx::verify::expected{gold}));
    CHECK(migraphx::verify::verify_range_with_tolerance(pres_key_vector,
                                                        migraphx::verify::expected{gold_k}));
    CHECK(migraphx::verify::verify_range_with_tolerance(pres_val_vector,
                                                        migraphx::verify::expected{gold_v}));
}

TEST_CASE(gqa_prefill)
{
    gqa_program_params params;
    params.batch              = 1;
    params.nhead              = 2;
    params.nhead_kv           = 2;
    params.seqlen             = 8;
    params.seqlen_old         = 10;
    params.head_size          = 16;
    params.local_window_size  = -1;
    params.scale              = 0.25;
    params.do_rotary          = true;
    params.rotary_interleaved = false;
    auto prog                 = make_gqa_program(params, true);

    std::cout << prog << std::endl;
    migraphx::compile_options opts;
    opts.offload_copy = true;
    prog.compile(migraphx::make_target("gpu"), opts);
    std::cout << prog << std::endl;

    migraphx::shape qkv_shape{migraphx::shape::half_type, {1, 8, 96}};
    std::vector<float> qkv_data = {
        -2.707, -9.217, 3.604,  -3.169, -7.648, -0.631, 4.736,  -4.145, -5.164, 9.856,  7.113,
        -0.016, -1.276, 7.265,  -0.914, -8.918, -4.952, 0.462,  -5.337, 4.891,  9.857,  5.038,
        2.986,  -4.757, -1.091, 4.046,  6.013,  6.098,  3.373,  -2.033, -8.047, -9.521, 7.813,
        -3.437, 7.428,  -6.121, 9.238,  -5.061, -2.192, -5.968, 0.548,  2.166,  9.094,  3.307,
        7.071,  -1.826, -7.451, 0.603,  -8.055, -4.086, -8.365, 3.251,  9.895,  -6.341, -2.914,
        1.132,  5.589,  6.264,  -5.479, 8.920,  4.931,  7.371,  5.822,  8.066,  -5.280, -8.744,
        -9.929, 7.546,  -7.343, 2.731,  -2.635, -9.712, 5.641,  -6.348, 5.393,  -0.089, -9.904,
        1.394,  9.283,  1.883,  3.094,  -4.556, -7.857, -8.703, -9.219, 7.730,  -5.009, -4.866,
        7.475,  5.562,  7.201,  -2.014, 3.634,  -1.830, -7.319, -2.499, 3.372,  -9.584, 0.925,
        0.434,  8.353,  4.521,  -0.350, -1.656, -5.599, 2.280,  1.699,  6.249,  -6.387, 5.099,
        1.757,  9.203,  6.494,  -2.987, -2.495, 1.338,  8.907,  7.232,  4.376,  -7.904, 9.765,
        -9.688, -9.706, -4.688, -2.828, 0.508,  2.638,  -9.166, -1.640, -5.619, -8.270, -2.222,
        8.149,  -7.529, 0.142,  -4.503, -4.180, -7.148, -9.294, 0.267,  5.392,  5.752,  -4.127,
        -6.299, -8.599, -3.539, -5.793, -6.177, 2.489,  8.176,  -5.034, 0.405,  -5.426, 1.554,
        -8.540, 9.316,  -4.098, 6.140,  -9.922, 1.598,  -1.452, 2.244,  -4.847, 4.874,  8.239,
        6.175,  -7.873, -8.535, 4.070,  4.103,  8.056,  -1.885, -1.551, -7.098, 2.213,  -2.610,
        4.253,  6.512,  -0.361, 8.135,  3.667,  8.747,  -1.656, -9.427, -0.268, 4.739,  -7.581,
        6.935,  5.196,  7.102,  0.735,  -3.179, 7.180,  -8.168, -0.760, 6.111,  9.000,  -2.693,
        -0.643, 5.644,  8.603,  1.522,  9.508,  7.690,  -5.484, -9.172, -3.051, 9.003,  -9.445,
        -8.449, 1.191,  -0.515, 1.571,  -6.036, 0.673,  8.919,  -8.547, 6.691,  8.855,  7.517,
        5.560,  2.345,  -4.032, 9.375,  -6.827, -9.711, 3.658,  -7.130, 5.405,  7.769,  5.266,
        0.684,  -1.354, 5.411,  6.191,  -2.695, 9.052,  1.049,  5.516,  -5.571, -1.176, -7.180,
        0.719,  -1.372, -4.986, -4.852, 1.104,  -3.382, -6.809, -0.034, -4.718, -3.621, 5.799,
        6.828,  -4.359, -0.123, -4.502, -0.933, 3.399,  -7.633, -5.410, -5.718, 1.630,  -3.689,
        -5.703, -0.235, -5.001, 7.421,  -0.958, -8.775, 0.354,  -2.508, 1.126,  0.173,  -8.197,
        -0.074, -2.248, 6.708,  -1.006, 1.312,  -2.736, -2.821, 5.691,  -9.502, 1.738,  -6.845,
        -3.883, -5.388, -5.451, 3.385,  -2.914, -4.535, -7.023, 9.257,  0.785,  0.875,  -0.504,
        4.715,  -5.588, -6.764, 1.855,  -8.275, 1.559,  3.612,  -8.691, -0.468, 7.569,  9.601,
        9.055,  -8.358, 3.614,  5.003,  -9.313, 6.412,  -7.283, -9.730, 3.663,  -9.483, -4.799,
        -9.138, 0.205,  9.026,  5.978,  5.475,  -6.973, -9.074, 1.540,  1.737,  8.624,  -1.555,
        -2.075, 5.661,  6.908,  -7.930, 3.978,  -0.618, -0.905, -3.416, -8.463, 5.438,  -6.267,
        -5.317, 8.852,  9.000,  3.680,  -0.640, 3.683,  8.243,  -4.587, 2.403,  -5.576, 8.174,
        -9.982, -1.729, -0.696, -6.938, -9.163, -8.007, 9.840,  -9.835, -5.867, 4.368,  -7.267,
        8.786,  6.352,  -9.456, 5.952,  4.503,  -8.209, 0.343,  -4.738, -9.670, 2.162,  3.678,
        9.154,  6.820,  -0.892, 1.661,  -9.081, -6.337, -3.107, 1.651,  4.055,  -6.301, 2.418,
        -5.933, 4.144,  9.895,  1.156,  -8.081, -3.528, 9.768,  0.483,  1.298,  -2.931, 8.827,
        1.165,  -6.051, -2.965, -1.264, 8.529,  -5.677, 7.944,  6.342,  -0.239, 8.987,  2.951,
        3.416,  -6.629, -5.323, 1.390,  8.639,  -8.434, 9.970,  -2.866, -2.263, -6.153, 5.636,
        -2.043, -3.472, -9.734, 4.655,  -7.232, 0.846,  -3.584, 8.488,  -4.681, -5.627, -2.729,
        6.770,  2.796,  -0.180, -1.664, 9.035,  1.725,  7.415,  -8.953, -0.940, 8.638,  9.477,
        2.824,  -9.540, -9.162, -1.884, 9.558,  -2.283, 0.512,  -6.405, 5.503,  9.424,  -5.669,
        -4.580, -9.641, 5.800,  -1.705, -3.077, -4.671, -3.444, 2.331,  -0.195, 0.474,  -9.232,
        2.589,  8.462,  3.872,  -5.712, -6.627, -4.677, 8.425,  -7.382, -7.707, -2.488, 6.735,
        2.150,  4.466,  -0.837, 7.132,  8.136,  -6.477, 8.708,  -7.757, 5.772,  -2.523, -0.897,
        -7.703, 3.661,  4.968,  -1.326, -0.846, -6.884, -5.274, 7.457,  -7.555, -4.273, 7.191,
        1.489,  8.630,  -1.270, -4.059, 3.887,  -5.932, 8.531,  4.057,  -2.472, -8.513, 4.731,
        6.814,  -0.008, -3.060, 6.529,  1.532,  7.095,  -3.110, 9.843,  -8.941, 1.129,  3.779,
        0.082,  -2.473, -3.190, -6.463, 9.658,  7.538,  -6.143, 9.423,  9.233,  -5.731, -1.320,
        -9.228, -6.455, -4.296, 6.263,  -4.169, 7.245,  2.983,  2.765,  4.597,  -5.823, 5.130,
        8.140,  0.623,  5.960,  1.728,  4.201,  6.599,  3.884,  -3.067, -7.194, 0.401,  4.215,
        -0.841, 6.757,  8.023,  7.044,  -5.321, -2.362, 8.459,  6.712,  2.918,  4.862,  6.957,
        -5.234, -0.550, 5.537,  -2.537, 4.416,  -7.745, -4.740, -3.520, -5.459, 7.875,  -8.023,
        -4.016, -9.048, 5.533,  -8.619, -7.614, -1.452, -6.988, -7.505, -6.085, 1.798,  -9.648,
        -8.903, 5.042,  -9.601, 4.289,  -4.397, -6.635, 6.882,  1.451,  6.506,  1.565,  -0.529,
        -7.154, 1.606,  -7.394, 8.064,  1.269,  8.908,  -5.531, -5.293, 1.658,  -0.301, -2.814,
        4.509,  7.937,  0.459,  5.714,  -8.383, 7.540,  1.137,  -0.118, -1.319, -3.052, 7.871,
        3.566,  5.789,  -3.266, -9.810, 1.681,  2.242,  7.868,  -4.146, -3.221, -8.030, 8.651,
        -6.748, 0.277,  -0.627, 1.380,  8.226,  -6.454, -2.499, -3.430, -9.451, -3.828, -8.902,
        -3.198, -5.245, 1.525,  -9.971, -8.961, -8.049, 4.364,  -0.326, 5.662,  -6.424, 9.126,
        -0.828, 7.106,  -5.813, 7.622,  7.198,  -2.044, -0.942, 2.991,  5.672,  -3.264, 1.870,
        4.974,  1.450,  1.992,  -3.817, -1.086, 2.200,  0.219,  -3.009, -2.247, -5.495, 6.933,
        -1.711, -8.377, 3.968,  -1.958, -5.523, -3.127, 7.829,  5.818,  1.695,  -3.425, 8.928,
        -3.025, 4.507,  -4.567, -5.167, -2.225, -5.601, 8.581,  6.219,  -4.023, -1.921, 8.816,
        -7.227, -2.608, 9.676,  2.334,  3.524,  -0.892, 0.628,  -7.928, 5.060,  -2.705, -5.276,
        0.638,  -4.713, -3.290, -9.432, -5.974, 6.965,  5.995,  -5.425, 3.966,  1.468,  7.970,
        1.582,  -7.724, 3.168,  -1.222, 7.728,  -3.536, -6.907, 4.146,  -3.046, 8.990,  4.917,
        5.019,  -1.291, -0.490, 3.891,  -6.154, 6.963,  3.121,  2.338,  9.394,  8.832,  -5.484,
        -3.277, 2.075,  5.174,  2.390,  7.313,  8.872,  -0.613, 6.507,  -0.298, -1.448, 2.750,
        4.410,  -1.614, -8.528, -4.927, -2.355, 7.592,  9.312,  -3.493, 2.129,  -1.156, 1.035,
        6.225,  6.896,  -5.849, -0.693, -9.459, -4.728, -3.931, 0.930,  -1.244};

    migraphx::shape past_key_values_shape{migraphx::shape::half_type, {1, 2, 10, 16}};
    std::vector<float> past_key_values_data(past_key_values_shape.elements(), 1);

    migraphx::shape slk_shape{migraphx::shape::int32_type, {1}};
    std::vector<int> slk_data = {8};

    migraphx::shape trig_cache_shape{migraphx::shape::half_type, {10, 8}};
    std::vector<float> trig_cache_data(trig_cache_shape.elements(), 1);

    migraphx::literal qkv{qkv_shape, qkv_data};
    migraphx::literal past_key_values_key{past_key_values_shape, past_key_values_data};
    migraphx::literal past_key_values_value{past_key_values_shape, past_key_values_data};
    migraphx::literal seqlens_k{slk_shape, slk_data};
    migraphx::literal cos_cache{trig_cache_shape, trig_cache_data};
    migraphx::literal sin_cache{trig_cache_shape, trig_cache_data};

    migraphx::parameter_map pp;
    pp["qkv"]       = qkv.get_argument();
    pp["k_old"]     = past_key_values_key.get_argument();
    pp["v_old"]     = past_key_values_value.get_argument();
    pp["slk"]       = seqlens_k.get_argument();
    pp["cos_cache"] = cos_cache.get_argument();
    pp["sin_cache"] = sin_cache.get_argument();

    auto outputs       = prog.eval(pp);
    const auto& result = outputs.front();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    const auto& pres_key = outputs.at(1);
    std::vector<float> pres_key_vector;
    pres_key.visit([&](auto output) { pres_key_vector.assign(output.begin(), output.end()); });
    const auto& pres_val = outputs.back();
    std::vector<float> pres_val_vector;
    pres_val.visit([&](auto output) { pres_val_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {
        -5.27734,  -8.74219,  -9.92188,  7.54297,    -7.33984,  2.73047,  -2.63477,  -9.71094,
        5.64062,   -6.34766,  5.39062,   -0.0889893, -9.89844,  1.39355,  9.28125,   1.88281,
        3.09375,   -4.55469,  -7.85547,  -8.69531,   -9.21875,  7.72656,  -5.00781,  -4.86328,
        7.47266,   5.55859,   7.19922,   -2.01367,   3.63281,   -1.8291,  -7.31641,  -2.49805,
        -5.27734,  -8.74219,  -9.92188,  7.54297,    -7.33984,  2.73047,  -2.63477,  -9.71094,
        5.64062,   -6.34766,  5.39062,   -0.0889893, -9.89844,  1.39355,  9.28125,   1.88281,
        4.24609,   6.50781,   -0.36084,  8.125,      3.66211,   8.73438,  -1.6543,   -9.41406,
        -0.26709,  4.73438,   -7.57031,  6.92969,    5.19141,   7.09766,  0.733887,  -3.17578,
        -5.27734,  -8.74219,  -9.92188,  7.54297,    -7.33984,  2.73047,  -2.63477,  -9.71094,
        5.64062,   -6.34766,  5.39062,   -0.0889893, -9.89844,  1.39355,  9.28125,   1.88281,
        3.09375,   -4.55469,  -7.85547,  -8.69531,   -9.21875,  7.72656,  -5.00781,  -4.86328,
        7.47266,   5.55859,   7.19922,   -2.01367,   3.63281,   -1.8291,  -7.31641,  -2.49805,
        -4.5,      -0.932617, 3.39844,   -7.63281,   -5.40625,  -5.71484, 1.62988,   -3.6875,
        -5.69922,  -0.234985, -5,        7.41797,    -0.95752,  -8.77344, 0.35376,   -2.50781,
        1.12598,   0.172974,  -8.19531,  -0.0739746, -2.24609,  6.70703,  -1.00586,  1.31152,
        -2.73438,  -2.82031,  5.6875,    -9.5,       1.7373,    -6.84375, -3.88281,  -5.38672,
        -9.97656,  -1.72852,  -0.695801, -6.9375,    -9.15625,  -8,       9.83594,   -9.82812,
        -5.86328,  4.36719,   -7.26562,  8.78125,    6.35156,   -9.45312, 5.94922,   4.5,
        4.24609,   6.50781,   -0.360596, 8.125,      3.66406,   8.73438,  -1.6543,   -9.41406,
        -0.267578, 4.73438,   -7.57422,  6.92969,    5.19141,   7.09766,  0.734375,  -3.17578,
        -9.97656,  -1.72852,  -0.695801, -6.9375,    -9.15625,  -8,       9.83594,   -9.82812,
        -5.86328,  4.36719,   -7.26562,  8.78125,    6.35156,   -9.45312, 5.94922,   4.5,
        -8.19531,  0.342529,  -4.73047,  -9.65625,   2.1582,    3.67578,  9.14062,   6.8125,
        -0.891113, 1.65918,   -9.07031,  -6.33203,   -3.10352,  1.64941,  4.05078,   -6.29688,
        5.5,       9.42188,   -5.66797,  -4.57812,   -9.64062,  5.79688,  -1.7041,   -3.07617,
        -4.66797,  -3.44336,  2.33008,   -0.194946,  0.473877,  -9.22656, 2.58789,   8.46094,
        6.95312,   -5.23047,  -0.549805, 5.53516,    -2.53516,  4.41406,  -7.74219,  -4.73828,
        -3.51953,  -5.45703,  7.875,     -8.01562,   -4.01562,  -9.04688, 5.53125,   -8.61719,
        5.5,       9.42188,   -5.66797,  -4.57812,   -9.64062,  5.79688,  -1.7041,   -3.07617,
        -4.66797,  -3.44336,  2.33008,   -0.194946,  0.473877,  -9.22656, 2.58789,   8.46094,
        -3.34375,  -1.7627,   -6.07422,  -9.24219,   -2.73242,  5.41406,  3.05859,   1.79297,
        2.70312,   3.33398,   -2.07617,  -4.47266,   -0.207642, 0.153931, -0.834961, -4.66406};
    std::vector<float> gold_k = {
        7.26172,  -5.59766,  -1.66797, -9.42188, 2.16406,    -3.23242, 5.25781,  -6.56641,
        8.35938,  -1.27148,  16.5156,  -2.81055, 16.2969,    -6.88281, -9.64062, -5.35938,
        2.53906,  1.52734,   1.02344,  -2.48633, 2.75781,    -13.2734, 4.26562,  1.79688,
        -5.81641, -12.7578,  -17.5469, -1.95312, 13.5391,    -1.77734, -3.98242, -10.7969,
        -5.46875, -15.1172,  -2.53125, -4.43359, -3.64453,   6.71484,  -0.25,    6.25391,
        -8.17188, -4.30078,  9.84375,  -9.82031, 14.4453,    8.8125,   10.7812,  -4.88672,
        -8.40625, 10.5781,   8.04688,  -0.1875,  -13.875,    -1.14062, -2.4375,  2.35352,
        8.82031,  7.46875,   3.90234,  11.1328,  -0.0664062, -17,      5.51562,  1.11816,
        -2.56836, -2.85156,  2.63672,  2.1543,   -6.99609,   -2.11719, -10.0234, 1.02539,
        -9.73438, 14.1172,   -6.72266, -9.09375, -12.4531,   11.4219,  -4.43359, 0.665527,
        3.35156,  0.179688,  -16.4688, 7.26953,  -5.64453,   -9.14062, 3.25781,  -1.87012,
        -9.57031, 19.4844,   -1.40234, -5.01172, 13.1953,    9.30469,  -8.20312, -4.50781,
        2.14648,  -14.1641,  10.8047,  10.9375,  -1.79785,   -3.55859, -10.9141, 12.0078,
        9.27344,  -2.59766,  4.27344,  -8.66406, 1.5625,     0.921875, 4.81641,  3.72266,
        -3.32617, -6.17578,  -11.2578, -11.0078, 1.75,       3.79688,  7.21094,  -13.1484,
        4.60156,  -3.24219,  4.67969,  -7.84766, -13.6953,   10.1328,  4.76953,  2.30469,
        1,        1,         1,        1,        1,          1,        1,        1,
        1,        1,         1,        1,        1,          1,        1,        1,
        1,        1,         1,        1,        1,          1,        1,        1,
        1,        1,         1,        1,        1,          1,        1,        1,
        -13.6406, -10.3438,  -2.88281, -5.66406, 4.96094,    -13.7031, -8.72656, -6.92969,
        -2.46875, 2.17578,   -13.8359, 12.1641,  14.8203,    1.02734,  2.9082,   9.1875,
        -3.16797, -5.08984,  2.74609,  -15.4844, 6.58594,    2.03516,  4.89062,  -1.19238,
        -14.0156, -1.9834,   -14.3281, 3.13672,  -1.60938,   14.3047,  -14.9531, 2.00195,
        5.63281,  -7.14453,  5.43359,  2.24805,  -10.7812,   -11.6719, 5.45703,  -3.25781,
        -7.98438, -7.21094,  -3.99609, -4.98828, 0.8125,     1.97266,  -3.25195, -3.50195,
        -4.58203, -2.77344,  -12.1406, -2.80469, -1.67969,   -7.71875, 14.4219,  0.828125,
        2.77344,  -4.05078,  -4.77734, 13.6797,  -10.8516,   -2.91406, 3.27734,  17.1719,
        -4.48438, 18.5625,   10.875,   9.29688,  -18.5,      1.34082,  8.11719,  15.875,
        1.15918,  -0.507812, -7.42969, 5.52734,  0.609375,   -3.2207,  9.14062,  3.07422,
        -13.8203, -0.632812, -9.42188, -1.87109, -4.78906,   1.28516,  1.25488,  -1.43555,
        -4.63281, -12.2734,  0.835938, 14.3906,  -3.54492,   13.1953,  4.70703,  6.96094,
        3.23242,  -5.52344,  12.0781,  2.69922,  4.10156,    8.26562,  4.57422,  13.4609,
        -9.67188, -10.5156,  5.21875,  -16.1875, -3.54883,   -9.52344, -1.81738, 2.97656,
        -3.04492, -10.7969,  10.2969,  -10,      5.86328,    2.57617,  -4.375,   -10.1172,
        -4.02344, -3.01562,  -2.00781, 3.91602,  12.1016,    7.25,     14.4062,  7.53516,
        1,        1,         1,        1,        1,          1,        1,        1,
        1,        1,         1,        1,        1,          1,        1,        1,
        1,        1,         1,        1,        1,          1,        1,        1,
        1,        1,         1,        1,        1,          1,        1,        1};
    std::vector<float> gold_v = {
        -5.27734,  -8.74219,  -9.92188,  7.54297,    -7.33984, 2.73047,   -2.63477, -9.71094,
        5.64062,   -6.34766,  5.39062,   -0.0889893, -9.89844, 1.39355,   9.28125,  1.88281,
        -1.45117,  2.24219,   -4.84375,  4.87109,    8.23438,  6.17188,   -7.87109, -8.53125,
        4.06641,   4.10156,   8.05469,   -1.88477,   -1.55078, -7.09766,  2.21289,  -2.60938,
        -4.5,      -0.932617, 3.39844,   -7.63281,   -5.40625, -5.71484,  1.62988,  -3.6875,
        -5.69922,  -0.234985, -5,        7.41797,    -0.95752, -8.77344,  0.35376,  -2.50781,
        -9.97656,  -1.72852,  -0.695801, -6.9375,    -9.15625, -8,        9.83594,  -9.82812,
        -5.86328,  4.36719,   -7.26562,  8.78125,    6.35156,  -9.45312,  5.94922,  4.5,
        5.5,       9.42188,   -5.66797,  -4.57812,   -9.64062, 5.79688,   -1.7041,  -3.07617,
        -4.66797,  -3.44336,  2.33008,   -0.194946,  0.473877, -9.22656,  2.58789,  8.46094,
        6.59766,   3.88281,   -3.06641,  -7.19141,   0.400879, 4.21484,   -0.84082, 6.75391,
        8.01562,   7.04297,   -5.32031,  -2.36133,   8.45312,  6.71094,   2.91797,  4.85938,
        1.52441,   -9.96875,  -8.96094,  -8.04688,   4.36328,  -0.325928, 5.66016,  -6.42188,
        9.125,     -0.827637, 7.10547,   -5.8125,    7.62109,  7.19531,   -2.04297, -0.941895,
        -5.48047,  -3.27539,  2.07422,   5.17188,    2.38867,  7.3125,    8.86719,  -0.612793,
        6.50391,   -0.297852, -1.44727,  2.75,       4.40625,  -1.61328,  -8.52344, -4.92578,
        1,         1,         1,         1,          1,        1,         1,        1,
        1,         1,         1,         1,          1,        1,         1,        1,
        1,         1,         1,         1,          1,        1,         1,        1,
        1,         1,         1,         1,          1,        1,         1,        1,
        3.09375,   -4.55469,  -7.85547,  -8.69531,   -9.21875, 7.72656,   -5.00781, -4.86328,
        7.47266,   5.55859,   7.19922,   -2.01367,   3.63281,  -1.8291,   -7.31641, -2.49805,
        4.25,      6.51172,   -0.36084,  8.13281,    3.66602,  8.74219,   -1.65527, -9.42188,
        -0.267822, 4.73828,   -7.57812,  6.93359,    5.19531,  7.10156,   0.734863, -3.17773,
        1.12598,   0.172974,  -8.19531,  -0.0739746, -2.24609, 6.70703,   -1.00586, 1.31152,
        -2.73438,  -2.82031,  5.6875,    -9.5,       1.7373,   -6.84375,  -3.88281, -5.38672,
        -8.20312,  0.342773,  -4.73438,  -9.66406,   2.16016,  3.67773,   9.14844,  6.81641,
        -0.891602, 1.66016,   -9.07812,  -6.33594,   -3.10547, 1.65039,   4.05469,  -6.30078,
        3.87109,   -5.71094,  -6.625,    -4.67578,   8.42188,  -7.37891,  -7.70312, -2.48633,
        6.73438,   2.14844,   4.46484,   -0.836914,  7.12891,  8.13281,   -6.47656, 8.70312,
        6.95312,   -5.23047,  -0.549805, 5.53516,    -2.53516, 4.41406,   -7.74219, -4.73828,
        -3.51953,  -5.45703,  7.875,     -8.01562,   -4.01562, -9.04688,  5.53125,  -8.61719,
        2.99023,   5.67188,   -3.26367,  1.86914,    4.97266,  1.44922,   1.99121,  -3.81641,
        -1.08594,  2.19922,   0.218994,  -3.00781,   -2.24609, -5.49219,  6.92969,  -1.71094,
        -2.35352,  7.58984,   9.30469,   -3.49219,   2.12891,  -1.15527,  1.03418,  6.22266,
        6.89453,   -5.84766,  -0.692871, -9.45312,   -4.72656, -3.92969,  0.929688, -1.24316,
        1,         1,         1,         1,          1,        1,         1,        1,
        1,         1,         1,         1,          1,        1,         1,        1,
        1,         1,         1,         1,          1,        1,         1,        1,
        1,         1,         1,         1,          1,        1,         1,        1};

    CHECK(migraphx::verify::verify_range_with_tolerance(result_vector,
                                                        migraphx::verify::expected{gold}));
    CHECK(migraphx::verify::verify_range_with_tolerance(pres_key_vector,
                                                        migraphx::verify::expected{gold_k}));
    CHECK(migraphx::verify::verify_range_with_tolerance(pres_val_vector,
                                                        migraphx::verify::expected{gold_v}));
}

constexpr auto make_models = [](gqa_program_params params, int slk) {
    auto prog_ck  = make_gqa_program(params, true);
    auto prog_ref = make_gqa_program(params, false);
    auto prog_mgx = prog_ref;
    migraphx::compile_options opts;
    opts.offload_copy = true;

    prog_ck.compile(migraphx::make_target("gpu"), opts);
    std::cout << "prog_ck: " << std::endl;
    std::cout << "===============================================================\n";
    std::cout << prog_ck << std::endl;
    std::cout << "===============================================================\n";
    prog_mgx.compile(migraphx::make_target("gpu"), opts);
    std::cout << "prog_mgx: " << std::endl;
    std::cout << "===============================================================\n";
    std::cout << prog_mgx << std::endl;
    std::cout << "===============================================================\n";

    migraphx::shape qkv_shape{
        migraphx::shape::half_type,
        {params.batch, params.seqlen, params.head_size * (params.nhead + 2 * params.nhead_kv)}};
    migraphx::shape past_key_values_shape{
        migraphx::shape::half_type,
        {params.batch, params.nhead_kv, params.seqlen_old, params.head_size}};
    migraphx::shape slk_shape{migraphx::shape::int32_type, {params.batch}};
    migraphx::shape trig_cache_shape{migraphx::shape::half_type,
                                     {params.seqlen_old, params.head_size / 2}};

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::uniform_real_distribution<float> cos_sin_dist(-1.0f, 1.0f);

    std::vector<float> qkv_data(qkv_shape.elements());
    std::generate(qkv_data.begin(), qkv_data.end(), [&]() { return dist(rng); });
    //
    std::vector<float> past_key_values_data(past_key_values_shape.elements());
    std::generate(
        past_key_values_data.begin(), past_key_values_data.end(), [&]() { return dist(rng); });
    std::vector<float> past_val_values_data(past_key_values_shape.elements());
    std::generate(
        past_val_values_data.begin(), past_val_values_data.end(), [&]() { return dist(rng); });
    //
    std::vector<float> trig_cache_data(trig_cache_shape.elements());
    std::generate(
        trig_cache_data.begin(), trig_cache_data.end(), [&]() { return cos_sin_dist(rng); });

    // TODO determine what the contents should be
    std::vector<int> slk_data(slk_shape.elements(), slk);

    migraphx::literal qkv{qkv_shape, qkv_data};
    migraphx::literal past_key_values_key{past_key_values_shape, past_key_values_data};
    migraphx::literal past_key_values_value{past_key_values_shape, past_val_values_data};
    migraphx::literal seqlens_k{slk_shape, slk_data};
    migraphx::literal cos_cache{trig_cache_shape, trig_cache_data};
    migraphx::literal sin_cache{trig_cache_shape, trig_cache_data};

    migraphx::parameter_map pp;
    pp["qkv"]       = qkv.get_argument();
    pp["k_old"]     = past_key_values_key.get_argument();
    pp["v_old"]     = past_key_values_value.get_argument();
    pp["slk"]       = seqlens_k.get_argument();
    pp["cos_cache"] = cos_cache.get_argument();
    pp["sin_cache"] = sin_cache.get_argument();

    std::vector<float> mgx_result_vector;
    std::vector<float> mgx_pres_key_vector;
    std::vector<float> mgx_pres_val_vector;
    {
        auto outputs       = prog_mgx.eval(pp);
        const auto& result = outputs.front();
        result.visit([&](auto output) { mgx_result_vector.assign(output.begin(), output.end()); });
        const auto& pres_key = outputs.at(1);
        pres_key.visit(
            [&](auto output) { mgx_pres_key_vector.assign(output.begin(), output.end()); });
        const auto& pres_val = outputs.back();
        pres_val.visit(
            [&](auto output) { mgx_pres_val_vector.assign(output.begin(), output.end()); });
    }

    std::vector<float> ck_result_vector;
    std::vector<float> ck_pres_key_vector;
    std::vector<float> ck_pres_val_vector;
    {
        auto ck_outputs       = prog_ck.eval(pp);
        const auto& result_ck = ck_outputs.front();
        result_ck.visit(
            [&](auto output) { ck_result_vector.assign(output.begin(), output.end()); });
        const auto& pres_key_ck = ck_outputs.at(1);
        pres_key_ck.visit(
            [&](auto output) { ck_pres_key_vector.assign(output.begin(), output.end()); });
        const auto& ck_pres_val = ck_outputs.back();
        ck_pres_val.visit(
            [&](auto output) { ck_pres_val_vector.assign(output.begin(), output.end()); });
    }

    std::cout << "ck_result_vector: " << std::endl;
    for(auto i = 0; i < 1024; ++i)
    {
        std::cout << static_cast<float>(ck_result_vector[i]) << " ";
    }
    std::cout << std::endl;
    std::cout << "mgx_result_vector: " << std::endl;
    for(auto i = 0; i < 1024; ++i)
    {
        std::cout << static_cast<float>(mgx_result_vector[i]) << " ";
    }
    std::cout << std::endl;
    EXPECT(migraphx::verify::verify_range_with_tolerance(
        ck_pres_key_vector, migraphx::verify::expected{mgx_pres_key_vector}));
    EXPECT(migraphx::verify::verify_range_with_tolerance(
        ck_pres_val_vector, migraphx::verify::expected{mgx_pres_val_vector}));
    EXPECT(migraphx::verify::verify_range_with_tolerance(
        ck_result_vector, migraphx::verify::expected{mgx_result_vector}));
};

TEST_CASE(gqa_gen)
{
    std::vector<std::pair<size_t, size_t>> seqlens{{1, 1024}, {1024, 2048}, {2048, 4096}, {4096, 8192}};
    std::vector<size_t> head_sizes{16, 32, 64, 128, 256};
    bool stop = false;
    for(const auto& seqlen : seqlens)
    {
        for(const auto& head_size : head_sizes)
        {
            if(stop) return;
            gqa_program_params params;
            params.batch = 1;
            params.nhead = 2;
            params.nhead_kv = 2;
            params.seqlen = seqlen.first;
            params.seqlen_old = seqlen.second;
            params.head_size = head_size;
            params.local_window_size = 4;
            params.scale = 1.0;
            params.do_rotary = true;
            params.rotary_interleaved = false;
            make_models(params, seqlen.second - seqlen.first - 1);
            stop = false;
        }
    }
}
