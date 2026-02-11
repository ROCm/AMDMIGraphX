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
#include <migraphx/instruction.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/program.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/verify.hpp>
#include <migraphx/load_save.hpp>

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

// Quick test for CK FMHA forward path.
// Builds: O = softmax(Q @ K^T * scale) @ V with 4D row-major tensors
// so that the prefuse_ops FMHA matcher fires (innermost stride == 1).
TEST_CASE(ck_fmha_fwd_test)
{
    // All different dims, batch > 1
    const std::size_t batch = 2;
    const std::size_t nhead = 4;
    const std::size_t M     = 512; // seqlen_q
    const std::size_t N     = 512; // seqlen_k
    const std::size_t K     = 32;  // hdim_q = hdim_k
    const std::size_t O     = 32;  // hdim_v

    migraphx::program p;
    auto* mm = p.get_main_module();

    // Q: [batch, nhead, M, K] — row-major, innermost stride = 1
    migraphx::shape q_s{migraphx::shape::half_type, {batch, nhead, M, K}};
    // K: [batch, nhead, N, K] — row-major
    migraphx::shape k_s{migraphx::shape::half_type, {batch, nhead, N, K}};
    // V: [batch, nhead, N, O] — row-major
    migraphx::shape v_s{migraphx::shape::half_type, {batch, nhead, N, O}};

    auto q = mm->add_parameter("q", q_s);
    auto k = mm->add_parameter("k", k_s);
    auto v = mm->add_parameter("v", v_s);

    // K needs to be transposed for dot(Q, K^T): [batch, nhead, N, K] -> [batch, nhead, K, N]
    auto kt =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), k);

    // gemm1 = Q @ K^T  -> [batch, nhead, M, N]
    auto gemm1 = mm->add_instruction(migraphx::make_op("dot"), q, kt);

    // scale = 1/sqrt(K)
    float scale_val = 1.0f / std::sqrt(static_cast<float>(K));
    auto scale_lit  = mm->add_literal(
        migraphx::literal{migraphx::shape{migraphx::shape::half_type, {1}}, {scale_val}});
    auto scale = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", gemm1->get_shape().lens()}}), scale_lit);
    auto scaled = mm->add_instruction(migraphx::make_op("mul"), gemm1, scale);

    // softmax along last axis
    auto softmax = mm->add_instruction(migraphx::make_op("softmax", {{"axis", -1}}), scaled);

    // gemm2 = softmax @ V -> [batch, nhead, M, O]
    auto gemm2 = mm->add_instruction(migraphx::make_op("dot"), softmax, v);
    mm->add_return({gemm2});

    // Compile for GPU and run

    // Generate random input data
    migraphx::parameter_map params;
    params["q"] = migraphx::generate_argument(q_s, 0);
    params["k"] = migraphx::generate_argument(k_s, 1);
    params["v"] = migraphx::generate_argument(v_s, 2);

    auto gpu_p = p;
    p.compile(migraphx::make_target("ref"));
    std::cout << p << std::endl;
    auto result = p.eval(params).back();
    std::vector<migraphx::half> reference_result;
    result.visit([&](auto output) { reference_result.assign(output.begin(), output.end()); });

    migraphx::compile_options gpu_opts;
    gpu_opts.offload_copy    = true;
    gpu_opts.exhaustive_tune = true;
    gpu_p.compile(migraphx::make_target("gpu"), gpu_opts);
    std::cout << gpu_p << std::endl;
    auto gpu_result = gpu_p.eval(params).back();
    std::vector<migraphx::half> gpu_result_vector;
    gpu_result.visit([&](auto output) { gpu_result_vector.assign(output.begin(), output.end()); });

    // If we get here without crashing, the kernel compiled and ran
    EXPECT(result.get_shape().lens() == std::vector<std::size_t>{batch, nhead, M, O});
    EXPECT(migraphx::verify::verify_rms_range(reference_result, gpu_result_vector));
}

// FMHA with bias: O = softmax(Q @ K^T * scale + bias) @ V
TEST_CASE(ck_fmha_fwd_bias_test)
{
    const std::size_t batch = 2;
    const std::size_t nhead = 4;
    const std::size_t M     = 128; // seqlen_q
    const std::size_t N     = 256; // seqlen_k
    const std::size_t K     = 64;  // hdim_q = hdim_k
    const std::size_t O     = 32;  // hdim_v

    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape q_s{migraphx::shape::half_type, {batch, nhead, M, K}};
    migraphx::shape k_s{migraphx::shape::half_type, {batch, nhead, N, K}};
    migraphx::shape v_s{migraphx::shape::half_type, {batch, nhead, N, O}};
    migraphx::shape bias_s{migraphx::shape::half_type, {batch, nhead, M, N}};

    auto q    = mm->add_parameter("q", q_s);
    auto k    = mm->add_parameter("k", k_s);
    auto v    = mm->add_parameter("v", v_s);
    auto bias = mm->add_parameter("bias", bias_s);

    // K transposed for dot(Q, K^T)
    auto kt =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), k);

    // gemm1 = Q @ K^T -> [batch, nhead, M, N]
    auto gemm1 = mm->add_instruction(migraphx::make_op("dot"), q, kt);

    // scale
    float scale_val = 1.0f / std::sqrt(static_cast<float>(K));
    auto scale_lit  = mm->add_literal(
        migraphx::literal{migraphx::shape{migraphx::shape::half_type, {1}}, {scale_val}});
    auto scale = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", gemm1->get_shape().lens()}}), scale_lit);
    auto scaled = mm->add_instruction(migraphx::make_op("mul"), gemm1, scale);

    // Add bias
    auto biased = mm->add_instruction(migraphx::make_op("add"), scaled, bias);

    // softmax
    auto softmax = mm->add_instruction(migraphx::make_op("softmax", {{"axis", -1}}), biased);

    // gemm2 = softmax @ V -> [batch, nhead, M, O]
    auto gemm2 = mm->add_instruction(migraphx::make_op("dot"), softmax, v);
    mm->add_return({gemm2});

    // Generate random input data
    migraphx::parameter_map params;
    params["q"]    = migraphx::generate_argument(q_s, 0);
    params["k"]    = migraphx::generate_argument(k_s, 1);
    params["v"]    = migraphx::generate_argument(v_s, 2);
    params["bias"] = migraphx::generate_argument(bias_s, 3);

    auto gpu_p = p;
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval(params).back();
    std::vector<migraphx::half> reference_result;
    result.visit([&](auto output) { reference_result.assign(output.begin(), output.end()); });

    migraphx::compile_options gpu_opts;
    gpu_opts.offload_copy    = true;
    gpu_opts.exhaustive_tune = true;
    gpu_p.compile(migraphx::make_target("gpu"), gpu_opts);
    std::cout << gpu_p << std::endl;
    auto gpu_result = gpu_p.eval(params).back();
    std::vector<migraphx::half> gpu_result_vector;
    gpu_result.visit([&](auto output) { gpu_result_vector.assign(output.begin(), output.end()); });

    EXPECT(result.get_shape().lens() == std::vector<std::size_t>{batch, nhead, M, O});
    EXPECT(migraphx::verify::verify_rms_range(reference_result, gpu_result_vector));
}

TEST_CASE(attention_models)
{
    const std::size_t batch = 2;
    const std::size_t nhead = 4;

    std::vector<size_t> seqlens_q{512, 1024};
    std::vector<size_t> seqlens_k{512, 1024};
    std::vector<size_t> hdims_q{32, 64, 96};
    std::vector<size_t> hdims_v{32, 64, 96};

    for(const auto& seqlen_q : seqlens_q)
    {
        for(const auto& seqlen_k : seqlens_k)
        {
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

                    migraphx::shape q_s{migraphx::shape::half_type, {batch, nhead, M, K}};
                    migraphx::shape k_s{migraphx::shape::half_type, {batch, nhead, N, K}};
                    migraphx::shape v_s{migraphx::shape::half_type, {batch, nhead, N, O}};
                    migraphx::shape bias_s{migraphx::shape::half_type, {batch, nhead, M, N}};

                    auto q = mm->add_parameter("q", q_s);
                    auto k = mm->add_parameter("k", k_s);
                    auto v = mm->add_parameter("v", v_s);

                    // K needs to be transposed for dot(Q, K^T): [batch, nhead, N, K] -> [batch,
                    // nhead, K, N]
                    auto kt = mm->add_instruction(
                        migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), k);

                    // gemm1 = Q @ K^T  -> [batch, nhead, M, N]
                    auto gemm1 = mm->add_instruction(migraphx::make_op("dot"), q, kt);

                    // scale = 1/sqrt(K)
                    float scale_val = 1.0f / std::sqrt(static_cast<float>(K));
                    auto scale_lit  = mm->add_literal(migraphx::literal{
                        migraphx::shape{migraphx::shape::half_type, {1}}, {scale_val}});
                    auto scale      = mm->add_instruction(
                        migraphx::make_op("multibroadcast",
                                          {{"out_lens", gemm1->get_shape().lens()}}),
                        scale_lit);
                    auto scaled = mm->add_instruction(migraphx::make_op("mul"), gemm1, scale);

                    // softmax along last axis
                    auto softmax =
                        mm->add_instruction(migraphx::make_op("softmax", {{"axis", -1}}), scaled);

                    // gemm2 = softmax @ V -> [batch, nhead, M, O]
                    auto gemm2 = mm->add_instruction(migraphx::make_op("dot"), softmax, v);
                    mm->add_return({gemm2});

                    auto gpu_p = p;
                    migraphx::compile_options gpu_opts;
                    // gpu_opts.offload_copy    = true;
                    gpu_opts.exhaustive_tune = true;
                    std::stringstream ss;
                    ss << "saved_models/ck_" << batch << "_" << nhead << "_" << M << "_" << N
                       << "_" << K << "_" << O << ".mxr";
                    std::string output_filename = ss.str();
                    std::cout << "Compiling " << output_filename << std::endl;
                    auto start_time = std::chrono::high_resolution_clock::now();
                    gpu_p.compile(migraphx::make_target("gpu"), gpu_opts);
                    auto end_time = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> elapsed = end_time - start_time;
                    std::cout << "Finished compiling " << output_filename << " in "
                              << elapsed.count() << " seconds" << std::endl;
                    migraphx::save(gpu_p, output_filename);
                }
            }
        }
    }
}

TEST_CASE(combinations)
{
    auto run_fmha_test = [](std::size_t batch,
                            std::size_t nhead,
                            std::size_t M,
                            std::size_t N,
                            std::size_t K,
                            std::size_t O) {
        std::cout << "Running FMHA test with batch=" << batch << ", nhead=" << nhead << ", M=" << M
                  << ", N=" << N << ", K=" << K << ", O=" << O << std::endl;
        migraphx::program p;
        auto* mm = p.get_main_module();

        migraphx::shape q_s{migraphx::shape::half_type, {batch, nhead, M, K}};
        migraphx::shape k_s{migraphx::shape::half_type, {batch, nhead, N, K}};
        migraphx::shape v_s{migraphx::shape::half_type, {batch, nhead, N, O}};

        auto q = mm->add_parameter("q", q_s);
        auto k = mm->add_parameter("k", k_s);
        auto v = mm->add_parameter("v", v_s);

        auto kt =
            mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), k);
        auto gemm1 = mm->add_instruction(migraphx::make_op("dot"), q, kt);

        float scale_val = 1.0f / std::sqrt(static_cast<float>(K));
        auto scale_lit  = mm->add_literal(
            migraphx::literal{migraphx::shape{migraphx::shape::half_type, {1}}, {scale_val}});
        auto scale = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", gemm1->get_shape().lens()}}),
            scale_lit);
        auto scaled = mm->add_instruction(migraphx::make_op("mul"), gemm1, scale);

        auto softmax = mm->add_instruction(migraphx::make_op("softmax", {{"axis", -1}}), scaled);
        auto gemm2   = mm->add_instruction(migraphx::make_op("dot"), softmax, v);
        mm->add_return({gemm2});

        migraphx::parameter_map params;
        auto names = p.get_parameter_names();
        for(std::size_t i = 0; i < names.size(); ++i)
            params[names[i]] = migraphx::generate_argument(p.get_parameter_shape(names[i]), 42 + i);

        auto gpu_p = p;
        p.compile(migraphx::make_target("ref"));
        auto result = p.eval(params).back();
        std::vector<migraphx::half> reference_result;
        result.visit([&](auto output) { reference_result.assign(output.begin(), output.end()); });

        migraphx::compile_options gpu_opts;
        gpu_opts.offload_copy    = true;
        gpu_opts.exhaustive_tune = true;
        gpu_p.compile(migraphx::make_target("gpu"), gpu_opts);
        auto gpu_result = gpu_p.eval(params).back();
        std::vector<migraphx::half> gpu_result_vector;
        gpu_result.visit(
            [&](auto output) { gpu_result_vector.assign(output.begin(), output.end()); });

        EXPECT(migraphx::verify::verify_rms_range(reference_result, gpu_result_vector));
    };

    std::vector<std::size_t> batches{1, 2, 16};
    std::vector<std::size_t> nheads{4, 8, 16};
    std::vector<std::size_t> seqlens_q{512, 1024};
    std::vector<std::size_t> seqlens_k{512, 1024};
    std::vector<std::size_t> hdims_q{32, 64, 96};
    std::vector<std::size_t> hdims_v{32, 64, 96};

    for(auto batch : batches)
        for(auto nhead : nheads)
            for(auto M : seqlens_q)
                for(auto N : seqlens_k)
                    for(auto K : hdims_q)
                        for(auto O : hdims_v)
                            run_fmha_test(batch, nhead, M, N, K, O);
}
