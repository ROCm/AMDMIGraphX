/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

template <migraphx::shape::type_t DType>
struct test_attention_flash_decoding_3d_output_fusion
    : verify_program<test_attention_flash_decoding_3d_output_fusion<DType>>
{
    migraphx::program create_program() const
    {
        // 3D Shape: [batch, sequence_length, head_dim]
        migraphx::shape s_3d{DType, {1, 256, 256}};

        migraphx::program p1;
        auto* mm = p1.get_main_module();

        // Input parameters for attention
        auto q = mm->add_parameter("q", s_3d);
        auto k = mm->add_parameter("k", s_3d);
        auto v = mm->add_parameter("v", s_3d);

        // Parameters for output fusion
        // Output projection weight matrix
        migraphx::shape proj_weight_shape{DType, {256, 256}};
        auto output_proj_weight = mm->add_parameter("output_proj_weight", proj_weight_shape);

        // Output bias
        migraphx::shape bias_shape{DType, {256}};
        auto output_bias = mm->add_parameter("output_bias", bias_shape);

        // Residual input for skip connection
        auto residual = mm->add_parameter("residual", s_3d);

        // Layer norm parameters (gamma and beta)
        auto ln_gamma = mm->add_parameter("ln_gamma", bias_shape);
        auto ln_beta  = mm->add_parameter("ln_beta", bias_shape);

        // Gate for gated output
        auto output_gate = mm->add_parameter("output_gate", s_3d);

        // Standard attention mechanism (no input fusion)
        // Transpose K and V for matrix multiplication
        auto k_transposed =
            mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), k);
        auto v_transposed =
            mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), v);

        // Compute attention scores: Q @ K^T
        auto scores = mm->add_instruction(migraphx::make_op("dot"), q, k_transposed);

        // Apply softmax
        auto scores_max =
            mm->add_instruction(migraphx::make_op("reduce_max", {{"axes", {2}}}), scores);
        scores_max = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s_3d.lens()}}), scores_max);
        auto scores_sub = mm->add_instruction(migraphx::make_op("sub"), scores, scores_max);
        auto scores_exp = mm->add_instruction(migraphx::make_op("exp"), scores_sub);
        auto scores_sum =
            mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), scores_exp);
        scores_sum = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s_3d.lens()}}), scores_sum);
        auto attention_weights =
            mm->add_instruction(migraphx::make_op("div"), scores_exp, scores_sum);

        // Apply attention weights to values: attention_weights @ V^T
        auto attention_output =
            mm->add_instruction(migraphx::make_op("dot"), attention_weights, v_transposed);

        // OUTPUT FUSION OPERATIONS START HERE

        // 1. Output projection (linear transformation)
        // Reshape for matrix multiplication with projection weight
        auto attn_reshaped = mm->add_instruction(
            migraphx::make_op("reshape", {{"dims", {256, 256}}}), attention_output);
        auto projected =
            mm->add_instruction(migraphx::make_op("dot"), attn_reshaped, output_proj_weight);
        projected =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", s_3d.lens()}}), projected);

        // 2. Add output bias
        output_bias = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s_3d.lens()}}), output_bias);
        auto with_bias = mm->add_instruction(migraphx::make_op("add"), projected, output_bias);

        // 3. Apply dropout-like operation (using a gate for deterministic testing)
        auto gate_sigmoid = mm->add_instruction(migraphx::make_op("sigmoid"), output_gate);
        auto gated        = mm->add_instruction(migraphx::make_op("mul"), with_bias, gate_sigmoid);

        // 4. Add residual connection
        auto with_residual = mm->add_instruction(migraphx::make_op("add"), gated, residual);

        // 5. Layer normalization
        // Compute mean
        auto mean =
            mm->add_instruction(migraphx::make_op("reduce_mean", {{"axes", {2}}}), with_residual);
        mean = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", s_3d.lens()}}),
                                   mean);
        auto centered = mm->add_instruction(migraphx::make_op("sub"), with_residual, mean);

        // Compute variance
        auto squared = mm->add_instruction(migraphx::make_op("mul"), centered, centered);
        auto variance =
            mm->add_instruction(migraphx::make_op("reduce_mean", {{"axes", {2}}}), squared);
        variance = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s_3d.lens()}}), variance);

        // Add epsilon for numerical stability
        migraphx::shape epsilon_shape{DType, {1}};
        auto epsilon = mm->add_literal(migraphx::literal{epsilon_shape, {1e-5f}});
        epsilon      = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s_3d.lens()}}), epsilon);
        auto var_plus_eps = mm->add_instruction(migraphx::make_op("add"), variance, epsilon);

        // Compute standard deviation
        auto std_dev = mm->add_instruction(migraphx::make_op("sqrt"), var_plus_eps);

        // Normalize
        auto normalized = mm->add_instruction(migraphx::make_op("div"), centered, std_dev);

        // Scale and shift
        ln_gamma = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s_3d.lens()}}), ln_gamma);
        ln_beta = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s_3d.lens()}}), ln_beta);
        auto scaled    = mm->add_instruction(migraphx::make_op("mul"), normalized, ln_gamma);
        auto ln_output = mm->add_instruction(migraphx::make_op("add"), scaled, ln_beta);

        // 6. Final activation (ReLU)
        auto final_output = mm->add_instruction(migraphx::make_op("relu"), ln_output);

        mm->add_return({final_output});
        return p1;
    }
};

// These tests are not run by default currently; the env vars below need to be set:
// MIGRAPHX_FLASH_DECODING_NUM_SPLITS=2 # or another split factor
// MIGRAPHX_MLIR_USE_SPECIFIC_OPS=attention
template struct test_attention_flash_decoding_3d_output_fusion<migraphx::shape::half_type>;
template struct test_attention_flash_decoding_3d_output_fusion<migraphx::shape::bf16_type>;
template struct test_attention_flash_decoding_3d_output_fusion<migraphx::shape::float_type>;
