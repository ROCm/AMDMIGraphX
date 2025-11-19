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
struct test_attention_flash_decoding_3d_input_fusion
    : verify_program<test_attention_flash_decoding_3d_input_fusion<DType>>
{
    migraphx::program create_program() const
    {
        // 3D Shape: [batch, sequence_length, head_dim]
        migraphx::shape s_3d{DType, {1, 256, 256}};
        
        migraphx::program p1;
        auto* mm = p1.get_main_module();
        
        // Input parameters
        auto q_input = mm->add_parameter("q", s_3d);
        auto k_input = mm->add_parameter("k", s_3d);
        auto v_input = mm->add_parameter("v", s_3d);
        
        // Bias parameters for input fusion
        auto q_bias = mm->add_parameter("q_bias", s_3d);
        auto k_bias = mm->add_parameter("k_bias", s_3d);
        auto v_bias = mm->add_parameter("v_bias", s_3d);
        
        // Scale parameter (typically 1/sqrt(head_dim))
        migraphx::shape scale_shape{DType, {1}};
        auto scale = mm->add_parameter("scale", scale_shape);
        
        // Input fusion operations
        // Add bias to Q, K, V
        auto q_with_bias = mm->add_instruction(migraphx::make_op("add"), q_input, q_bias);
        auto k_with_bias = mm->add_instruction(migraphx::make_op("add"), k_input, k_bias);
        auto v_with_bias = mm->add_instruction(migraphx::make_op("add"), v_input, v_bias);
        
        // Scale Q (common in attention mechanisms)
        scale = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s_3d.lens()}}), scale);
        auto q_scaled = mm->add_instruction(migraphx::make_op("mul"), q_with_bias, scale);
        
        // Apply activation (optional input fusion)
        auto q_activated = mm->add_instruction(migraphx::make_op("tanh"), q_scaled);
        auto k_activated = mm->add_instruction(migraphx::make_op("tanh"), k_with_bias);
        auto v_activated = mm->add_instruction(migraphx::make_op("tanh"), v_with_bias);
        
        // Now perform the attention mechanism with fused inputs
        // Transpose K and V for matrix multiplication
        auto k_transposed = mm->add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), k_activated);
        auto v_transposed = mm->add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), v_activated);
        
        // Compute attention scores: Q @ K^T
        auto scores = mm->add_instruction(migraphx::make_op("dot"), q_activated, k_transposed);
        
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
        auto output =
            mm->add_instruction(migraphx::make_op("dot"), attention_weights, v_transposed);
        
        mm->add_return({output});
        return p1;
    }
};

// These tests are not run by default currently; the env vars below need to be set:
// MIGRAPHX_FLASH_DECODING_NUM_SPLITS=2 # or another split factor
// MIGRAPHX_MLIR_USE_SPECIFIC_OPS=attention
template struct test_attention_flash_decoding_3d_input_fusion<migraphx::shape::half_type>;
template struct test_attention_flash_decoding_3d_input_fusion<migraphx::shape::bf16_type>;
template struct test_attention_flash_decoding_3d_input_fusion<migraphx::shape::float_type>;
