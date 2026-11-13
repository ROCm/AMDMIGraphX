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
#ifndef MIGRAPHX_GUARD_GPU_DEVICE_MOE_HPP
#define MIGRAPHX_GUARD_GPU_DEVICE_MOE_HPP

#include <migraphx/argument.hpp>
#include <migraphx/gpu/device/config.hpp>
#include <hip/hip_runtime_api.h>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

// GPT-OSS sparse top-4 MoE forward.
//
// Ported from dml/hip_qmoe (topk_moe.h + moe_kernels.h), validated standalone
// (test_moe_kernels 5/5, test_topk_moe 3/3) and benchmarked (bench_moe_stack:
// 6.55 ms/token, 228 GB/s on gfx1151).
//
// Computes, for a single decode/prefill call:
//   1. top-k routing from router_logits (softmax over selected experts)
//   2. for each selected expert: fused gather + FC1 + SwiGLU + FC2 + weighted mix
//
// Activations are FP32 (the op converts FP16<->FP32 around this call). INT4
// weights are MatMulNBits-packed in the kernel-native layout:
//   fc1_weights [E, N_fc1, K/8] uint32 ; fc1_scales [E, N_fc1, K/32] f32
//   fc2_weights [E, N_fc2, K/8] uint32 ; fc2_scales [E, N_fc2, K/32] f32
// where N_fc1 = 2*intermediate (interleaved gate/up), N_fc2 = hidden.
//
// `output` is written (not accumulated by the caller); this routine zeroes it.
// All `argument`s are device buffers. Scratch buffers are caller-allocated.
struct moe_params
{
    int num_tokens;
    int hidden_size;       // 2880
    int intermediate_size; // 2880
    int num_experts;       // 32
    int top_k;             // 4
    int max_tokens_per_expert;
    float swiglu_alpha; // 1.702
    float swiglu_beta;  // 1.0
    float swiglu_limit; // 7.0
};

void MIGRAPHX_DEVICE_EXPORT gptoss_moe(hipStream_t stream,
                                       const argument& output,        // [S, hidden] f32
                                       const argument& hidden_states, // [S, hidden] f32
                                       const argument& router_logits, // [S, experts] f32
                                       const argument& fc1_weights,   // [E,N_fc1,K/8] u32
                                       const argument& fc1_scales,    // [E,N_fc1,K/32] f32
                                       const argument& fc2_weights,   // [E,N_fc2,K/8] u32
                                       const argument& fc2_scales,    // [E,N_fc2,K/32] f32
                                       const argument& fc1_bias,      // [E,N_fc1] f32 (gate_up_proj.bias)
                                       const argument& fc2_bias,      // [E,N_fc2] f32 (down_proj.bias)
                                       // scratch (caller-allocated, device)
                                       const argument& topk_weights,        // [S,top_k] f32
                                       const argument& topk_expert_ids,     // [S,top_k] i32
                                       const argument& expert_token_ids,    // [E,maxtok] i32
                                       const argument& expert_token_counts, // [E] i32
                                       const moe_params& params);

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
