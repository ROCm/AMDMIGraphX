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

/*
 * SkipSimplifiedLayerNorm FP32 kernel for GPT-OSS-20B correctness.
 *
 * Mirrors the DML validated reference (ComputeSkipSLNCPU in dml/hip_qmoe/
 * qmoe_hip_combined_op.cpp, lines 2178-2263): mean_sq in FP32, inv_std in FP32,
 * x*inv_std*gamma all in FP32. Only the output is converted to fp16.
 *
 * Root cause this fixes: MIGraphX's fuse_pointwise_reduce merges the SLN ops
 * into a single kernel where the gamma multiply reverts to fp16 because gamma
 * is a shared fp16 weight tensor. This custom kernel keeps everything in FP32.
 *
 * Kernel: one block per token, blockDim.x threads reduce hidden_size elements.
 * Uses warp shuffle for the variance reduction.
 */
#ifndef MIGRAPHX_GUARD_KERNELS_SKIP_SIMPLIFIED_LAYER_NORM_HPP
#define MIGRAPHX_GUARD_KERNELS_SKIP_SIMPLIFIED_LAYER_NORM_HPP

#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/tensor_view.hpp>

namespace migraphx {

/*
 * skip_simplified_layer_norm<BLOCK_SIZE>(input, skip, gamma, output, eps, hidden_size)
 *
 * All of mean_sq, inv_std, x*inv_std*gamma computed in FP32.
 * Input/output in fp16; gamma in fp16 (converted to fp32 inside kernel).
 *
 * Launch: gridDim.x = num_tokens, blockDim.x = BLOCK_SIZE (e.g. 256)
 */
template <index_int BLOCK_SIZE, class Input, class Skip, class Gamma, class Output>
__device__ void skip_simplified_layer_norm(const Input  input,
                                           const Skip   skip,
                                           const Gamma  gamma,
                                           Output       output,
                                           float        eps,
                                           index_int    hidden_size)
{
    const index_int token_idx  = blockIdx.x;
    const index_int thread_idx = threadIdx.x;

    // Shared memory for warp-level reduction of sum_sq
    __shared__ float shmem[BLOCK_SIZE / 32]; // one slot per warp

    // Each thread accumulates sum of (x+skip)^2 over its chunk
    float local_sum_sq = 0.0f;
    for(index_int i = thread_idx; i < hidden_size; i += BLOCK_SIZE)
    {
        float x_val  = __half2float(input[token_idx * hidden_size + i]);
        float sk_val = __half2float(skip[token_idx * hidden_size + i]);
        float v      = x_val + sk_val;
        local_sum_sq += v * v;
    }

    // Warp-level reduction
    for(int offset = 16; offset > 0; offset >>= 1)
        local_sum_sq += __shfl_xor(local_sum_sq, offset);

    if((thread_idx & 31) == 0) // lane 0 of each warp writes to shared
        shmem[thread_idx >> 5] = local_sum_sq;
    __syncthreads();

    // Block-level reduction in first warp
    float block_sum_sq = 0.0f;
    if(thread_idx < (BLOCK_SIZE / 32))
        block_sum_sq = shmem[thread_idx];
    for(int offset = (BLOCK_SIZE / 64); offset > 0; offset >>= 1)
        block_sum_sq += __shfl_xor(block_sum_sq, offset);

    // Broadcast inv_std to all threads
    float inv_std = __shfl(1.0f / sqrtf(block_sum_sq / (float)hidden_size + eps), 0);

    // Each thread applies normalization in FP32 and writes fp16 output
    for(index_int i = thread_idx; i < hidden_size; i += BLOCK_SIZE)
    {
        float x_val  = __half2float(input[token_idx * hidden_size + i]);
        float sk_val = __half2float(skip[token_idx * hidden_size + i]);
        float g_val  = __half2float(gamma[i]); // gamma converted to FP32 here
        float result = (x_val + sk_val) * inv_std * g_val;
        output[token_idx * hidden_size + i] = __float2half(result);
    }
}

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_SKIP_SIMPLIFIED_LAYER_NORM_HPP
