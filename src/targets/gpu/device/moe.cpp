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
#include <migraphx/gpu/device/moe.hpp>
#include <migraphx/errors.hpp>
#include <hip/hip_runtime.h>
#include <vector>
#include <cstdint>
#include <cstdio>

// First-light diagnostics: surface HIP errors with location instead of a silent
// segfault. (Remove / downgrade once the op is proven.)
#define MOE_HIP_CHECK(call)                                                       \
    do                                                                            \
    {                                                                             \
        hipError_t _e = (call);                                                   \
        if(_e != hipSuccess)                                                      \
        {                                                                         \
            std::fprintf(stderr, "[gptoss_moe] HIP error %d (%s) at %s:%d: %s\n", \
                         (int)_e, hipGetErrorString(_e), __FILE__, __LINE__, #call); \
            std::fflush(stderr);                                                  \
        }                                                                         \
    } while(0)

// Kernels ported verbatim (logic-preserving) from dml/hip_qmoe:
//   topk_moe.h         -> moe_topk_kernel
//   moe_kernels.h      -> moe_q4_swiglu_indirect_kernel, moe_q4_accum_warp_kernel
// Validated standalone (test_moe_kernels 5/5, test_topk_moe 3/3) and benchmarked
// (bench_moe_stack: 6.55 ms/token, 228 GB/s on gfx1151). See
// gptoss/docs/GPTOSS_PERF_OPTIMIZATION.md (Route A / Route B sections).

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

namespace {

constexpr int MOE_WARP_SIZE        = 32;
constexpr int TOPK_WARPS_PER_BLOCK = 4;
constexpr int MM_WARPS_PER_BLOCK   = 8;

// ---------------- top-k routing (from topk_moe.h) ----------------
template <int N_EXPERTS, int TOP_K>
__global__ void __launch_bounds__(TOPK_WARPS_PER_BLOCK* MOE_WARP_SIZE, 1) moe_topk_kernel(
    const float* __restrict__ router_logits,
    float* __restrict__ topk_weights,
    int32_t* __restrict__ topk_expert_ids,
    int32_t* __restrict__ expert_token_ids,
    int32_t* __restrict__ expert_token_counts,
    const int num_tokens,
    const int max_tokens_per_expert)
{
    const int warp_id_global = (blockIdx.x * blockDim.x + threadIdx.x) / MOE_WARP_SIZE;
    const int lane           = threadIdx.x % MOE_WARP_SIZE;
    if(warp_id_global >= num_tokens)
        return;

    constexpr int experts_per_thread = (N_EXPERTS > MOE_WARP_SIZE) ? (N_EXPERTS / MOE_WARP_SIZE) : 1;
    float wt[experts_per_thread];
    for(int i = 0; i < experts_per_thread; i++)
    {
        const int expert = lane + i * MOE_WARP_SIZE;
        wt[i] = (expert < N_EXPERTS) ? router_logits[warp_id_global * N_EXPERTS + expert]
                                     : -INFINITY;
    }

    float selected_weights[TOP_K];
    int selected_experts[TOP_K];
    for(int k = 0; k < TOP_K; k++)
    {
        float max_val  = wt[0];
        int max_expert = lane;
        for(int i = 1; i < experts_per_thread; i++)
        {
            const int expert = lane + i * MOE_WARP_SIZE;
            if(wt[i] > max_val)
            {
                max_val    = wt[i];
                max_expert = expert;
            }
        }
        for(int mask = MOE_WARP_SIZE / 2; mask > 0; mask >>= 1)
        {
            const float other_val  = __shfl_xor(max_val, mask);
            const int other_expert = __shfl_xor(max_expert, mask);
            if(other_val > max_val or (other_val == max_val and other_expert < max_expert))
            {
                max_val    = other_val;
                max_expert = other_expert;
            }
        }
        selected_weights[k] = max_val;
        selected_experts[k] = max_expert;
        if((max_expert & (MOE_WARP_SIZE - 1)) == lane)
            wt[max_expert / MOE_WARP_SIZE] = -INFINITY;
    }

    {
        float max_w = selected_weights[0];
        for(int k = 1; k < TOP_K; k++)
            max_w = fmaxf(max_w, selected_weights[k]);
        float sum = 0.0f;
        for(int k = 0; k < TOP_K; k++)
        {
            selected_weights[k] = expf(selected_weights[k] - max_w);
            sum += selected_weights[k];
        }
        const float inv_sum = 1.0f / sum;
        for(int k = 0; k < TOP_K; k++)
            selected_weights[k] *= inv_sum;
    }

    if(lane == 0)
    {
        const int token = warp_id_global;
        for(int k = 0; k < TOP_K; k++)
        {
            topk_weights[token * TOP_K + k]    = selected_weights[k];
            topk_expert_ids[token * TOP_K + k] = selected_experts[k];
            const int expert                   = selected_experts[k];
            int slot                           = atomicAdd(&expert_token_counts[expert], 1);
            if(slot < max_tokens_per_expert)
                expert_token_ids[expert * max_tokens_per_expert + slot] = token;
        }
    }
}

// ============================================================================
// E1(b) S1 — PAIRED device-side dispatch kernels (capture-enabling).
// One grid over the compacted (token,expert) pair axis: blockIdx.z = pair,
// pair in [0, S*top_k). token = pair/top_k, expert = topk_expert_ids[pair],
// router_weight = topk_weights[pair] — all read on-device. NO host sync, NO
// per-expert host loop. Inner GEMV math is identical to the kernels above
// (rel_err 0 vs HF); only index derivation changes. swiglu uses a per-PAIR
// slab row [S*top_k, inter] (race-free: each pair owns its row). FC2 uses
// atomicAdd (a token's top_k pairs write the same output row concurrently).
// ============================================================================

// FC1 + SwiGLU, paired. swiglu_all is [num_pairs, intermediate_size].
__global__ void __launch_bounds__(MM_WARPS_PER_BLOCK* MOE_WARP_SIZE, 2)
    moe_q4_swiglu_paired_kernel(const float* __restrict__ A,        // [tokens, K] hidden
                                const int32_t* __restrict__ topk_expert_ids, // [tokens*top_k]
                                const uint32_t* __restrict__ fc1_w, // [E, N_fc1, K/8]
                                const float* __restrict__ fc1_s,    // [E, N_fc1, K/32]
                                const float* __restrict__ fc1_b,    // [E, N_fc1] or null
                                float* __restrict__ swiglu_all,     // [num_pairs, inter]
                                const int top_k,
                                const int num_experts,
                                const int K,
                                const int intermediate_size,
                                const size_t fc1_w_stride,
                                const size_t fc1_s_stride,
                                const float alpha,
                                const float beta,
                                const float limit)
{
    const int pair    = blockIdx.z;
    const int token   = pair / top_k;
    const int expert  = topk_expert_ids[pair];
    const int warp_id = threadIdx.x / MOE_WARP_SIZE;
    const int lane    = threadIdx.x % MOE_WARP_SIZE;

    // Guard a stray expert id (bad routing) without a host check.
    const bool valid = (expert >= 0 and expert < num_experts);

    const uint32_t* B   = fc1_w + (size_t)(valid ? expert : 0) * fc1_w_stride;
    const float* scales = fc1_s + (size_t)(valid ? expert : 0) * fc1_s_stride;
    const float* bias   = fc1_b ? fc1_b + (size_t)(valid ? expert : 0) * (intermediate_size * 2)
                                : nullptr;
    float* C            = swiglu_all + (size_t)pair * intermediate_size;

    extern __shared__ float smem_A[];
    for(int i = threadIdx.x; i < K; i += blockDim.x)
        smem_A[i + (i >> 5)] = A[(size_t)token * K + i];
    __syncthreads(); // uniform — guard/return is AFTER (deadlock-safe)

    const int col = blockIdx.x * MM_WARPS_PER_BLOCK + warp_id;
    if(col >= intermediate_size or not valid)
        return;

    const int gate_row       = col * 2;
    const int up_row         = col * 2 + 1;
    const int K_over_8       = K >> 3;
    const int blocks_per_col = K >> 5;

    float sum_gate = 0.0f;
    for(int qb = lane; qb < blocks_per_col; qb += MOE_WARP_SIZE)
    {
        const float scale = scales[gate_row * blocks_per_col + qb];
        const int b_off   = gate_row * K_over_8 + qb * 4;
        const int pa_base = qb * 33;
#pragma unroll
        for(int j = 0; j < 4; j++)
        {
            const uint32_t packed = B[b_off + j];
            const int pa          = pa_base + j * 8;
            sum_gate += smem_A[pa + 0] * ((float)((packed >> 0) & 0xF) - 8.0f) * scale;
            sum_gate += smem_A[pa + 1] * ((float)((packed >> 4) & 0xF) - 8.0f) * scale;
            sum_gate += smem_A[pa + 2] * ((float)((packed >> 8) & 0xF) - 8.0f) * scale;
            sum_gate += smem_A[pa + 3] * ((float)((packed >> 12) & 0xF) - 8.0f) * scale;
            sum_gate += smem_A[pa + 4] * ((float)((packed >> 16) & 0xF) - 8.0f) * scale;
            sum_gate += smem_A[pa + 5] * ((float)((packed >> 20) & 0xF) - 8.0f) * scale;
            sum_gate += smem_A[pa + 6] * ((float)((packed >> 24) & 0xF) - 8.0f) * scale;
            sum_gate += smem_A[pa + 7] * ((float)((packed >> 28) & 0xF) - 8.0f) * scale;
        }
    }
    for(int offset = MOE_WARP_SIZE / 2; offset > 0; offset >>= 1)
        sum_gate += __shfl_xor(sum_gate, offset);

    float sum_up = 0.0f;
    for(int qb = lane; qb < blocks_per_col; qb += MOE_WARP_SIZE)
    {
        const float scale = scales[up_row * blocks_per_col + qb];
        const int b_off   = up_row * K_over_8 + qb * 4;
        const int pa_base = qb * 33;
#pragma unroll
        for(int j = 0; j < 4; j++)
        {
            const uint32_t packed = B[b_off + j];
            const int pa          = pa_base + j * 8;
            sum_up += smem_A[pa + 0] * ((float)((packed >> 0) & 0xF) - 8.0f) * scale;
            sum_up += smem_A[pa + 1] * ((float)((packed >> 4) & 0xF) - 8.0f) * scale;
            sum_up += smem_A[pa + 2] * ((float)((packed >> 8) & 0xF) - 8.0f) * scale;
            sum_up += smem_A[pa + 3] * ((float)((packed >> 12) & 0xF) - 8.0f) * scale;
            sum_up += smem_A[pa + 4] * ((float)((packed >> 16) & 0xF) - 8.0f) * scale;
            sum_up += smem_A[pa + 5] * ((float)((packed >> 20) & 0xF) - 8.0f) * scale;
            sum_up += smem_A[pa + 6] * ((float)((packed >> 24) & 0xF) - 8.0f) * scale;
            sum_up += smem_A[pa + 7] * ((float)((packed >> 28) & 0xF) - 8.0f) * scale;
        }
    }
    for(int offset = MOE_WARP_SIZE / 2; offset > 0; offset >>= 1)
        sum_up += __shfl_xor(sum_up, offset);

    if(lane == 0)
    {
        float gate = sum_gate;
        float up   = sum_up;
        if(bias)
        {
            gate += bias[gate_row];
            up += bias[up_row];
        }
        gate              = fminf(gate, limit);
        up                = fminf(fmaxf(up, -limit), limit);
        float sigmoid_val = 1.0f / (1.0f + expf(-alpha * gate));
        C[col]            = (up + beta) * gate * sigmoid_val;
    }
}

// FC2 + weighted accumulate, paired. atomicAdd into output[token*N + n].
__global__ void __launch_bounds__(MM_WARPS_PER_BLOCK* MOE_WARP_SIZE, 2)
    moe_q4_accum_paired_kernel(const float* __restrict__ swiglu_all, // [num_pairs, K=inter]
                               const int32_t* __restrict__ topk_expert_ids, // [tokens*top_k]
                               const float* __restrict__ topk_weights,      // [tokens*top_k]
                               const uint32_t* __restrict__ fc2_w, // [E, N, K/8]
                               const float* __restrict__ fc2_s,    // [E, N, K/32]
                               const float* __restrict__ fc2_b,    // [E, N] or null
                               float* __restrict__ output,         // [tokens, N]
                               const int top_k,
                               const int num_experts,
                               const int N,
                               const int K,
                               const size_t fc2_w_stride,
                               const size_t fc2_s_stride)
{
    const int pair    = blockIdx.z;
    const int token   = pair / top_k;
    const int expert  = topk_expert_ids[pair];
    const int warp_id = threadIdx.x / MOE_WARP_SIZE;
    const int lane    = threadIdx.x % MOE_WARP_SIZE;
    const bool valid  = (expert >= 0 and expert < num_experts);

    const float* A      = swiglu_all + (size_t)pair * K;
    const uint32_t* B   = fc2_w + (size_t)(valid ? expert : 0) * fc2_w_stride;
    const float* scales = fc2_s + (size_t)(valid ? expert : 0) * fc2_s_stride;
    const float* bias   = fc2_b ? fc2_b + (size_t)(valid ? expert : 0) * N : nullptr;

    extern __shared__ float smem_A[];
    for(int i = threadIdx.x; i < K; i += blockDim.x)
        smem_A[i + (i >> 5)] = A[i];
    __syncthreads(); // uniform — guard/return is AFTER (deadlock-safe)

    const int n = blockIdx.x * MM_WARPS_PER_BLOCK + warp_id;
    if(n >= N or not valid)
        return;

    const int K_over_8       = K >> 3;
    const int blocks_per_col = K >> 5;
    float sum                = 0.0f;
    for(int qb = lane; qb < blocks_per_col; qb += MOE_WARP_SIZE)
    {
        const float scale = scales[n * blocks_per_col + qb];
        const int b_off   = n * K_over_8 + qb * 4;
        const int pa_base = qb * 33;
#pragma unroll
        for(int j = 0; j < 4; j++)
        {
            const uint32_t packed = B[b_off + j];
            const int pa          = pa_base + j * 8;
            sum += smem_A[pa + 0] * ((float)((packed >> 0) & 0xF) - 8.0f) * scale;
            sum += smem_A[pa + 1] * ((float)((packed >> 4) & 0xF) - 8.0f) * scale;
            sum += smem_A[pa + 2] * ((float)((packed >> 8) & 0xF) - 8.0f) * scale;
            sum += smem_A[pa + 3] * ((float)((packed >> 12) & 0xF) - 8.0f) * scale;
            sum += smem_A[pa + 4] * ((float)((packed >> 16) & 0xF) - 8.0f) * scale;
            sum += smem_A[pa + 5] * ((float)((packed >> 20) & 0xF) - 8.0f) * scale;
            sum += smem_A[pa + 6] * ((float)((packed >> 24) & 0xF) - 8.0f) * scale;
            sum += smem_A[pa + 7] * ((float)((packed >> 28) & 0xF) - 8.0f) * scale;
        }
    }
    for(int offset = MOE_WARP_SIZE / 2; offset > 0; offset >>= 1)
        sum += __shfl_xor(sum, offset);

    if(lane == 0)
    {
        if(bias)
            sum += bias[n];
        const float router_weight = topk_weights[pair];
        atomicAdd(&output[(size_t)token * N + n], router_weight * sum);
    }
}

} // namespace

void gptoss_moe(hipStream_t stream,
                const argument& output,
                const argument& hidden_states,
                const argument& router_logits,
                const argument& fc1_weights,
                const argument& fc1_scales,
                const argument& fc2_weights,
                const argument& fc2_scales,
                const argument& fc1_bias,
                const argument& fc2_bias,
                const argument& topk_weights,
                const argument& topk_expert_ids,
                const argument& expert_token_ids,
                const argument& expert_token_counts,
                const moe_params& p)
{
    auto* d_out      = reinterpret_cast<float*>(output.data());
    auto* d_hidden   = reinterpret_cast<const float*>(hidden_states.data());
    auto* d_router   = reinterpret_cast<const float*>(router_logits.data());
    auto* d_fc1_w    = reinterpret_cast<const uint32_t*>(fc1_weights.data());
    auto* d_fc1_s    = reinterpret_cast<const float*>(fc1_scales.data());
    auto* d_fc2_w    = reinterpret_cast<const uint32_t*>(fc2_weights.data());
    auto* d_fc2_s    = reinterpret_cast<const float*>(fc2_scales.data());
    auto* d_fc1_b    = reinterpret_cast<const float*>(fc1_bias.data());
    auto* d_fc2_b    = reinterpret_cast<const float*>(fc2_bias.data());
    auto* d_topk_w   = reinterpret_cast<float*>(topk_weights.data());
    auto* d_topk_e   = reinterpret_cast<int32_t*>(topk_expert_ids.data());
    auto* d_etok_ids = reinterpret_cast<int32_t*>(expert_token_ids.data());
    auto* d_etok_cnt = reinterpret_cast<int32_t*>(expert_token_counts.data());

    const int S        = p.num_tokens;
    const int hidden   = p.hidden_size;
    const int inter    = p.intermediate_size;
    const int E        = p.num_experts;
    const int top_k    = p.top_k;
    const int maxtok   = p.max_tokens_per_expert;
    const int N_fc1    = inter * 2;
    const int N_fc2    = hidden;
    const size_t fc1_w_stride = (size_t)N_fc1 * (hidden / 8);
    const size_t fc1_s_stride = (size_t)N_fc1 * (hidden / 32);
    const size_t fc2_w_stride = (size_t)N_fc2 * (inter / 8);
    const size_t fc2_s_stride = (size_t)N_fc2 * (inter / 32);
    const size_t fc1_b_stride = (size_t)N_fc1;  // [E, N_fc1]
    const size_t fc2_b_stride = (size_t)N_fc2;  // [E, N_fc2]

    if(d_out == nullptr or d_hidden == nullptr or d_router == nullptr or d_fc1_w == nullptr or
       d_fc2_w == nullptr)
    {
        std::fprintf(stderr,
                     "[gptoss_moe] null buffer: out=%p hidden=%p router=%p fc1_w=%p fc2_w=%p\n",
                     (void*)d_out, (void*)d_hidden, (void*)d_router, (void*)d_fc1_w,
                     (void*)d_fc2_w);
        std::fflush(stderr);
        return;
    }

    // zero output + expert counts
    MOE_HIP_CHECK(hipMemsetAsync(d_out, 0, (size_t)S * hidden * sizeof(float), stream));
    MOE_HIP_CHECK(hipMemsetAsync(d_etok_cnt, 0, (size_t)E * sizeof(int32_t), stream));

    // 1) routing (32 experts / top-4 specialization, matching the model)
    {
        const int threads = TOPK_WARPS_PER_BLOCK * MOE_WARP_SIZE;
        const int blocks  = (S + TOPK_WARPS_PER_BLOCK - 1) / TOPK_WARPS_PER_BLOCK;
        if(E == 32 and top_k == 4)
            moe_topk_kernel<32, 4><<<blocks, threads, 0, stream>>>(
                d_router, d_topk_w, d_topk_e, d_etok_ids, d_etok_cnt, S, maxtok);
        else
        {
            std::fprintf(stderr, "[gptoss_moe] unsupported E=%d top_k=%d\n", E, top_k);
            std::fflush(stderr);
            return;
        }
        MOE_HIP_CHECK(hipGetLastError());
    }

    // 2) E1(b) S1 — DEVICE-SIDE COMPACTED dispatch. NO host readback, NO sync, NO
    //    per-expert host loop. The routing kernel already wrote, for each pair
    //    p in [0, S*top_k): topk_expert_ids[p] and topk_weights[p]. That IS the
    //    compacted (token,expert,weight) work-list, of host-known size S*top_k.
    //    We launch exactly 2 kernels gridded over the pair axis (blockIdx.z=pair);
    //    each block derives token=pair/top_k, expert=topk_expert_ids[pair] on
    //    device. This makes compute() free of host control-flow / sync, so the
    //    decode program is hipGraph-capturable (E1(b) prerequisite). It is also
    //    compacted (S*top_k pairs, e.g. 4 at decode) — NOT the dense maxtok*E grid
    //    that regressed in B3. expert_token_ids/_counts are now unused by this path
    //    (routing still computes them harmlessly).
    const int num_pairs = S * top_k;

    // Persistent per-PAIR swiglu slab [num_pairs, inter] (B1-style grow-only;
    // race-free: each pair owns its row). Decode: 4*2880*4 = 46 KB.
    static thread_local float* d_swiglu     = nullptr;
    static thread_local size_t d_swiglu_cap = 0;
    const size_t swiglu_bytes = (size_t)num_pairs * inter * sizeof(float);
    if(d_swiglu_cap < swiglu_bytes)
    {
        if(d_swiglu != nullptr)
            MOE_HIP_CHECK(hipFree(d_swiglu));
        MOE_HIP_CHECK(hipMalloc(&d_swiglu, swiglu_bytes));
        d_swiglu_cap = swiglu_bytes;
    }
    if(d_swiglu == nullptr)
        return;

    const int mm_threads = MM_WARPS_PER_BLOCK * MOE_WARP_SIZE;
    const size_t smem    = (size_t)(hidden + (hidden >> 5)) * sizeof(float);

    // FC1 + SwiGLU over all pairs (grid.z = num_pairs).
    {
        dim3 g1((inter + MM_WARPS_PER_BLOCK - 1) / MM_WARPS_PER_BLOCK, 1,
                static_cast<unsigned>(num_pairs));
        moe_q4_swiglu_paired_kernel<<<g1, mm_threads, smem, stream>>>(
            d_hidden, d_topk_e, d_fc1_w, d_fc1_s, d_fc1_b, d_swiglu, top_k, E, hidden, inter,
            fc1_w_stride, fc1_s_stride, p.swiglu_alpha, p.swiglu_beta, p.swiglu_limit);
        MOE_HIP_CHECK(hipGetLastError());

        // FC2 + weighted accumulate over all pairs (atomicAdd; d_out pre-zeroed).
        dim3 g2((N_fc2 + MM_WARPS_PER_BLOCK - 1) / MM_WARPS_PER_BLOCK, 1,
                static_cast<unsigned>(num_pairs));
        moe_q4_accum_paired_kernel<<<g2, mm_threads, smem, stream>>>(
            d_swiglu, d_topk_e, d_topk_w, d_fc2_w, d_fc2_s, d_fc2_b, d_out, top_k, E, N_fc2, inter,
            fc2_w_stride, fc2_s_stride);
        MOE_HIP_CHECK(hipGetLastError());
    }

    // E1(b) S1: compute() now issues ZERO hipStreamSynchronize and NO host
    // control flow — the op is hipGraph-capturable. Correctness rests on
    // single-stream in-order execution (routing -> FC1 -> FC2; downstream reads
    // d_out on the same stream; cross-call d_swiglu reuse is ordered because
    // call N+1's FC1 cannot start until call N's FC2 finishes on the stream).
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
