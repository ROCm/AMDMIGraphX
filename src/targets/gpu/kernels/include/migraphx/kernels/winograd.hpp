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
#ifndef MIGRAPHX_GUARD_KERNELS_WINOGRAD_HPP
#define MIGRAPHX_GUARD_KERNELS_WINOGRAD_HPP

#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/types.hpp>

namespace migraphx {
namespace winograd {

// =============================================================================
// Winograd F(4x4, 3x3) transforms using __builtin_fmaf for optimal scheduling
// =============================================================================

__device__ inline float fmaf_(float a, float b, float c) { return __builtin_fmaf(a, b, c); }

// B^T column (6 -> 6)
__device__ inline void bt_col(float s0,
                              float s1,
                              float s2,
                              float s3,
                              float s4,
                              float s5,
                              float& d0,
                              float& d1,
                              float& d2,
                              float& d3,
                              float& d4,
                              float& d5)
{
    float p = s3 + s4;
    float q = s4 - s3;
    d0      = fmaf_(4.0f, s0, s4) + (-5.0f) * s2;
    d5      = fmaf_(4.0f, s1, s5) + (-5.0f) * s3;
    d1      = fmaf_(-4.0f, s2, fmaf_(-4.0f, s1, p));
    d2      = fmaf_(-4.0f, s2, fmaf_(4.0f, s1, q));
    d3      = fmaf_(2.0f, s3, fmaf_(-2.0f, s1, s4) - s2);
    d4      = fmaf_(-2.0f, s3, fmaf_(2.0f, s1, s4) - s2);
}

// G column (3 -> 6)
__device__ inline void g_col(
    float g0, float g1, float g2, float& u0, float& u1, float& u2, float& u3, float& u4, float& u5)
{
    constexpr float c6  = 1.0f / 6.0f;
    constexpr float c12 = 1.0f / 12.0f;
    constexpr float c24 = 1.0f / 24.0f;
    u0                  = 0.25f * g0;
    u5                  = g2;
    float t             = -c6 * (g0 + g2);
    u1                  = fmaf_(-c6, g1, t);
    u2                  = fmaf_(c6, g1, t);
    float base          = fmaf_(c6, g2, c24 * g0);
    u3                  = fmaf_(c12, g1, base);
    u4                  = fmaf_(-c12, g1, base);
}

// A^T column (6 -> 4)
__device__ inline void at_col(float s0,
                              float s1,
                              float s2,
                              float s3,
                              float s4,
                              float s5,
                              float& y0,
                              float& y1,
                              float& y2,
                              float& y3)
{
    float t0 = s1 + s2;
    float t1 = s1 - s2;
    float t2 = s3 + s4;
    float t3 = s3 - s4;
    y0       = s0 + t0 + t2;
    y1       = fmaf_(2.0f, t3, t1);
    y2       = fmaf_(4.0f, t2, t0);
    y3       = fmaf_(8.0f, t3, t1) + s5;
}

// =============================================================================
// 2D transforms on flat [36] arrays (row-major 6x6)
// =============================================================================

__device__ inline void input_xform(const float* __restrict__ d, float* __restrict__ V)
{
    float tmp[36];
    for(index_int j = 0; j < 6; j++)
        bt_col(d[j],
               d[6 + j],
               d[12 + j],
               d[18 + j],
               d[24 + j],
               d[30 + j],
               tmp[j],
               tmp[6 + j],
               tmp[12 + j],
               tmp[18 + j],
               tmp[24 + j],
               tmp[30 + j]);
    for(index_int i = 0; i < 6; i++)
    {
        index_int r = i * 6;
        bt_col(tmp[r],
               tmp[r + 1],
               tmp[r + 2],
               tmp[r + 3],
               tmp[r + 4],
               tmp[r + 5],
               V[r],
               V[r + 1],
               V[r + 2],
               V[r + 3],
               V[r + 4],
               V[r + 5]);
    }
}

__device__ inline void filter_xform(const float* __restrict__ g, float* __restrict__ U)
{
    float tmp[18];
    for(index_int j = 0; j < 3; j++)
        g_col(g[j],
              g[3 + j],
              g[6 + j],
              tmp[j],
              tmp[3 + j],
              tmp[6 + j],
              tmp[9 + j],
              tmp[12 + j],
              tmp[15 + j]);
    for(index_int i = 0; i < 6; i++)
    {
        index_int o = i * 6;
        index_int r = i * 3;
        g_col(
            tmp[r], tmp[r + 1], tmp[r + 2], U[o], U[o + 1], U[o + 2], U[o + 3], U[o + 4], U[o + 5]);
    }
}

__device__ inline void output_xform(const float* __restrict__ M, float* __restrict__ Y)
{
    float tmp[24];
    for(index_int j = 0; j < 6; j++)
        at_col(M[j],
               M[6 + j],
               M[12 + j],
               M[18 + j],
               M[24 + j],
               M[30 + j],
               tmp[j],
               tmp[6 + j],
               tmp[12 + j],
               tmp[18 + j]);
    for(index_int i = 0; i < 4; i++)
    {
        index_int o = i * 4;
        index_int r = i * 6;
        at_col(tmp[r],
               tmp[r + 1],
               tmp[r + 2],
               tmp[r + 3],
               tmp[r + 4],
               tmp[r + 5],
               Y[o],
               Y[o + 1],
               Y[o + 2],
               Y[o + 3]);
    }
}

// =============================================================================
// Optimized Winograd F(4x4, 3x3) convolution
//
// Key optimizations:
// 1. Shared memory filter caching (precompute G*g*G^T once per workgroup)
// 2. K_BATCH: each thread processes multiple output filters, loading input
//    tiles once and reusing across filters. This is the critical optimization
//    that reduces global memory traffic by K_BATCH×.
// 3. Workgroup decomposition ordered so adjacent workgroups share the same
//    spatial tiles for L2 cache reuse across filter groups.
// 4. Uses __builtin_fmaf for compiler-friendly FMA (no asm volatile).
// =============================================================================

template <index_int Group,
          index_int N,
          index_int C,
          index_int H,
          index_int W,
          index_int K,
          index_int CHUNK_C,
          index_int K_BATCH>
__device__ void conv(const float* __restrict__ input,
                     const float* __restrict__ weight,
                     float* __restrict__ output,
                     float* __restrict__ s_filt)
{
    constexpr index_int OH        = H;
    constexpr index_int OW        = W;
    constexpr index_int TILE_H    = 4;
    constexpr index_int TILE_W    = 4;
    constexpr index_int TILES_H   = (OH + TILE_H - 1) / TILE_H;
    constexpr index_int TILES_W   = (OW + TILE_W - 1) / TILE_W;
    constexpr index_int NUM_TILES = TILES_H * TILES_W;
    constexpr index_int C_PER_GRP = C / Group;
    constexpr index_int K_PER_GRP = K / Group;
    constexpr index_int BLOCK     = MIGRAPHX_NLOCAL;
    constexpr index_int TILE_GRPS = (NUM_TILES + BLOCK - 1) / BLOCK;
    constexpr index_int K_GRPS    = (K + K_BATCH - 1) / K_BATCH;

    index_int tid = threadIdx.x; // NOLINT
    index_int wg  = blockIdx.x;  // NOLINT

    // Decompose: iterate k-groups fastest for L2 reuse of input tiles
    index_int tile_grp = wg / K_GRPS;
    index_int rem      = wg % K_GRPS;
    index_int n_val    = tile_grp / TILE_GRPS;
    index_int tg       = tile_grp % TILE_GRPS;
    index_int k_grp    = rem;
    index_int k_base   = k_grp * K_BATCH;
    index_int my_tile  = tg * BLOCK + tid;

    // How many k values this workgroup actually processes (handle tail)
    index_int k_actual = K_BATCH;
    if(k_base + K_BATCH > K)
        k_actual = K - k_base;

    // Group convolution: which input channel group
    index_int group_id = k_base / K_PER_GRP;
    index_int c_base   = group_id * C_PER_GRP;

    // Precompute tile coords
    index_int tile_row = my_tile / TILES_W;
    index_int tile_col = my_tile % TILES_W;
    diff_int ih_start  = static_cast<diff_int>(tile_row * TILE_H) - 1;
    diff_int iw_start  = static_cast<diff_int>(tile_col * TILE_W) - 1;

    // Per-thread accumulators: M[kb][36]
    float M[K_BATCH * 36];
    for(index_int p = 0; p < K_BATCH * 36; p++)
        M[p] = 0.0f;

    for(index_int c_chunk = 0; c_chunk < C_PER_GRP; c_chunk += CHUNK_C)
    {
        index_int chunk_sz = C_PER_GRP - c_chunk;
        if(chunk_sz > CHUNK_C)
            chunk_sz = CHUNK_C;

        // Phase 1: Cooperatively transform K_BATCH * chunk_sz filters
        index_int total_filt = k_actual * chunk_sz;
        for(index_int idx = tid; idx < total_filt; idx += BLOCK)
        {
            index_int kb = idx / chunk_sz;
            index_int cl = idx % chunk_sz;
            index_int kk = k_base + kb;
            float g[9];
            index_int w_off = (kk * C_PER_GRP + c_chunk + cl) * 9;
            for(index_int p = 0; p < 9; p++)
                g[p] = weight[w_off + p];
            filter_xform(g, s_filt + (kb * chunk_sz + cl) * 36);
        }
        __syncthreads();

        // Phase 2: Each thread accumulates its tile
        if(my_tile < NUM_TILES)
        {
            for(index_int cc = 0; cc < chunk_sz; cc++)
            {
                index_int ic = c_base + c_chunk + cc;

                // Load 6x6 input tile with zero-padding
                float d[36];
                for(index_int i = 0; i < 6; i++)
                {
                    diff_int ih     = ih_start + static_cast<diff_int>(i);
                    bool ih_ok      = ih >= 0 and ih < static_cast<diff_int>(H);
                    index_int rbase = ((n_val * C + ic) * H + static_cast<index_int>(ih)) * W;
                    for(index_int j = 0; j < 6; j++)
                    {
                        diff_int iw  = iw_start + static_cast<diff_int>(j);
                        d[i * 6 + j] = (ih_ok and iw >= 0 and iw < static_cast<diff_int>(W))
                                           ? input[rbase + static_cast<index_int>(iw)]
                                           : 0.0f;
                    }
                }

                // Input transform (done once, reused K_BATCH times)
                float V[36];
                input_xform(d, V);

                // Accumulate across K_BATCH filters from shared memory
                for(index_int kb = 0; kb < k_actual; kb++)
                {
                    const float* U_ptr = s_filt + (kb * chunk_sz + cc) * 36;
                    float* M_ptr       = M + kb * 36;
                    for(index_int p = 0; p < 36; p++)
                        M_ptr[p] = fmaf_(U_ptr[p], V[p], M_ptr[p]);
                }
            }
        }
        __syncthreads();
    }

    // Phase 3: Output transform and store for each filter
    if(my_tile < NUM_TILES)
    {
        index_int oh_base = tile_row * TILE_H;
        index_int ow_base = tile_col * TILE_W;

        for(index_int kb = 0; kb < k_actual; kb++)
        {
            float Y[16];
            output_xform(M + kb * 36, Y);

            index_int kk = k_base + kb;
            for(index_int i = 0; i < TILE_H; i++)
            {
                index_int oh = oh_base + i;
                if(oh >= OH)
                    break;
                index_int out_row = ((n_val * K + kk) * OH + oh) * OW;
                for(index_int j = 0; j < TILE_W; j++)
                {
                    index_int ow = ow_base + j;
                    if(ow >= OW)
                        break;
                    output[out_row + ow] = Y[i * 4 + j];
                }
            }
        }
    }
}

} // namespace winograd
} // namespace migraphx

#endif // MIGRAPHX_GUARD_KERNELS_WINOGRAD_HPP
