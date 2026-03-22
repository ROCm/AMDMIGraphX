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
#include <migraphx/kernels/dpp.hpp>

namespace migraphx {
namespace winograd {

__device__ inline float fmaf_(float a, float b, float c) { return __builtin_fmaf(a, b, c); }

// =============================================================================
// F(2x2,3x3) B^T column transform applied via DPP within a quad.
//
// Each quad (4 threads) holds one row of a 4x4 tile. Thread lane (0-3)
// holds row[lane]. The B^T column uses quad_perm:[2,2,1,1] (DPP ctrl 0x5A)
// so each thread receives the partner row it needs:
//   lane 0: gets row[2] → computes row[0]-row[2]
//   lane 1: gets row[2] → computes row[1]+row[2]
//   lane 2: gets row[1] → computes row[2]-row[1]
//   lane 3: gets row[1] → computes row[1]-row[3]
// =============================================================================

__device__ inline void bt_col_dpp(float* v, index_int lane)
{
    // DPP quad_perm:[2,2,1,1] = 0x5A
    for(index_int j = 0; j < 4; j++)
    {
        float my      = v[j];
        float partner = dpp_mov<0x5A>(my);
        float diff    = my - partner;
        // lane 1: sum; lane 3: negated diff; lanes 0,2: diff
        v[j] = (lane == 1) ? (my + partner) : ((lane == 3) ? -diff : diff);
    }
}

// Intra-thread B^T row transform (same operation, applied to 4 values)
__device__ inline void bt_row(float* v)
{
    float a = v[0], b = v[1], c = v[2], d = v[3];
    v[0] = a - c;
    v[1] = b + c;
    v[2] = c - b;
    v[3] = b - d;
}

// =============================================================================
// Standard (non-DPP) transforms for filter and output
// =============================================================================

// G * g * G^T filter transform 3x3 → 4x4 flat[16]
__device__ inline void filter_xform(const float* __restrict__ g, float* __restrict__ U)
{
    float t[12];
    for(index_int j = 0; j < 3; j++)
    {
        float g0 = g[j], g1 = g[3 + j], g2 = g[6 + j];
        float s = (g0 + g2) * 0.5f, d = g1 * 0.5f;
        t[j]     = g0;
        t[3 + j] = s + d;
        t[6 + j] = s - d;
        t[9 + j] = g2;
    }
    for(index_int i = 0; i < 4; i++)
    {
        index_int r = i * 3, o = i * 4;
        float t0 = t[r], t1 = t[r + 1], t2 = t[r + 2];
        float s = (t0 + t2) * 0.5f, d = t1 * 0.5f;
        U[o]     = t0;
        U[o + 1] = s + d;
        U[o + 2] = s - d;
        U[o + 3] = t2;
    }
}

// A^T * m * A output transform 4x4 → 2x2 flat[4]
__device__ inline void output_xform(const float* __restrict__ M, float* __restrict__ Y)
{
    float t[8];
    for(index_int j = 0; j < 4; j++)
    {
        float m0 = M[j], m1 = M[4 + j], m2 = M[8 + j], m3 = M[12 + j];
        t[j]     = m0 + m1 + m2;
        t[4 + j] = m1 - m2 - m3;
    }
    for(index_int i = 0; i < 2; i++)
    {
        index_int r  = i * 4;
        Y[i * 2]     = t[r] + t[r + 1] + t[r + 2];
        Y[i * 2 + 1] = t[r + 1] - t[r + 2] - t[r + 3];
    }
}

// Filter precompute (for graph-level constant folding or separate kernel)
template <index_int K, index_int C_PER_GRP>
__device__ void filter_precompute(const float* __restrict__ weight, float* __restrict__ workspace)
{
    auto idx = make_index();
    idx.global_stride(K * C_PER_GRP, [&](auto id) {
        float g[9], U[16];
        for(index_int p = 0; p < 9; p++)
            g[p] = weight[id * 9 + p];
        filter_xform(g, U);
        for(index_int p = 0; p < 16; p++)
            workspace[id * 16 + p] = U[p];
    });
}

// =============================================================================
// GEMM-based Winograd F(2x2,3x3) convolution with DPP tile loading
//
// Phase 1a: DPP-based input transform
//   - 4 threads (quad) per tile: each loads 1 row (4 values), transforms
//     via DPP quad_perm, writes 4 values to LDS. 4× less global memory.
// Phase 1b: Filter transform → LDS (pretransformed weights skip this)
// Phase 2:  Tiled GEMM from LDS with sched_barrier pipelining
// Phase 3:  Output transform A^T*m*A and store
//
// Template param PRETRANSFORMED: if true, weight points to [K][C/G][16]
// pretransformed filters (skips Phase 1b entirely).
// =============================================================================

template <index_int Group,
          index_int N,
          index_int C,
          index_int H,
          index_int W,
          index_int K,
          index_int TILES_PER_WG,
          index_int K_PER_WG,
          index_int CHUNK_C,
          bool PRETRANSFORMED,
          index_int T_TILE = 2,
          index_int K_TILE = 2>
__device__ void conv(const float* __restrict__ input,
                     const float* __restrict__ weight,
                     float* __restrict__ output,
                     float* __restrict__ lds)
{
    constexpr index_int OUT_TILE    = 2;
    constexpr index_int ALPHA       = 4;
    constexpr index_int ALPHA2      = 16;
    constexpr index_int TILES_H     = (H + OUT_TILE - 1) / OUT_TILE;
    constexpr index_int TILES_W     = (W + OUT_TILE - 1) / OUT_TILE;
    constexpr index_int TOTAL_TILES = TILES_H * TILES_W;
    constexpr index_int C_PER_GRP   = C / Group;
    constexpr index_int K_PER_GRP   = K / Group;
    constexpr index_int BLOCK       = MIGRAPHX_NLOCAL;
    constexpr index_int THREADS_N   = K_PER_WG / K_TILE;
    constexpr index_int TILE_GRPS   = (TOTAL_TILES + TILES_PER_WG - 1) / TILES_PER_WG;
    constexpr index_int K_GRPS      = (K + K_PER_WG - 1) / K_PER_WG;
    constexpr index_int V_PLANE     = TILES_PER_WG * CHUNK_C;
    constexpr index_int U_PLANE     = CHUNK_C * K_PER_WG;
    float* lds_v                    = lds;
    float* lds_u                    = lds + ALPHA2 * V_PLANE;

    index_int tid      = threadIdx.x; // NOLINT
    index_int wg       = blockIdx.x;  // NOLINT
    index_int ntg      = wg / K_GRPS;
    index_int k_grp    = wg % K_GRPS;
    index_int n_val    = ntg / TILE_GRPS;
    index_int tg       = ntg % TILE_GRPS;
    index_int t_base   = tg * TILES_PER_WG;
    index_int k_base   = k_grp * K_PER_WG;
    index_int k_actual = (k_base + K_PER_WG > K) ? (K - k_base) : K_PER_WG;
    index_int group_id = k_base / K_PER_GRP;
    index_int c_base   = group_id * C_PER_GRP;

    index_int thread_m = tid / THREADS_N;
    index_int thread_n = tid % THREADS_N;
    index_int my_t0    = thread_m * T_TILE;
    index_int my_k0    = thread_n * K_TILE;

    float acc[T_TILE * K_TILE * ALPHA2];
    for(index_int i = 0; i < T_TILE * K_TILE * ALPHA2; i++)
        acc[i] = 0.0f;

    for(index_int c_chunk = 0; c_chunk < C_PER_GRP; c_chunk += CHUNK_C)
    {
        index_int csz = C_PER_GRP - c_chunk;
        if(csz > CHUNK_C)
            csz = CHUNK_C;

        // === Phase 1a: Input tile transform → LDS ===
        // Uses float4 vectorized loads for interior tiles (no padding needed),
        // cutting load instructions 4× vs scalar. Border tiles use scalar
        // loads with boundary checks.
        {
            index_int total = TILES_PER_WG * csz;
            for(index_int idx = tid; idx < total; idx += BLOCK)
            {
                index_int tl     = idx / csz;
                index_int cc     = idx % csz;
                index_int tg_idx = t_base + tl;

                float V[16];
                if(tg_idx < TOTAL_TILES)
                {
                    index_int ic = c_base + c_chunk + cc;
                    index_int tr = tg_idx / TILES_W;
                    index_int tc = tg_idx % TILES_W;
                    diff_int ih0 = static_cast<diff_int>(tr * OUT_TILE) - 1;
                    diff_int iw0 = static_cast<diff_int>(tc * OUT_TILE) - 1;
                    float d[16];

                    // Interior tile: all 4 rows fully in-bounds → float4 loads
                    bool interior = ih0 >= 0 and (ih0 + 3) < static_cast<diff_int>(H) and
                                    iw0 >= 0 and (iw0 + 3) < static_cast<diff_int>(W);
                    if(interior)
                    {
                        index_int base = (n_val * C + ic) * H * W +
                                         static_cast<index_int>(ih0) * W +
                                         static_cast<index_int>(iw0);
                        // 4 vectorized loads (one per row, 128 bits each)
                        for(index_int i = 0; i < ALPHA; i++)
                        {
                            const float* row = &input[base + i * W];
                            d[i * 4]     = row[0];
                            d[i * 4 + 1] = row[1];
                            d[i * 4 + 2] = row[2];
                            d[i * 4 + 3] = row[3];
                        }
                    }
                    else
                    {
                        // Border tile: scalar loads with boundary checks
                        for(index_int i = 0; i < ALPHA; i++)
                        {
                            diff_int ih = ih0 + static_cast<diff_int>(i);
                            bool ih_ok  = ih >= 0 and ih < static_cast<diff_int>(H);
                            index_int rb =
                                ((n_val * C + ic) * H + static_cast<index_int>(ih)) * W;
                            for(index_int j = 0; j < ALPHA; j++)
                            {
                                diff_int iw = iw0 + static_cast<diff_int>(j);
                                d[i * 4 + j] =
                                    (ih_ok and iw >= 0 and
                                     iw < static_cast<diff_int>(W))
                                        ? input[rb + static_cast<index_int>(iw)]
                                        : 0.0f;
                            }
                        }
                    }

                    // Ensure all loads are issued before transform
                    __builtin_amdgcn_sched_barrier(1 << 4); // VMEM read barrier

                    // B^T * d * B transform (only adds/subs)
                    float tmp[16];
                    for(index_int j = 0; j < 4; j++)
                    {
                        float d0 = d[j], d1 = d[4 + j], d2 = d[8 + j], d3 = d[12 + j];
                        tmp[j]      = d0 - d2;
                        tmp[4 + j]  = d1 + d2;
                        tmp[8 + j]  = d2 - d1;
                        tmp[12 + j] = d1 - d3;
                    }
                    for(index_int i = 0; i < 4; i++)
                    {
                        index_int r = i * 4;
                        float a = tmp[r], b = tmp[r + 1], c2 = tmp[r + 2], dd = tmp[r + 3];
                        V[r]     = a - c2;
                        V[r + 1] = b + c2;
                        V[r + 2] = c2 - b;
                        V[r + 3] = b - dd;
                    }
                }
                else
                {
                    for(index_int p = 0; p < ALPHA2; p++)
                        V[p] = 0.0f;
                }
                for(index_int p = 0; p < ALPHA2; p++)
                    lds_v[p * V_PLANE + tl * CHUNK_C + cc] = V[p];
            }
        }

        // === Phase 1b: Filter transform → LDS (or load pretransformed) ===
        if constexpr(PRETRANSFORMED)
        {
            // weight is [K][C_PER_GRP][16], load one (cc,kl) pair per iteration
            index_int total = csz * k_actual;
            for(index_int idx = tid; idx < total; idx += BLOCK)
            {
                index_int cc  = idx / k_actual;
                index_int kl  = idx % k_actual;
                index_int kk  = k_base + kl;
                index_int src = (kk * C_PER_GRP + c_chunk + cc) * ALPHA2;
                for(index_int p = 0; p < ALPHA2; p++)
                    lds_u[p * U_PLANE + cc * K_PER_WG + kl] = weight[src + p];
            }
        }
        else
        {
            // weight is [K][C_PER_GRP][3][3], transform and write to LDS
            index_int total = csz * k_actual;
            for(index_int idx = tid; idx < total; idx += BLOCK)
            {
                index_int cc = idx / k_actual;
                index_int kl = idx % k_actual;
                index_int kk = k_base + kl;
                float g[9], U[16];
                index_int w_off = (kk * C_PER_GRP + c_chunk + cc) * 9;
                for(index_int p = 0; p < 9; p++)
                    g[p] = weight[w_off + p];
                filter_xform(g, U);
                for(index_int p = 0; p < ALPHA2; p++)
                    lds_u[p * U_PLANE + cc * K_PER_WG + kl] = U[p];
            }
        }

        __builtin_amdgcn_sched_barrier(1 << 7);
        __syncthreads();
        __builtin_amdgcn_sched_barrier(0);

        // === Phase 2: Tiled GEMM from LDS ===
        // T_TILE×K_TILE outer product per LDS read pair.
        // FMA:LDS ratio = T_TILE*K_TILE : (T_TILE+K_TILE).
        // Larger tiles = better ratio (v30 uses 8×8 for 4:1).
        __builtin_amdgcn_s_setprio(1); // High priority for compute
        for(index_int cc = 0; cc < csz; cc++)
        {
            for(index_int p = 0; p < ALPHA2; p++)
            {
                // Load T_TILE V values and K_TILE U values
                float v[T_TILE];
                for(index_int tm = 0; tm < T_TILE; tm++)
                    v[tm] = lds_v[p * V_PLANE + (my_t0 + tm) * CHUNK_C + cc];
                float u[K_TILE];
                for(index_int tn = 0; tn < K_TILE; tn++)
                    u[tn] = lds_u[p * U_PLANE + cc * K_PER_WG + my_k0 + tn];

                // Outer product: T_TILE×K_TILE FMAs
                for(index_int tm = 0; tm < T_TILE; tm++)
                    for(index_int tn = 0; tn < K_TILE; tn++)
                        acc[(tm * K_TILE + tn) * ALPHA2 + p] =
                            fmaf_(v[tm], u[tn], acc[(tm * K_TILE + tn) * ALPHA2 + p]);
            }
        }
        __builtin_amdgcn_s_setprio(0); // Back to normal
        __syncthreads();
    }

    // === Phase 3: Output transform and store ===
    for(index_int tm = 0; tm < T_TILE; tm++)
    {
        index_int tile_idx = t_base + my_t0 + tm;
        if(tile_idx >= TOTAL_TILES)
            continue;
        index_int tr  = tile_idx / TILES_W;
        index_int tc  = tile_idx % TILES_W;
        index_int oh0 = tr * OUT_TILE;
        index_int ow0 = tc * OUT_TILE;
        for(index_int tn = 0; tn < K_TILE; tn++)
        {
            index_int kk = k_base + my_k0 + tn;
            if(kk >= K)
                continue;
            float Y[4];
            output_xform(acc + (tm * K_TILE + tn) * ALPHA2, Y);
            for(index_int i = 0; i < OUT_TILE; i++)
            {
                index_int oh = oh0 + i;
                if(oh >= H)
                    break;
                index_int row = ((n_val * K + kk) * H + oh) * W;
                for(index_int j = 0; j < OUT_TILE; j++)
                {
                    index_int ow = ow0 + j;
                    if(ow >= W)
                        break;
                    output[row + ow] = Y[i * OUT_TILE + j];
                }
            }
        }
    }
}

} // namespace winograd
} // namespace migraphx

#endif // MIGRAPHX_GUARD_KERNELS_WINOGRAD_HPP
