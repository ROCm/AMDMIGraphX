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
#ifndef MIGRAPHX_GUARD_KERNELS_WINOGRAD_CONV_FP32_HPP
#define MIGRAPHX_GUARD_KERNELS_WINOGRAD_CONV_FP32_HPP

#include <migraphx/kernels/array.hpp>
#include <migraphx/kernels/bit_cast.hpp>
#include <migraphx/kernels/dpp.hpp>
#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/tensor_view.hpp>
#include <migraphx/kernels/vec.hpp>
#include <migraphx/kernels/types.hpp>
#include <migraphx/kernels/uninitialized_buffer.hpp>
#include <migraphx/kernels/functional.hpp>
#include <migraphx/kernels/integral_constant.hpp>

namespace migraphx {

// FMA + DPP Winograd F(2x2, 3x3) for fp32 on gfx12 (RDNA4, wave32), modeled on
// MIOpen's Conv_Winograd_v40_6_0_gfx12_fp32_f2x3_stride1 (an FMA/DPP kernel,
// NOT a matrix-core/WMMA kernel).
//
// The winograd 4x4 input tile is laid out so that lane%4 = the winograd
// "v" column (the tile's W axis, positions 0..3) and the 4 winograd "u" rows
// (the tile's H axis) live in registers. With this mapping the winograd
// elementwise product M[u,v] = sum_c U_c[u,v] * V_c[u,v] is entirely
// lane-local (each lane owns one v column), so the channel contraction is a
// plain per-lane FMA accumulation with no cross-lane traffic. Only the input
// and output transforms need to cross the v axis, and those use intra-quad
// cross-lane shuffles (DPP for the input transform, ds_swizzle/bpermute for the
// output) -- no shared memory (LDS) is allocated.
//
// Transforms (canonical Lavin-Gray F(2,3)):
//   B^T = | 1  0 -1  0 |   A^T = | 1  1  1  0 |   G = | 1    0    0   |
//         | 0  1  1  0 |         | 0  1 -1 -1 |       | .5   .5   .5  |
//         | 0 -1  1  0 |                             | .5  -.5   .5  |
//         | 0  1  0 -1 |                             | 0    0    1   |
// U = G g G^T is precomputed on the host as a [4,4,K,C] literal (u,v,k,c).
// The v-axis DPP butterfly (quad_perm:[2,2,1,1]) can only realize the input
// transform's v=3 column with a sign flip (d3-d1 instead of d1-d3); the host
// weight bakes a matching negation into U[:,3,:,:] so the product is exact.

// gfx12 buffer-resource word 3 (from composable_kernel): makes raw buffer loads
// return 0 for out-of-range byte offsets, so bounds/halo checks collapse to an
// offset select against a sentinel instead of a branch per load.
constexpr uint32_t winograd_fp32_buffer_rsrc_word3 = 0x31004000;

__device__ inline auto wino_fp32_make_rsrc(const float* p, uint32_t byte_count)
{
    auto* base = const_cast<float*>(p); // NOLINT(cppcoreguidelines-pro-type-const-cast)
    return __builtin_amdgcn_make_buffer_rsrc(base, 0, byte_count, winograd_fp32_buffer_rsrc_word3);
}

__device__ inline float wino_fp32_load(__amdgpu_buffer_rsrc_t rsrc, int byte_offset)
{
    uint32_t v = __builtin_amdgcn_raw_buffer_load_b32(rsrc, byte_offset, 0, 0);
    return bit_cast<float>(v);
}

// 4 contiguous fp32 (b128). gfx12 buffer loads tolerate 4-byte alignment.
__device__ inline vec<float, 4> wino_fp32_load4(__amdgpu_buffer_rsrc_t rsrc, int byte_offset)
{
    auto v = __builtin_amdgcn_raw_buffer_load_b128(rsrc, byte_offset, 0, 0);
    return bit_cast<vec<float, 4>>(v);
}

// quad_perm:[2,2,1,1] control byte: lane0<-2, lane1<-2, lane2<-1, lane3<-1.
//   0b01'01'10'10 = 0x5A. Broadcasts lane2 to {0,1} and lane1 to {2,3}, which
//   is the pair of neighbours the 4-point B^T needs on the lane (v) axis.
constexpr unsigned int dpp_quad_2211 = 0x5A;

// Input transform, v axis (across the 4 lanes of a quad). Given this lane's raw
// datum d for one tile row, returns P = B^T applied along the v (W) axis.
//   lane0: d0 - d2   lane1: d1 + d2   lane2: d2 - d1   lane3: d3 - d1
// shuf_sign encodes the +/- on the shuffled neighbour (self coefficient is
// always +1); lane3 uses -1, which yields the sign-variant d3-d1 that the host
// weight compensates for.
__device__ inline float wino_f23_bt_v(float d, float shuf_sign)
{
    float s = dpp_mov<dpp_quad_2211>(d);
    return d + s * shuf_sign;
}

// Input transform, u axis (across the 4 registers P[0..3]). B^T along H.
__device__ inline array<float, 4> wino_f23_bt_u(const array<float, 4>& p)
{
    return {p[0] - p[2], p[1] + p[2], p[2] - p[1], p[1] - p[3]};
}

// FMA + DPP Winograd F(2x2, 3x3) kernel.
//   NW    : waves per workgroup (each wave = 32 lanes = 8 quads = 8*TILES tiles).
//   KO    : output channels held per lane (register K tile).
//   TILES : winograd tiles processed per quad (amortizes the weight load).
//   SK    : within-workgroup channel-split factor. The NW waves form NW/SK
//           NT-groups; the SK waves of a group cover the SAME tiles but split
//           the channel contraction (each does 1/SK of the channels), then
//           reduce their partial M accumulators through LDS. SK=1 is the plain
//           no-split path (no LDS). SK>1 fills otherwise-idle waves and cuts
//           per-wave input traffic on shapes with few tiles + many channels.
//
// The channel contraction is unrolled by CU=4 so the shared weight can be read
// with one b128 per (u,k) covering 4 channels, and so the load latency of the
// per-channel input transform is pipelined across 4 channels.
//
// PostInput / F / Inputs...: fused pointwise post-op, same contract as the
// fp16 kernel -- f(cast(y), inputs[idx]...) is applied at each output position,
// collapsing to a plain cast when F = op::id{} and Inputs... is empty.
template <index_int NW,
          index_int KO,
          index_int TILES,
          index_int SK,
          class PostInput,
          class F,
          class Output,
          class Input,
          class Weights,
          class... Inputs>
// NOLINTNEXTLINE(readability-function-size)
__device__ void
winograd_conv_f23_fp32(F f, Output output, Input x, Weights weights, Inputs... inputs)
{
    static_assert(KO >= 1, "KO must be >= 1");
    static_assert(TILES >= 1, "TILES must be >= 1");
    static_assert(SK >= 1 and SK <= NW and (NW % SK) == 0, "SK must divide NW");
    constexpr index_int CU = 4; // channel unroll (b128 weight = 4 channels)

    auto idx       = make_index();
    auto out_shape = output.get_shape();
    auto x_shape   = x.get_shape();
    auto w_shape   = weights.get_shape();

    const auto out_c = out_shape.lens[1];
    const auto out_h = out_shape.lens[2];
    const auto out_w = out_shape.lens[3];
    const auto n     = out_shape.lens[0];
    const auto in_c  = x_shape.lens[1];
    const auto in_h  = x_shape.lens[2];
    const auto in_w  = x_shape.lens[3];

    const auto tiles_w  = (out_w + 1) / 2;
    const auto tiles_h  = (out_h + 1) / 2;
    const auto nt_total = n * tiles_h * tiles_w;

    constexpr index_int quads_per_wave = 32 / 4;         // 8 quads per wave
    constexpr index_int nt_groups      = NW / SK;         // independent tile groups per WG
    constexpr index_int quads_per_wg   = quads_per_wave * nt_groups; // tiles-worth of quads
    const auto k_blocks                = (out_c + KO - 1) / KO;

    // One workgroup per k_block; its nt_groups NT-groups cover a contiguous run
    // of tiles (the SK waves of a group split the channel contraction, not the
    // tiles). Consecutive workgroups cycle the k_block (idx.group % k_blocks) so
    // concurrently-scheduled workgroups tend to share input tiles (same
    // tile_group, different output channels) -> input-cache reuse.
    const index_int k_block    = idx.group % k_blocks;
    const index_int tile_group = idx.group / k_blocks;
    const index_int k_base     = k_block * KO;

    const index_int lane         = idx.local % 32;
    const index_int wave_id      = idx.local / 32;
    const index_int wave_nt_idx  = wave_id / SK; // which NT-group (tile range)
    const index_int wave_sk_part = wave_id % SK; // which channel subset
    const index_int v_col        = lane % 4;
    const index_int quad_in_wave = lane / 4;
    const index_int quad_id =
        tile_group * quads_per_wg + wave_nt_idx * quads_per_wave + quad_in_wave;

    // Shuffle-sign for the input v-axis butterfly (see wino_f23_bt_v).
    const float in_shuf_sign = (v_col == 1) ? 1.0f : -1.0f;

    // Per-tile geometry for the TILES tiles this quad owns.
    array<index_int, TILES> n_arr{};
    array<index_int, TILES> th_arr{};
    array<index_int, TILES> tw_arr{};
    array<bool, TILES> active_arr{};
    repeat_c<TILES>([&](auto tt) {
        constexpr index_int t = tt;
        const index_int nt    = quad_id * TILES + t;
        const bool active     = nt < nt_total;
        const auto tile =
            active ? array<index_int, 3>{n, tiles_h, tiles_w}.multi(nt) : array<index_int, 3>{};
        n_arr[t]      = tile[0];
        th_arr[t]     = tile[1];
        tw_arr[t]     = tile[2];
        active_arr[t] = active;
    });

    // ---- Buffer resources + precomputed base offsets ----
    const auto x_str            = x_shape.strides; // {sn, sc, sh, sw}
    const uint32_t x_byte_count = static_cast<uint32_t>(x_shape.element_space()) * sizeof(float);
    auto x_rsrc                 = wino_fp32_make_rsrc(x.data(), x_byte_count);
    const int32_t x_oob         = static_cast<int32_t>(x_byte_count);
    const index_int c_stride_x  = x_str[1];

    const auto w_str            = w_shape.strides; // {su, sv, sk, sc}
    const uint32_t w_byte_count = static_cast<uint32_t>(w_shape.element_space()) * sizeof(float);
    auto w_rsrc                 = wino_fp32_make_rsrc(weights.data(), w_byte_count);
    const int32_t w_oob         = static_cast<int32_t>(w_byte_count);
    // The b128-over-channels weight load below reads 4 contiguous channels, so
    // the weight's C axis must be innermost (stride 1) -- guaranteed by the host
    // U literal layout [4,4,K,C].
    MIGRAPHX_ASSERT(w_str[3] == 1);

    // Per (tile,row) input byte offset for channel 0 of this lane's column; the
    // channel stride is added in the loop. OOB rows/cols use the sentinel so the
    // buffer load returns 0 (the winograd zero-padding halo).
    array<array<int32_t, 4>, TILES> x_off{};
    repeat_c<TILES>([&](auto tt) {
        constexpr index_int t = tt;
        const int h0          = static_cast<int>(2 * th_arr[t]) - 1;
        const int w0          = static_cast<int>(2 * tw_arr[t]) - 1;
        const int ww          = w0 + static_cast<int>(v_col);
        const bool w_in       = active_arr[t] and ww >= 0 and ww < static_cast<int>(in_w);
        repeat_c<4>([&](auto aa) {
            constexpr int a = aa;
            const int hh    = h0 + a;
            const bool ok   = w_in and hh >= 0 and hh < static_cast<int>(in_h);
            x_off[t][a]     = ok ? static_cast<int32_t>((n_arr[t] * x_str[0] +
                                                     static_cast<index_int>(hh) * x_str[2] +
                                                     static_cast<index_int>(ww) * x_str[3]) *
                                                    sizeof(float))
                                 : x_oob;
        });
    });

    // Weight byte-offset bases for this lane's v column (channel 0). Computed
    // inline in the loop rather than precomputed per (u,k) so we don't pin
    // 4*KO offset registers live across the channel loop. w_str[3] (the c
    // stride) is 1, so channels are contiguous -> b128 over CU.
    const int32_t w_lane_base =
        static_cast<int32_t>((v_col * w_str[1] + k_base * w_str[2]) * sizeof(float));
    const int32_t w_u_stride = static_cast<int32_t>(w_str[0] * sizeof(float));
    const int32_t w_k_stride = static_cast<int32_t>(w_str[2] * sizeof(float));
    auto w_byte_off          = [&](index_int u, index_int k) {
        return (k_base + k < out_c) ? (w_lane_base + static_cast<int32_t>(u) * w_u_stride +
                                       static_cast<int32_t>(k) * w_k_stride)
                                             : w_oob;
    };

    // Accumulators M[u][t][k].
    array<array<array<float, KO>, TILES>, 4> m{};

    auto do_channels = [&](index_int c0, index_int nchan) {
        // Input transform for nchan channels -> v_reg[t][u][cu].
        array<array<array<float, CU>, 4>, TILES> v_reg{};
        repeat_c<TILES>([&](auto tt) {
            constexpr index_int t = tt;
            repeat_c<CU>([&](auto cc) {
                constexpr index_int cu = cc;
                if(cu >= nchan)
                    return;
                const int32_t coff = static_cast<int32_t>((c0 + cu) * c_stride_x * sizeof(float));
                array<float, 4> p{};
                repeat_c<4>([&](auto aa) {
                    constexpr int a = aa;
                    const float d   = wino_fp32_load(x_rsrc, x_off[t][a] + coff);
                    p[a]            = wino_f23_bt_v(d, in_shuf_sign);
                });
                const auto vu = wino_f23_bt_u(p);
                repeat_c<4>([&](auto uu) { v_reg[t][uu][cu] = vu[uu]; });
            });
        });
        // Weight load (b128 over CU channels) + FMA accumulate. The FMA nest is
        // channel-outer (cu), then (k, t): each cu-iteration updates KO*TILES
        // *independent* accumulators, so consecutive FMAs are dependency-free and
        // the matrix pipe stays busy. Accumulating cu-inner instead would chain
        // all CU contributions into one accumulator (a length-CU serial FMA
        // chain), which stalls on gfx12's VALU write latency. This u's KO weight
        // vectors are loaded up front so they can feed the cu loop.
        const int32_t coff_w = static_cast<int32_t>(c0 * sizeof(float));
        repeat_c<4>([&](auto uu) {
            constexpr index_int u = uu;
            array<vec<float, CU>, KO> wv{};
            repeat_c<KO>([&](auto kk) {
                constexpr index_int k    = kk;
                const int32_t w_off_base = w_byte_off(u, k) + coff_w;
                if(nchan == CU)
                {
                    wv[k] = wino_fp32_load4(w_rsrc, w_off_base);
                }
                else
                {
                    repeat_c<CU>([&](auto cc) {
                        constexpr index_int cu = cc;
                        wv[k][cu] =
                            (cu < nchan)
                                ? wino_fp32_load(
                                      w_rsrc, w_off_base + static_cast<int32_t>(cu * sizeof(float)))
                                : 0.0f;
                    });
                }
            });
            repeat_c<CU>([&](auto cc) {
                constexpr index_int cu = cc;
                if(cu >= nchan)
                    return;
                repeat_c<KO>([&](auto kk) {
                    constexpr index_int k = kk;
                    repeat_c<TILES>([&](auto tt) {
                        constexpr index_int t = tt;
                        m[u][t][k] += v_reg[t][u][cu] * wv[k][cu];
                    });
                });
            });
        });
    };

    // Channel loop, CU channels per step. With SK>1 the CU-blocks are split
    // round-robin across the SK waves of this NT-group (each wave sums 1/SK of
    // the channels into its own partial M).
    for(index_int cb = wave_sk_part; cb * CU < in_c; cb += SK)
    {
        const index_int c     = cb * CU;
        const index_int avail = in_c - c;
        do_channels(c, avail < CU ? avail : CU);
    }

    // ---- Split-c cross-wave reduce (SK>1): sum the SK partial M accumulators
    // of this NT-group through LDS; the wave_sk_part==0 wave ends up with the
    // full M and does the output transform + writeback. ----
    constexpr index_int m_per_lane = 4 * TILES * KO;
    constexpr index_int red_len    = (SK > 1) ? (NW * 32 * m_per_lane) : 1;
    __shared__ uninitialized_buffer<float, red_len> m_reduce;
    if constexpr(SK > 1)
    {
        const index_int lane_base = (wave_id * 32 + lane) * m_per_lane;
        repeat_c<4>([&](auto uu) {
            constexpr index_int u = uu;
            repeat_c<TILES>([&](auto tt) {
                constexpr index_int t = tt;
                repeat_c<KO>([&](auto kk) {
                    constexpr index_int k   = kk;
                    constexpr index_int off = u * (TILES * KO) + t * KO + k;
                    m_reduce[lane_base + off] = m[u][t][k];
                });
            });
        });
        __syncthreads();
        if(wave_sk_part == 0)
        {
            for(index_int s = 1; s < SK; ++s)
            {
                const index_int s_base = ((wave_nt_idx * SK + s) * 32 + lane) * m_per_lane;
                repeat_c<4>([&](auto uu) {
                    constexpr index_int u = uu;
                    repeat_c<TILES>([&](auto tt) {
                        constexpr index_int t = tt;
                        repeat_c<KO>([&](auto kk) {
                            constexpr index_int k   = kk;
                            constexpr index_int off = u * (TILES * KO) + t * KO + k;
                            m[u][t][k] += m_reduce[s_base + off];
                        });
                    });
                });
            }
        }
        else
        {
            return; // only wave_sk_part==0 writes back
        }
    }

    // ---- Output transform A^T M A + writeback ----
    using out_type = typename Output::type;
    repeat_c<TILES>([&](auto tt) {
        constexpr index_int t = tt;
        if(not active_arr[t])
            return;
        const index_int oh0 = 2 * th_arr[t];
        const index_int ow0 = 2 * tw_arr[t];
        repeat_c<KO>([&](auto kk) {
            constexpr index_int k2 = kk;
            const index_int k      = k_base + k2;
            if(k >= out_c)
                return;

            // u-axis reduce (registers): N[i] for i = 0,1.
            const float m0 = m[0][t][k2];
            const float m1 = m[1][t][k2];
            const float m2 = m[2][t][k2];
            const float m3 = m[3][t][k2];
            float n_row0   = m0 + m1 + m2; // i = 0
            float n_row1   = m1 - m2 - m3; // i = 1

            // v-axis reduce (cross-lane within the quad). Every lane must join
            // the gathers, but only lanes 0 and 1 (output columns 0 and 1) store,
            // so bail the other two before forming the results.
            const float g0_1 = readlane_xor<1>(n_row0);
            const float g0_2 = readlane_xor<2>(n_row0);
            const float g0_3 = readlane_xor<3>(n_row0);
            const float g1_1 = readlane_xor<1>(n_row1);
            const float g1_2 = readlane_xor<2>(n_row1);
            const float g1_3 = readlane_xor<3>(n_row1);
            if(v_col > 1)
                return;

            const index_int ow = ow0 + v_col;
            if(ow >= out_w)
                return;
            // On lane0 the "+n1+n2" form equals Y[i][col0]; on lane1 the
            // "-n3-n2" form equals Y[i][col1].
            const float y_i0 = (v_col == 0) ? (n_row0 + g0_1 + g0_2) : (n_row0 - g0_3 - g0_2);
            const float y_i1 = (v_col == 0) ? (n_row1 + g1_1 + g1_2) : (n_row1 - g1_3 - g1_2);
            auto store       = [&](index_int oh, float y) {
                const array<index_int, 4> oid{n_arr[t], k, oh, ow};
                output[oid] = static_cast<out_type>(f(static_cast<PostInput>(y), inputs[oid]...));
            };
            store(oh0, y_i0);
            if(oh0 + 1 < out_h)
                store(oh0 + 1, y_i1);
        });
    });
}

} // namespace migraphx

#endif // MIGRAPHX_GUARD_KERNELS_WINOGRAD_CONV_FP32_HPP
