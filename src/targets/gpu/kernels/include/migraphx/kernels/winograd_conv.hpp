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
#ifndef MIGRAPHX_GUARD_KERNELS_WINOGRAD_CONV_HPP
#define MIGRAPHX_GUARD_KERNELS_WINOGRAD_CONV_HPP

#include <migraphx/kernels/array.hpp>
#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/tensor_view.hpp>
#include <migraphx/kernels/vec.hpp>
#include <migraphx/kernels/types.hpp>
#include <migraphx/kernels/uninitialized_buffer.hpp>
#include <migraphx/kernels/functional.hpp>
#include <migraphx/kernels/integral_constant.hpp>

namespace migraphx {

// gfx12 buffer-resource word 3 constant (from composable_kernel).
// Setting this in the SRD makes raw_buffer_load_* return 0 for OOB accesses,
// which lets us drop the per-element bounds checks in the input transform.
#define MIGRAPHX_BUFFER_RSRC_3RD_DWORD_GFX12 0x31004000

// Pure inline-asm WMMA. Helps the compiler not reorder this across other
// VALU ops, keeping the dependency chain tight.
__device__ inline vec<float, 8> wmma_asm(vec<half, 8> a, vec<half, 8> b, vec<float, 8> c)
{
    asm volatile("v_wmma_f32_16x16x16_f16 %0, %1, %2, %0" : "+v"(c) : "v"(a), "v"(b));
    return c;
}

// Inline-asm WMMA pair: issue both loads first, then both WMMAs back-to-back.
// The asm block forces the compiler to keep this sequence atomic, preventing
// it from sinking the WMMAs further apart from their loads.
__device__ inline void wmma_pair_asm(vec<half, 8> a0,
                                     vec<half, 8> b0,
                                     vec<half, 8> a1,
                                     vec<half, 8> b1,
                                     vec<float, 8>& m0,
                                     vec<float, 8>& m1)
{
    asm volatile("v_wmma_f32_16x16x16_f16 %0, %2, %3, %0\n\t"
                 "v_wmma_f32_16x16x16_f16 %1, %4, %5, %1"
                 : "+v"(m0), "+v"(m1)
                 : "v"(a0), "v"(b0), "v"(a1), "v"(b1));
}

// Quad of WMMAs in a single inline-asm block. Forces the compiler to issue
// them back-to-back (each is 8-cycle wait state but to a DIFFERENT
// accumulator, so the next can issue ~1 cycle later). The compiler is then
// free to schedule the alpha-fold v_add_f32 ops *outside* this block, which
// gives them a continuous block of VALU cycles to fill while the matrix pipe
// processes the WMMAs.
__device__ inline void wmma_quad_asm(vec<half, 8> a0,
                                     vec<half, 8> b0,
                                     vec<half, 8> a1,
                                     vec<half, 8> b1,
                                     vec<half, 8> a2,
                                     vec<half, 8> b2,
                                     vec<half, 8> a3,
                                     vec<half, 8> b3,
                                     vec<float, 8>& m0,
                                     vec<float, 8>& m1,
                                     vec<float, 8>& m2,
                                     vec<float, 8>& m3)
{
    asm volatile("v_wmma_f32_16x16x16_f16 %0, %4, %5, %0\n\t"
                 "v_wmma_f32_16x16x16_f16 %1, %6, %7, %1\n\t"
                 "v_wmma_f32_16x16x16_f16 %2, %8, %9, %2\n\t"
                 "v_wmma_f32_16x16x16_f16 %3, %10, %11, %3"
                 : "+v"(m0), "+v"(m1), "+v"(m2), "+v"(m3)
                 : "v"(a0), "v"(b0), "v"(a1), "v"(b1), "v"(a2), "v"(b2), "v"(a3), "v"(b3));
}

// Octet of WMMAs in a single inline-asm block. Costs 8 live fp32 vec<8>
// accumulators (64 VGPRs) but gives the matrix pipe a continuous run.
__device__ inline void wmma_octet_asm(vec<half, 8> a0,
                                      vec<half, 8> b0,
                                      vec<half, 8> a1,
                                      vec<half, 8> b1,
                                      vec<half, 8> a2,
                                      vec<half, 8> b2,
                                      vec<half, 8> a3,
                                      vec<half, 8> b3,
                                      vec<half, 8> a4,
                                      vec<half, 8> b4,
                                      vec<half, 8> a5,
                                      vec<half, 8> b5,
                                      vec<half, 8> a6,
                                      vec<half, 8> b6,
                                      vec<half, 8> a7,
                                      vec<half, 8> b7,
                                      vec<float, 8>& m0,
                                      vec<float, 8>& m1,
                                      vec<float, 8>& m2,
                                      vec<float, 8>& m3,
                                      vec<float, 8>& m4,
                                      vec<float, 8>& m5,
                                      vec<float, 8>& m6,
                                      vec<float, 8>& m7)
{
    asm volatile("v_wmma_f32_16x16x16_f16 %0, %8, %9, %0\n\t"
                 "v_wmma_f32_16x16x16_f16 %1, %10, %11, %1\n\t"
                 "v_wmma_f32_16x16x16_f16 %2, %12, %13, %2\n\t"
                 "v_wmma_f32_16x16x16_f16 %3, %14, %15, %3\n\t"
                 "v_wmma_f32_16x16x16_f16 %4, %16, %17, %4\n\t"
                 "v_wmma_f32_16x16x16_f16 %5, %18, %19, %5\n\t"
                 "v_wmma_f32_16x16x16_f16 %6, %20, %21, %6\n\t"
                 "v_wmma_f32_16x16x16_f16 %7, %22, %23, %7"
                 : "+v"(m0), "+v"(m1), "+v"(m2), "+v"(m3), "+v"(m4), "+v"(m5), "+v"(m6), "+v"(m7)
                 : "v"(a0),
                   "v"(b0),
                   "v"(a1),
                   "v"(b1),
                   "v"(a2),
                   "v"(b2),
                   "v"(a3),
                   "v"(b3),
                   "v"(a4),
                   "v"(b4),
                   "v"(a5),
                   "v"(b5),
                   "v"(a6),
                   "v"(b6),
                   "v"(a7),
                   "v"(b7));
}

__device__ inline auto make_input_buffer_rsrc(const half* p, uint32_t byte_count)
{
    return __builtin_amdgcn_make_buffer_rsrc(
        const_cast<half*>(p), 0, byte_count, MIGRAPHX_BUFFER_RSRC_3RD_DWORD_GFX12);
}

// Lane-indexed raw buffer load of a single fp16. OOB returns 0.
__device__ inline half buffer_load_half(__amdgpu_buffer_rsrc_t rsrc, int byte_offset)
{
    uint16_t v = __builtin_amdgcn_raw_buffer_load_b16(rsrc, byte_offset, 0, 0);
    half h;
    __builtin_memcpy(&h, &v, 2);
    return h;
}

// Lane-indexed raw buffer load of 4 fp16 = 8 bytes. OOB bytes return 0.
// Caller is responsible for alignment (byte_offset divisible by 4 to avoid
// faulting; gfx12 buffer loads tolerate 4-byte alignment for b64).
__device__ inline vec<half, 4> buffer_load_half4(__amdgpu_buffer_rsrc_t rsrc, int byte_offset)
{
    auto v = __builtin_amdgcn_raw_buffer_load_b64(rsrc, byte_offset, 0, 0);
    vec<half, 4> result;
    __builtin_memcpy(&result, &v, sizeof(result));
    return result;
}

// F(2x2, 3x3) Winograd transforms used inline by the WMMA path.
// B^T (input):  | 1  0 -1  0 |     A^T (output): | 1  1  1  0 |
//               | 0  1  1  0 |                   | 0  1 -1 -1 |
//               | 0 -1  1  0 |
//               | 0  1  0 -1 |

__device__ inline array<half, 16> winograd_input_transform_f23(const array<half, 16>& d_arr)
{
    using h2 = vec<half, 2>;
    // First pass B^T d: per-row sub/add of 4 columns. All 4 columns are
    // independent so we express each row as TWO packed half2 ops --
    // generates v_pk_add_f16 / v_pk_sub_f16.
    const h2* dp   = reinterpret_cast<const h2*>(d_arr.data());
    const h2 d0_lo = dp[0]; // d[0,0..1]
    const h2 d0_hi = dp[1]; // d[0,2..3]
    const h2 d1_lo = dp[2];
    const h2 d1_hi = dp[3];
    const h2 d2_lo = dp[4];
    const h2 d2_hi = dp[5];
    const h2 d3_lo = dp[6];
    const h2 d3_hi = dp[7];

    const h2 t0_lo = d0_lo - d2_lo;
    const h2 t0_hi = d0_hi - d2_hi;
    const h2 t1_lo = d1_lo + d2_lo;
    const h2 t1_hi = d1_hi + d2_hi;
    const h2 t2_lo = d2_lo - d1_lo;
    const h2 t2_hi = d2_hi - d1_hi;
    const h2 t3_lo = d1_lo - d3_lo;
    const h2 t3_hi = d1_hi - d3_hi;

    // Second pass d B: V[i,j] in terms of t[i,*].
    //   V[i,0] = t[i,0] - t[i,2]
    //   V[i,1] = t[i,1] + t[i,2]
    //   V[i,2] = t[i,2] - t[i,1]
    //   V[i,3] = t[i,1] - t[i,3]
    array<half, 16> v;
    auto fill = [&](int row, h2 t_lo, h2 t_hi) {
        v[row * 4 + 0] = t_lo.x - t_hi.x;
        v[row * 4 + 1] = t_lo.y + t_hi.x;
        v[row * 4 + 2] = t_hi.x - t_lo.y;
        v[row * 4 + 3] = t_lo.y - t_hi.y;
    };
    fill(0, t0_lo, t0_hi);
    fill(1, t1_lo, t1_hi);
    fill(2, t2_lo, t2_hi);
    fill(3, t3_lo, t3_hi);
    return v;
}

// WMMA-based winograd F(2x2, 3x3) kernel for gfx12 wave32.
//
// Per WMMA call (16x16x16 fp16 -> fp32):
//   A operand: lane l holds A[m=l%16, k=(l/16)*8 + i] for i in 0..7  (V8h)
//   B operand: lane l holds B[k=l%16, n=(l/16)*8 + i] for i in 0..7  (V8h)
//   D output : lane l holds D[m=(l/16)*8 + i, n=l%16] for i in 0..7  (V8f)
//
// Block organization:
//   - workgroup = NW waves * 32 lanes (wave32)
//   - All NW waves share K_block (BK = 16 = WMMA M dim); waves split NT
//   - Per wave: 16 K x 16 NT outputs per wp = 256 / 32 lanes = 8 per lane
//   - We stream the output transform per WMMA so we don't keep all 16 wp
//     accumulators alive simultaneously (which would force register spill).
//   - CB must be a multiple of 16 (WMMA K dim).

template <index_int NW, index_int CB, index_int KW, class Output, class Input, class Weights>
__device__ void winograd_conv_f23_wmma(Output output, Input x, Weights u)
{
    static_assert(CB % 16 == 0, "CB must be a multiple of WMMA K (16)");
    static_assert(KW >= 1, "KW must be >= 1");
    constexpr index_int BK          = 16;
    constexpr index_int BT_per_wave = 16;
    constexpr index_int BT          = BT_per_wave * NW;
    constexpr index_int BK_WG       = BK * KW; // K outputs processed per workgroup

    auto idx = make_index();

    auto out_shape = output.get_shape();
    auto x_shape   = x.get_shape();

    const auto N     = out_shape.lens[0];
    const auto K     = out_shape.lens[1];
    const auto H_out = out_shape.lens[2];
    const auto W_out = out_shape.lens[3];
    const auto C     = x_shape.lens[1];

    const auto tiles_w       = (W_out + 1) / 2;
    const auto tiles_h       = (H_out + 1) / 2;
    const auto tiles_per_img = tiles_h * tiles_w;
    const auto NT_total      = N * tiles_per_img;

    // Each workgroup covers KW consecutive K_blocks for one t_block. The V
    // transform is shared across all KW k_blocks, so we only pay the
    // V-transform cost once per (t_block, c_block) instead of KW times.
    const auto k_wg_blocks = (K + BK_WG - 1) / BK_WG;
    const auto k_wg_block  = idx.group % k_wg_blocks;
    const auto t_block     = idx.group / k_wg_blocks;
    const auto k_base      = k_wg_block * BK_WG;
    const auto t_base      = t_block * BT;

    const auto lane    = idx.local % 32;
    const auto wave_id = idx.local / 32;

    // ---- V layout (REGISTER-RESIDENT) ----
    // V values are kept in per-lane registers instead of LDS. The lane
    // assignment is chosen so that each lane already holds the exact 8 fp16
    // values that the WMMA B operand expects for its (wp, nt_lane, c_off..c_off+7)
    // slice — no cross-lane permute needed. Each lane handles:
    //   nt_lane = wave_base + lane%16              (1 nt per lane)
    //   c_lane in c_off + 0..7  with c_off = (lane/16)*8   (8 c per lane)
    // Per lane: 8 c values × 16 wp = 128 fp16 V values (64 VGPRs).
    //
    // This frees 16*BT*(CB+2)*sizeof(half) of LDS — for NW=4 CB=16 that's
    // 36KB, letting many more workgroups fit per CU LDS-wise.
    //
    // U is still LDS-resident — its layout already matches the WMMA A operand
    // and a register-based U would 36× balloon VGPR pressure for KW=2.
    static_assert(CB == 16 or CB == 32, "DPP V path supports CB=16 or CB=32");
    constexpr index_int v_chunks      = CB / 16; // number of 8-c register banks per lane
    constexpr index_int wp_count      = 16;
    __shared__ uninitialized_buffer<half, KW * 16 * BK * CB> u_smem;

    auto u_cache_idx = [&](index_int k_idx, index_int wp, index_int k, index_int c) {
        return k_idx * (16 * BK * CB) + wp * BK * CB + k * CB + c;
    };

    // alpha[wp,r,c] = A^T[r, wp/4] * A[wp%4, c]
    constexpr float at[2][4] = {{1.f, 1.f, 1.f, 0.f}, {0.f, 1.f, -1.f, -1.f}};

    // Y[k_idx][r*2+c][k_offset] running accumulator. KW * 4 outputs * 8 K rows per lane.
    array<array<array<float, 8>, 4>, KW> y{};

    // Buffer descriptors for X (input) and U (weights). raw_buffer_load
    // returns 0 for OOB byte offsets so we don't need explicit bounds checks.
    const auto x_sh             = x_shape.strides;
    const auto* x_data          = x.data();
    const uint32_t x_byte_count = static_cast<uint32_t>(x_shape.element_space()) * sizeof(half);
    auto x_rsrc                 = make_input_buffer_rsrc(x_data, x_byte_count);

    const auto* u_data = u.data();
    const uint32_t u_byte_count =
        static_cast<uint32_t>(u.get_shape().element_space()) * sizeof(half);
    auto u_rsrc = make_input_buffer_rsrc(u_data, u_byte_count);
    // U layout: [16, K, C] -- strides for byte offset computation.
    const auto u_sh = u.get_shape().strides;

    const auto cblocks = (C + CB - 1) / CB;

    // V-in-registers storage: v_lane[c_chunk][c_in_chunk][wp].
    // For CB=16: v_chunks=1, c_in_chunk in 0..7 (lane's 8 c values).
    // For CB=32: v_chunks=2 (two c-chunks; one b128 read per chunk for WMMA).
    array<array<array<half, wp_count>, 8>, v_chunks> v_lane;

    // Cached per-wave / per-lane geometry.
    const index_int wave_nt_base_in_block = wave_id * BT_per_wave;
    const index_int m_in_wave             = lane % 16;
    const index_int c_off                 = (lane / 16) * 8;
    const index_int nt_slot               = wave_nt_base_in_block + m_in_wave;
    const index_int nt_global             = t_base + nt_slot;

    // Lane's tile (n, th, tw). Same for all c (only c changes per tile_idx).
    const bool nt_active   = (nt_global < NT_total);
    const index_int n_idx  = nt_active ? nt_global / tiles_per_img : index_int{0};
    const auto rem_lane    = nt_active ? nt_global - n_idx * tiles_per_img : index_int{0};
    const index_int th_idx = nt_active ? rem_lane / tiles_w : index_int{0};
    const index_int tw_idx = nt_active ? rem_lane - th_idx * tiles_w : index_int{0};
    const int h0           = static_cast<int>(2 * th_idx) - 1;
    const int w0           = static_cast<int>(2 * tw_idx) - 1;
    const int32_t n_off    = static_cast<int32_t>(n_idx * x_sh[0]) * sizeof(half);
    const int32_t sh_b     = static_cast<int32_t>(x_sh[2] * sizeof(half));
    const int32_t sw_b     = static_cast<int32_t>(x_sh[3] * sizeof(half));
    const int32_t hw_off   = h0 * sh_b + w0 * sw_b;

    for(index_int cb = 0; cb < cblocks; ++cb)
    {
        const index_int c_base = cb * CB;

        // ---- Per-lane V compute into registers ----
        // Each lane processes its own 8 c values (per v_chunk). The natural
        // lane mapping (lane%16 -> nt, lane/16*8 -> c_chunk_start) places the
        // V values right where the WMMA B operand expects them — no LDS round
        // trip and no cross-lane permute.
        repeat_c<v_chunks>([&](auto vc_val) {
            constexpr index_int vc = vc_val;
            repeat_c<8>([&](auto ci_val) {
                constexpr index_int ci = ci_val;
                const index_int c_in_block = vc * 16 + c_off + ci;
                const index_int c          = c_base + c_in_block;
                const bool active          = nt_active and (c < C);
                const int32_t base_off = n_off + static_cast<int32_t>(c * x_sh[1]) * sizeof(half);
                const int32_t tile_off = base_off + hw_off;
                const int32_t off =
                    active ? tile_off : static_cast<int32_t>(x_byte_count);

                array<half, 16> d;
                if(sw_b == 2)
                {
                    repeat_c<4>([&](auto i) {
                        const int32_t row_off = off + static_cast<int>(i) * sh_b;
                        auto row              = buffer_load_half4(x_rsrc, row_off);
                        d[i * 4 + 0]          = row.x;
                        d[i * 4 + 1]          = row.y;
                        d[i * 4 + 2]          = row.z;
                        d[i * 4 + 3]          = row.w;
                    });
                }
                else
                {
                    repeat_c<4>([&](auto i) {
                        repeat_c<4>([&](auto j) {
                            const int32_t e_off = off + static_cast<int>(i) * sh_b +
                                                  static_cast<int>(j) * sw_b;
                            d[i * 4 + j] = buffer_load_half(x_rsrc, e_off);
                        });
                    });
                }
                auto V = winograd_input_transform_f23(d);
                repeat_c<16>([&](auto wp) { v_lane[vc][ci][wp] = V[wp]; });
            });
        });

        // ---- Cooperative U load: one b128 per task (8 fp16), for all KW k_blocks ----
        static_assert(CB % 8 == 0, "CB must be a multiple of 8 for b128 U loads");
        constexpr index_int U_TASKS = KW * 16 * BK * (CB / 8);
        idx.local_stride(_c<U_TASKS>, [&](auto task) {
            const index_int c_half     = task % (CB / 8);
            const index_int rest       = task / (CB / 8);
            const index_int k_in_block = rest % BK;
            const index_int rest2      = rest / BK;
            const index_int wp         = rest2 % 16;
            const index_int k_idx      = rest2 / 16;
            const index_int c_in_block = c_half * 8;
            const index_int k          = k_base + k_idx * BK + k_in_block;
            const int32_t off          = static_cast<int32_t>(
                (wp * u_sh[0] + k * u_sh[1] + (c_base + c_in_block) * u_sh[2]) * sizeof(half));
            vec<half, 8> v8;
            if(k < K)
            {
                auto raw = __builtin_amdgcn_raw_buffer_load_b128(u_rsrc, off, 0, 0);
                __builtin_memcpy(&v8, &raw, sizeof(v8));
            }
            else
            {
                v8 = vec<half, 8>{0};
            }
            half* dst       = &u_smem[u_cache_idx(k_idx, wp, k_in_block, c_in_block)];
            *as_vec<8>(dst) = v8;
        });

        __syncthreads();

        // ---- WMMA with fused incremental output transform ----
        constexpr index_int wmma_chunks = CB / 16;

        // Row-sum alpha fold: compute 4 wp (one full wp_i row of M) and apply
        // the A^T M A output transform via S0/S1 row sums into y[k_idx].
        auto fold_row = [&](auto k_idx_val,
                            auto wp_i_val,
                            const vec<float, 8>& m0,
                            const vec<float, 8>& m1,
                            const vec<float, 8>& m2,
                            const vec<float, 8>& m3) {
            const vec<float, 8> s0 = m0 + m1 + m2;
            const vec<float, 8> s1 = m1 - m2 - m3;
            repeat_c<2>([&](auto r) {
                constexpr float coef_r = at[r][wp_i_val];
                if constexpr(coef_r != 0.f)
                {
                    repeat_c<8>([&](auto ki) {
                        float v0, v1;
                        if constexpr(ki == 0)
                        {
                            v0 = s0.s0;
                            v1 = s1.s0;
                        }
                        else if constexpr(ki == 1)
                        {
                            v0 = s0.s1;
                            v1 = s1.s1;
                        }
                        else if constexpr(ki == 2)
                        {
                            v0 = s0.s2;
                            v1 = s1.s2;
                        }
                        else if constexpr(ki == 3)
                        {
                            v0 = s0.s3;
                            v1 = s1.s3;
                        }
                        else if constexpr(ki == 4)
                        {
                            v0 = s0.s4;
                            v1 = s1.s4;
                        }
                        else if constexpr(ki == 5)
                        {
                            v0 = s0.s5;
                            v1 = s1.s5;
                        }
                        else if constexpr(ki == 6)
                        {
                            v0 = s0.s6;
                            v1 = s1.s6;
                        }
                        else
                        {
                            v0 = s0.s7;
                            v1 = s1.s7;
                        }
                        y[k_idx_val][r * 2 + 0][ki] = y[k_idx_val][r * 2 + 0][ki] + coef_r * v0;
                        y[k_idx_val][r * 2 + 1][ki] = y[k_idx_val][r * 2 + 1][ki] + coef_r * v1;
                    });
                }
            });
        };
        // Construct V[wp] for this lane from register-resident v_lane.
        // c_offset selects which c-chunk (0 for CB=16; 0 or 16 for CB=32).
        // The 8 fp16 values come from v_lane[vc][0..7][wp] where vc = c_offset/16.
        auto load_v = [&](index_int wp, index_int c_offset) {
            const index_int vc = c_offset / 16;
            vec<half, 8> b;
            b.s0 = v_lane[vc][0][wp];
            b.s1 = v_lane[vc][1][wp];
            b.s2 = v_lane[vc][2][wp];
            b.s3 = v_lane[vc][3][wp];
            b.s4 = v_lane[vc][4][wp];
            b.s5 = v_lane[vc][5][wp];
            b.s6 = v_lane[vc][6][wp];
            b.s7 = v_lane[vc][7][wp];
            return b;
        };
        auto load_u = [&](index_int k_idx, index_int wp, index_int c_offset) {
            return *as_vec<8>(&u_smem[u_cache_idx(k_idx, wp, m_in_wave, c_offset + c_off)]);
        };

        repeat_c<KW>([&](auto k_idx_val) {
            constexpr int k_idx = k_idx_val;
            repeat_c<4>([&](auto wp_i_val) {
                constexpr int wp_i = wp_i_val;
                vec<float, 8> m0{}, m1{}, m2{}, m3{};
                if constexpr(wmma_chunks == 2)
                {
                    auto a0 = load_u(k_idx, wp_i * 4 + 0, 0);
                    auto b0 = load_v(wp_i * 4 + 0, 0);
                    auto a1 = load_u(k_idx, wp_i * 4 + 1, 0);
                    auto b1 = load_v(wp_i * 4 + 1, 0);
                    auto a2 = load_u(k_idx, wp_i * 4 + 2, 0);
                    auto b2 = load_v(wp_i * 4 + 2, 0);
                    auto a3 = load_u(k_idx, wp_i * 4 + 3, 0);
                    auto b3 = load_v(wp_i * 4 + 3, 0);
                    auto a4 = load_u(k_idx, wp_i * 4 + 0, 16);
                    auto b4 = load_v(wp_i * 4 + 0, 16);
                    auto a5 = load_u(k_idx, wp_i * 4 + 1, 16);
                    auto b5 = load_v(wp_i * 4 + 1, 16);
                    auto a6 = load_u(k_idx, wp_i * 4 + 2, 16);
                    auto b6 = load_v(wp_i * 4 + 2, 16);
                    auto a7 = load_u(k_idx, wp_i * 4 + 3, 16);
                    auto b7 = load_v(wp_i * 4 + 3, 16);
                    vec<float, 8> m4{}, m5{}, m6{}, m7{};
                    wmma_octet_asm(a0,
                                   b0,
                                   a1,
                                   b1,
                                   a2,
                                   b2,
                                   a3,
                                   b3,
                                   a4,
                                   b4,
                                   a5,
                                   b5,
                                   a6,
                                   b6,
                                   a7,
                                   b7,
                                   m0,
                                   m1,
                                   m2,
                                   m3,
                                   m4,
                                   m5,
                                   m6,
                                   m7);
                    m0 = m0 + m4;
                    m1 = m1 + m5;
                    m2 = m2 + m6;
                    m3 = m3 + m7;
                }
                else
                {
                    for(index_int ck = 0; ck < wmma_chunks; ++ck)
                    {
                        const index_int c_offset = ck * 16;
                        auto a0                  = load_u(k_idx, wp_i * 4 + 0, c_offset);
                        auto b0                  = load_v(wp_i * 4 + 0, c_offset);
                        auto a1                  = load_u(k_idx, wp_i * 4 + 1, c_offset);
                        auto b1                  = load_v(wp_i * 4 + 1, c_offset);
                        auto a2                  = load_u(k_idx, wp_i * 4 + 2, c_offset);
                        auto b2                  = load_v(wp_i * 4 + 2, c_offset);
                        auto a3                  = load_u(k_idx, wp_i * 4 + 3, c_offset);
                        auto b3                  = load_v(wp_i * 4 + 3, c_offset);
                        wmma_quad_asm(a0, b0, a1, b1, a2, b2, a3, b3, m0, m1, m2, m3);
                    }
                }
                fold_row(_c<k_idx>, _c<wp_i>, m0, m1, m2, m3);
            });
        });

        __syncthreads();
    }

    // ---- Output writeback for each k_block this workgroup covered ----
    // Reuse the per-lane (n_idx, th_idx, tw_idx) computed up at V-load setup.
    using out_type               = typename Output::type;
    const index_int k_row_offset = c_off; // (lane / 16) * 8, same lane mapping
    if(not nt_active)
        return;

    const auto sn  = out_shape.strides[0];
    const auto sk  = out_shape.strides[1];
    const auto sh  = out_shape.strides[2];
    const auto sw  = out_shape.strides[3];
    auto* out_data = output.data();

    repeat_c<KW>([&](auto k_idx_val) {
        constexpr int k_idx = k_idx_val;
        const index_int base_offset = n_idx * sn +
                                      (k_base + k_idx * BK + k_row_offset) * sk +
                                      (2 * th_idx) * sh + (2 * tw_idx) * sw;
        repeat_c<8>([&](auto ki) {
            const index_int k = k_base + k_idx * BK + k_row_offset + ki;
            if(k < K)
            {
                const index_int kbase = base_offset + ki * sk;
                repeat_c<2>([&](auto i) {
                    const int h_out = static_cast<int>(2 * th_idx) + static_cast<int>(i);
                    if(static_cast<unsigned>(h_out) < H_out)
                    {
                        const index_int hbase = kbase + i * sh;
                        repeat_c<2>([&](auto j) {
                            const int w_out = static_cast<int>(2 * tw_idx) + static_cast<int>(j);
                            if(static_cast<unsigned>(w_out) < W_out)
                            {
                                out_data[hbase + j * sw] =
                                    static_cast<out_type>(y[k_idx][i * 2 + j][ki]);
                            }
                        });
                    }
                });
            }
        });
    });
}

} // namespace migraphx

#endif // MIGRAPHX_GUARD_KERNELS_WINOGRAD_CONV_HPP
