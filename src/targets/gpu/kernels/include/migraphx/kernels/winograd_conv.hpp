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
__device__ inline vec<float, 8> wmma_asm(vec<half, 8> a, vec<half, 8> b,
                                         vec<float, 8> c)
{
    asm volatile("v_wmma_f32_16x16x16_f16 %0, %1, %2, %0"
                 : "+v"(c)
                 : "v"(a), "v"(b));
    return c;
}

// Inline-asm WMMA pair: issue both loads first, then both WMMAs back-to-back.
// The asm block forces the compiler to keep this sequence atomic, preventing
// it from sinking the WMMAs further apart from their loads.
__device__ inline void wmma_pair_asm(vec<half, 8> a0, vec<half, 8> b0,
                                     vec<half, 8> a1, vec<half, 8> b1,
                                     vec<float, 8>& m0, vec<float, 8>& m1)
{
    asm volatile(
        "v_wmma_f32_16x16x16_f16 %0, %2, %3, %0\n\t"
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
__device__ inline void wmma_quad_asm(vec<half, 8> a0, vec<half, 8> b0,
                                     vec<half, 8> a1, vec<half, 8> b1,
                                     vec<half, 8> a2, vec<half, 8> b2,
                                     vec<half, 8> a3, vec<half, 8> b3,
                                     vec<float, 8>& m0, vec<float, 8>& m1,
                                     vec<float, 8>& m2, vec<float, 8>& m3)
{
    asm volatile(
        "v_wmma_f32_16x16x16_f16 %0, %4, %5, %0\n\t"
        "v_wmma_f32_16x16x16_f16 %1, %6, %7, %1\n\t"
        "v_wmma_f32_16x16x16_f16 %2, %8, %9, %2\n\t"
        "v_wmma_f32_16x16x16_f16 %3, %10, %11, %3"
        : "+v"(m0), "+v"(m1), "+v"(m2), "+v"(m3)
        : "v"(a0), "v"(b0), "v"(a1), "v"(b1),
          "v"(a2), "v"(b2), "v"(a3), "v"(b3));
}

// Octet of WMMAs in a single inline-asm block. Costs 8 live fp32 vec<8>
// accumulators (64 VGPRs) but gives the matrix pipe a continuous run.
__device__ inline void wmma_octet_asm(vec<half, 8> a0, vec<half, 8> b0,
                                      vec<half, 8> a1, vec<half, 8> b1,
                                      vec<half, 8> a2, vec<half, 8> b2,
                                      vec<half, 8> a3, vec<half, 8> b3,
                                      vec<half, 8> a4, vec<half, 8> b4,
                                      vec<half, 8> a5, vec<half, 8> b5,
                                      vec<half, 8> a6, vec<half, 8> b6,
                                      vec<half, 8> a7, vec<half, 8> b7,
                                      vec<float, 8>& m0, vec<float, 8>& m1,
                                      vec<float, 8>& m2, vec<float, 8>& m3,
                                      vec<float, 8>& m4, vec<float, 8>& m5,
                                      vec<float, 8>& m6, vec<float, 8>& m7)
{
    asm volatile(
        "v_wmma_f32_16x16x16_f16 %0, %8, %9, %0\n\t"
        "v_wmma_f32_16x16x16_f16 %1, %10, %11, %1\n\t"
        "v_wmma_f32_16x16x16_f16 %2, %12, %13, %2\n\t"
        "v_wmma_f32_16x16x16_f16 %3, %14, %15, %3\n\t"
        "v_wmma_f32_16x16x16_f16 %4, %16, %17, %4\n\t"
        "v_wmma_f32_16x16x16_f16 %5, %18, %19, %5\n\t"
        "v_wmma_f32_16x16x16_f16 %6, %20, %21, %6\n\t"
        "v_wmma_f32_16x16x16_f16 %7, %22, %23, %7"
        : "+v"(m0), "+v"(m1), "+v"(m2), "+v"(m3),
          "+v"(m4), "+v"(m5), "+v"(m6), "+v"(m7)
        : "v"(a0), "v"(b0), "v"(a1), "v"(b1),
          "v"(a2), "v"(b2), "v"(a3), "v"(b3),
          "v"(a4), "v"(b4), "v"(a5), "v"(b5),
          "v"(a6), "v"(b6), "v"(a7), "v"(b7));
}

__device__ inline auto make_input_buffer_rsrc(const half* p,
                                              uint32_t byte_count)
{
    return __builtin_amdgcn_make_buffer_rsrc(
        const_cast<half*>(p), 0, byte_count, MIGRAPHX_BUFFER_RSRC_3RD_DWORD_GFX12);
}

// Lane-indexed raw buffer load of a single fp16. OOB returns 0.
__device__ inline half buffer_load_half(__amdgpu_buffer_rsrc_t rsrc,
                                        int byte_offset)
{
    uint16_t v = __builtin_amdgcn_raw_buffer_load_b16(rsrc, byte_offset, 0, 0);
    half h;
    __builtin_memcpy(&h, &v, 2);
    return h;
}

// Lane-indexed raw buffer load of 4 fp16 = 8 bytes. OOB bytes return 0.
// Caller is responsible for alignment (byte_offset divisible by 4 to avoid
// faulting; gfx12 buffer loads tolerate 4-byte alignment for b64).
__device__ inline vec<half, 4> buffer_load_half4(__amdgpu_buffer_rsrc_t rsrc,
                                                 int byte_offset)
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
    const h2* dp = reinterpret_cast<const h2*>(d_arr.data());
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

// Fallback for float (no packed ops apply)
template <class T>
__device__ inline array<T, 16> winograd_input_transform_f23_scalar(const array<T, 16>& d)
{
    array<T, 16> t;
    repeat_c<4>([&](auto j) {
        t[0 * 4 + j] = d[0 * 4 + j] - d[2 * 4 + j];
        t[1 * 4 + j] = d[1 * 4 + j] + d[2 * 4 + j];
        t[2 * 4 + j] = d[2 * 4 + j] - d[1 * 4 + j];
        t[3 * 4 + j] = d[1 * 4 + j] - d[3 * 4 + j];
    });
    array<T, 16> v;
    repeat_c<4>([&](auto i) {
        v[i * 4 + 0] = t[i * 4 + 0] - t[i * 4 + 2];
        v[i * 4 + 1] = t[i * 4 + 1] + t[i * 4 + 2];
        v[i * 4 + 2] = t[i * 4 + 2] - t[i * 4 + 1];
        v[i * 4 + 3] = t[i * 4 + 1] - t[i * 4 + 3];
    });
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

template <index_int NW,
          index_int CB,
          class Output,
          class Input,
          class Weights>
__device__ void winograd_conv_f23_wmma(Output output, Input x, Weights u)
{
    static_assert(CB % 16 == 0, "CB must be a multiple of WMMA K (16)");
    constexpr index_int BK          = 16;
    constexpr index_int BT_per_wave = 16;
    constexpr index_int BT          = BT_per_wave * NW;

    auto idx = make_index();

    auto out_shape = output.get_shape();
    auto x_shape   = x.get_shape();

    const auto N     = out_shape.lens[0];
    const auto K     = out_shape.lens[1];
    const auto H_out = out_shape.lens[2];
    const auto W_out = out_shape.lens[3];
    const auto C     = x_shape.lens[1];
    // H_in, W_in are no longer needed for bounds checks; the buffer descriptor
    // handles OOB transparently.

    const auto tiles_w       = (W_out + 1) / 2;
    const auto tiles_h       = (H_out + 1) / 2;
    const auto tiles_per_img = tiles_h * tiles_w;
    const auto NT_total      = N * tiles_per_img;

    const auto k_blocks = (K + BK - 1) / BK;
    const auto k_block  = idx.group % k_blocks;
    const auto t_block  = idx.group / k_blocks;
    const auto k_base   = k_block * BK;
    const auto t_base   = t_block * BT;

    const auto lane    = idx.local % 32;
    const auto wave_id = idx.local / 32;

    // LDS: U_lds[wp][k][c] and V_lds[wp][nt][c] (c innermost so WMMA operand
    // load is one ds_load_b128 of 8 contiguous fp16 per lane).
    // Pad the V c stride by 2 fp16 (4 bytes) so that 32 lanes writing
    // consecutive t_slots at stride (CB+2)*2 = 36 bytes hit 32 distinct LDS
    // banks (gcd(36, 128) = 4 = bank width), avoiding a 4-way conflict.
    constexpr index_int V_CB_PAD = CB + 2;
    __shared__ uninitialized_buffer<half, 16 * BK * CB> u_smem;
    __shared__ uninitialized_buffer<half, 16 * BT * V_CB_PAD> v_smem;

    auto u_cache_idx = [&](index_int wp, index_int k, index_int c) {
        return wp * BK * CB + k * CB + c;
    };
    auto v_cache_idx = [&](index_int wp, index_int t, index_int c) {
        return wp * BT * V_CB_PAD + t * V_CB_PAD + c;
    };

    // alpha[wp,r,c] = A^T[r, wp/4] * A[wp%4, c]
    constexpr float at[2][4] = {{1.f, 1.f, 1.f, 0.f}, {0.f, 1.f, -1.f, -1.f}};

    // Y[r*2+c][k_offset] running accumulator. 4 Y outputs * 8 K rows per lane.
    array<array<float, 8>, 4> y{};

    // Buffer descriptors for X (input) and U (weights). raw_buffer_load
    // returns 0 for OOB byte offsets so we don't need explicit bounds checks.
    const auto x_sh    = x_shape.strides;
    const auto* x_data = x.data();
    const uint32_t x_byte_count =
        static_cast<uint32_t>(x_shape.element_space()) * sizeof(half);
    auto x_rsrc = make_input_buffer_rsrc(x_data, x_byte_count);

    const auto* u_data = u.data();
    const uint32_t u_byte_count =
        static_cast<uint32_t>(u.get_shape().element_space()) * sizeof(half);
    auto u_rsrc = make_input_buffer_rsrc(u_data, u_byte_count);
    // U layout: [16, K, C] -- strides for byte offset computation.
    const auto u_sh = u.get_shape().strides;

    const auto cblocks = (C + CB - 1) / CB;

    for(index_int cb = 0; cb < cblocks; ++cb)
    {
        const index_int c_base = cb * CB;

        // ---- Cooperative V load (input transform per (nt, c)) ----
        idx.local_stride(_c<BT * CB>, [&](auto task) {
            const index_int t_slot     = task % BT;
            const index_int c_in_block = task / BT;
            const index_int c          = c_base + c_in_block;
            const index_int nt         = t_base + t_slot;
            const bool active          = (c < C) and (nt < NT_total);

            const index_int n  = active ? nt / tiles_per_img : index_int{0};
            const auto rem     = active ? nt - n * tiles_per_img : index_int{0};
            const index_int th = active ? rem / tiles_w : index_int{0};
            const index_int tw = active ? rem - th * tiles_w : index_int{0};
            const int h0       = static_cast<int>(2 * th) - 1;
            const int w0       = static_cast<int>(2 * tw) - 1;

            const int32_t base_off =
                static_cast<int32_t>((n * x_sh[0] + c * x_sh[1]) * sizeof(half));
            const int32_t sh_b = static_cast<int32_t>(x_sh[2] * sizeof(half));
            const int32_t sw_b = static_cast<int32_t>(x_sh[3] * sizeof(half));
            const int32_t tile_off = base_off + h0 * sh_b + w0 * sw_b;
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
            repeat_c<16>([&](auto wp) {
                v_smem[v_cache_idx(wp, t_slot, c_in_block)] = V[wp];
            });
        });

        // ---- Cooperative U load: one b128 per task (8 fp16) ----
        static_assert(CB % 8 == 0, "CB must be a multiple of 8 for b128 U loads");
        constexpr index_int U_TASKS = 16 * BK * (CB / 8);
        idx.local_stride(_c<U_TASKS>, [&](auto task) {
            const index_int c_half     = task % (CB / 8);
            const index_int rest       = task / (CB / 8);
            const index_int k_in_block = rest % BK;
            const index_int wp         = rest / BK;
            const index_int c_in_block = c_half * 8;
            const index_int k          = k_base + k_in_block;
            const int32_t off          = static_cast<int32_t>(
                (wp * u_sh[0] + k * u_sh[1] + (c_base + c_in_block) * u_sh[2]) *
                sizeof(half));
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
            half* dst = &u_smem[u_cache_idx(wp, k_in_block, c_in_block)];
            *as_vec<8>(dst) = v8;
        });

        __syncthreads();

        // ---- WMMA with fused incremental output transform ----
        constexpr index_int wmma_chunks       = CB / 16;
        const index_int wave_nt_base_in_block = wave_id * BT_per_wave;
        const index_int m_in_wave             = lane % 16;
        const index_int c_off                 = (lane / 16) * 8;
        const index_int nt_slot               = wave_nt_base_in_block + m_in_wave;

        // Row-sum alpha fold: compute 4 wp (one full wp_i row of M) and apply
        // the A^T M A output transform via S0/S1 row sums.
        auto fold_row = [&](auto wp_i_val,
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
                        if constexpr(ki == 0) { v0 = s0.s0; v1 = s1.s0; }
                        else if constexpr(ki == 1) { v0 = s0.s1; v1 = s1.s1; }
                        else if constexpr(ki == 2) { v0 = s0.s2; v1 = s1.s2; }
                        else if constexpr(ki == 3) { v0 = s0.s3; v1 = s1.s3; }
                        else if constexpr(ki == 4) { v0 = s0.s4; v1 = s1.s4; }
                        else if constexpr(ki == 5) { v0 = s0.s5; v1 = s1.s5; }
                        else if constexpr(ki == 6) { v0 = s0.s6; v1 = s1.s6; }
                        else { v0 = s0.s7; v1 = s1.s7; }
                        y[r * 2 + 0][ki] = y[r * 2 + 0][ki] + coef_r * v0;
                        y[r * 2 + 1][ki] = y[r * 2 + 1][ki] + coef_r * v1;
                    });
                }
            });
        };
        // Helper: load 8 fp16 from V_lds for wp at (nt_slot, c_offset + c_off).
        auto load_v = [&](index_int wp, index_int c_offset) {
            return *as_vec<8>(
                &v_smem[v_cache_idx(wp, nt_slot, c_offset + c_off)]);
        };
        auto load_u = [&](index_int wp, index_int c_offset) {
            return *as_vec<8>(
                &u_smem[u_cache_idx(wp, m_in_wave, c_offset + c_off)]);
        };

        repeat_c<4>([&](auto wp_i_val) {
            constexpr int wp_i = wp_i_val;
            vec<float, 8> m0{}, m1{}, m2{}, m3{};
            if constexpr(wmma_chunks == 2)
            {
                // CB=32: issue both c-chunks' 8 WMMAs as one octet block to
                // give the matrix pipe a longer continuous run. The extra 4
                // accumulators only live across the octet; the compiler fuses
                // the post-octet add(mN, mN+4) into the fold.
                auto a0 = load_u(wp_i * 4 + 0, 0);
                auto b0 = load_v(wp_i * 4 + 0, 0);
                auto a1 = load_u(wp_i * 4 + 1, 0);
                auto b1 = load_v(wp_i * 4 + 1, 0);
                auto a2 = load_u(wp_i * 4 + 2, 0);
                auto b2 = load_v(wp_i * 4 + 2, 0);
                auto a3 = load_u(wp_i * 4 + 3, 0);
                auto b3 = load_v(wp_i * 4 + 3, 0);
                auto a4 = load_u(wp_i * 4 + 0, 16);
                auto b4 = load_v(wp_i * 4 + 0, 16);
                auto a5 = load_u(wp_i * 4 + 1, 16);
                auto b5 = load_v(wp_i * 4 + 1, 16);
                auto a6 = load_u(wp_i * 4 + 2, 16);
                auto b6 = load_v(wp_i * 4 + 2, 16);
                auto a7 = load_u(wp_i * 4 + 3, 16);
                auto b7 = load_v(wp_i * 4 + 3, 16);
                vec<float, 8> m4{}, m5{}, m6{}, m7{};
                wmma_octet_asm(a0, b0, a1, b1, a2, b2, a3, b3,
                               a4, b4, a5, b5, a6, b6, a7, b7,
                               m0, m1, m2, m3, m4, m5, m6, m7);
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
                    auto a0 = load_u(wp_i * 4 + 0, c_offset);
                    auto b0 = load_v(wp_i * 4 + 0, c_offset);
                    auto a1 = load_u(wp_i * 4 + 1, c_offset);
                    auto b1 = load_v(wp_i * 4 + 1, c_offset);
                    auto a2 = load_u(wp_i * 4 + 2, c_offset);
                    auto b2 = load_v(wp_i * 4 + 2, c_offset);
                    auto a3 = load_u(wp_i * 4 + 3, c_offset);
                    auto b3 = load_v(wp_i * 4 + 3, c_offset);
                    wmma_quad_asm(a0, b0, a1, b1, a2, b2, a3, b3,
                                  m0, m1, m2, m3);
                }
            }
            fold_row(_c<wp_i>, m0, m1, m2, m3);
        });

        __syncthreads();
    }

    // ---- Output writeback ----
    using out_type               = typename Output::type;
    const index_int nt_in_wave   = lane % 16;
    const index_int k_row_offset = (lane / 16) * 8;
    const index_int nt_global    = t_base + wave_id * BT_per_wave + nt_in_wave;
    if(nt_global >= NT_total)
        return;

    const index_int n  = nt_global / tiles_per_img;
    const auto rem     = nt_global - n * tiles_per_img;
    const index_int th = rem / tiles_w;
    const index_int tw = rem - th * tiles_w;

    const auto sn = out_shape.strides[0];
    const auto sk = out_shape.strides[1];
    const auto sh = out_shape.strides[2];
    const auto sw = out_shape.strides[3];
    auto* out_data = output.data();
    const index_int base_offset =
        n * sn + (k_base + k_row_offset) * sk + (2 * th) * sh + (2 * tw) * sw;

    repeat_c<8>([&](auto ki) {
        const index_int k = k_base + k_row_offset + ki;
        if(k < K)
        {
            const index_int kbase = base_offset + ki * sk;
            repeat_c<2>([&](auto i) {
                const int h_out = static_cast<int>(2 * th) + static_cast<int>(i);
                if(static_cast<unsigned>(h_out) < H_out)
                {
                    const index_int hbase = kbase + i * sh;
                    repeat_c<2>([&](auto j) {
                        const int w_out =
                            static_cast<int>(2 * tw) + static_cast<int>(j);
                        if(static_cast<unsigned>(w_out) < W_out)
                        {
                            out_data[hbase + j * sw] =
                                static_cast<out_type>(y[i * 2 + j][ki]);
                        }
                    });
                }
            });
        }
    });
}

} // namespace migraphx

#endif // MIGRAPHX_GUARD_KERNELS_WINOGRAD_CONV_HPP
