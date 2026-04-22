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
#include <migraphx/kernels/tensor_view.hpp>
#include <migraphx/kernels/array.hpp>
#include <migraphx/kernels/vec.hpp>
#include <migraphx/kernels/functional.hpp>
#include <migraphx/kernels/integral_constant.hpp>
#include <migraphx/kernels/uninitialized_buffer.hpp>
#include <migraphx/kernels/dpp.hpp>
#include <migraphx/kernels/type_traits.hpp>

namespace migraphx {

namespace winograd {

// ----------------------------------------------------------------------------
//   Wavefront scheduling helpers (mirror MIOpen's gfx12 fp16_dot2 pattern)
// ----------------------------------------------------------------------------

template <int Prio>
inline __device__ void set_prio()
{
#if defined(__AMDGCN__)
    if constexpr(Prio == 1)
        asm volatile("s_setprio 1" ::: "memory");
    else
        asm volatile("s_setprio 0" ::: "memory");
#endif
}

inline __device__ void sched_barrier_full()
{
#if defined(__AMDGCN__)
    __builtin_amdgcn_sched_barrier(0);
#endif
}

// s_clause N: declare the next N+1 memory ops as a single hardware clause.
// MIOpen uses `s_clause 0x7` (= 8 ops) ahead of ds_load runs.
template <unsigned int N>
inline __device__ void s_clause()
{
#if defined(__AMDGCN__)
    static_assert(N >= 1 and N <= 64);
    asm volatile("s_clause %0" ::"n"(N - 1) : "memory");
#endif
}

// ----------------------------------------------------------------------------
//   Arithmetic primitives
// ----------------------------------------------------------------------------

// fp16 packed dot product: returns acc + a.x*b.x + a.y*b.y as fp32.
// Maps to v_dot2_f32_f16 on gfx10+/gfx11+/gfx12.
inline __device__ float dot2_acc(vec<half, 2> a, vec<half, 2> b, float acc)
{
#if defined(__gfx10__) || defined(__gfx11__) || defined(__gfx1100__) || defined(__gfx1101__) || \
    defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1200__) || defined(__gfx1201__)
    return __builtin_amdgcn_fdot2(a, b, acc, false);
#else
    return acc + static_cast<float>(a[0]) * static_cast<float>(b[0]) +
           static_cast<float>(a[1]) * static_cast<float>(b[1]);
#endif
}

// DPP quad-permutation. lane[i] receives lane[Pat>>(2i) & 3]'s value within a
// 4-lane group. MIOpen's `v_mov_b32 ... quad_perm:[a,b,c,d]` analogue.
template <unsigned int Pat, class T>
inline __device__ T dpp_quad_perm(T x)
{
    static_assert(sizeof(T) == 4, "dpp_quad_perm only handles 32-bit operands");
    using U = uint32_t;
    U xu    = __builtin_bit_cast(U, x);
    U yu    = dpp_mov<Pat, 0xf, 0xf, false>(xu);
    return __builtin_bit_cast(T, yu);
}

// MIOpen-exact `v_mov_b32 vDst, vSrc quad_perm:[a,b,c,d]` via inline asm.
// Encodes the lane permutation as ASCII so the assembler builds the DPP
// modifier byte-for-byte the way MIOpen does.
#define MIGRAPHX_WINOGRAD_QUAD_PERM(NAME, A, B, C, D)                            \
    template <class T>                                                           \
    inline __device__ T NAME(T x)                                                \
    {                                                                            \
        static_assert(sizeof(T) == 4, "quad_perm only handles 32-bit operands"); \
        T y;                                                                     \
        asm("v_mov_b32 %[y], %[x] quad_perm:[" #A "," #B "," #C "," #D           \
            "] row_mask:0xf bank_mask:0xf"                                       \
            : [y] "=v"(y)                                                        \
            : [x] "v"(x));                                                       \
        return y;                                                                \
    }

// The exact patterns MIOpen emits in its gfx12 fp16_dot2 Winograd asm.
#if defined(__AMDGCN__)
MIGRAPHX_WINOGRAD_QUAD_PERM(dpp_perm_2211, 2, 2, 1, 1)
MIGRAPHX_WINOGRAD_QUAD_PERM(dpp_perm_1111, 1, 1, 1, 1)
MIGRAPHX_WINOGRAD_QUAD_PERM(dpp_perm_2222, 2, 2, 2, 2)
MIGRAPHX_WINOGRAD_QUAD_PERM(dpp_perm_3333, 3, 3, 3, 3)
MIGRAPHX_WINOGRAD_QUAD_PERM(dpp_perm_0001, 0, 0, 0, 1)
MIGRAPHX_WINOGRAD_QUAD_PERM(dpp_perm_0021, 0, 0, 2, 1)
#else
template <class T> inline __device__ T dpp_perm_2211(T x) { return dpp_quad_perm<0x55u | (2u<<4) | (2u<<6)>(x); }
template <class T> inline __device__ T dpp_perm_1111(T x) { return dpp_quad_perm<0x55u>(x); }
template <class T> inline __device__ T dpp_perm_2222(T x) { return dpp_quad_perm<0xAAu>(x); }
template <class T> inline __device__ T dpp_perm_3333(T x) { return dpp_quad_perm<0xFFu>(x); }
template <class T> inline __device__ T dpp_perm_0001(T x) { return dpp_quad_perm<0x40u>(x); }
template <class T> inline __device__ T dpp_perm_0021(T x) { return dpp_quad_perm<0x60u>(x); }
#endif

// row_shl:N via inline asm: lane[i] reads lane[i+N] within row of 16.
// MIOpen-exact `v_mov_b32 vDst, vSrc row_shl:N row_mask:0xf bank_mask:0xf`.
#define MIGRAPHX_WINOGRAD_ROW_SHL(NAME, N)                                       \
    template <class T>                                                           \
    inline __device__ T NAME(T x)                                                \
    {                                                                            \
        static_assert(sizeof(T) == 4, "row_shl only handles 32-bit operands");   \
        T y;                                                                     \
        asm("v_mov_b32 %[y], %[x] row_shl:" #N " row_mask:0xf bank_mask:0xf"     \
            : [y] "=v"(y)                                                        \
            : [x] "v"(x));                                                       \
        return y;                                                                \
    }

#if defined(__AMDGCN__)
MIGRAPHX_WINOGRAD_ROW_SHL(dpp_row_shl_4, 4)
MIGRAPHX_WINOGRAD_ROW_SHL(dpp_row_shl_8, 8)
MIGRAPHX_WINOGRAD_ROW_SHL(dpp_row_shl_12, 12)
#else
template <class T> inline __device__ T dpp_row_shl_4(T x) { return x; }
template <class T> inline __device__ T dpp_row_shl_8(T x) { return x; }
template <class T> inline __device__ T dpp_row_shl_12(T x) { return x; }
#endif

// ----------------------------------------------------------------------------
//   Winograd F(2x2, 3x3) transforms
// ----------------------------------------------------------------------------

// B^T * d * B for F(2x2, 3x3). Returns the transformed 4x4 tile.
template <class T>
__device__ __attribute__((const)) array<T, 16> input_transform(array<T, 16> d)
{
    array<T, 16> t{};
    repeat_c<4>([&](auto j) {
        t[0u * 4u + j] = d[0u * 4u + j] - d[2u * 4u + j];
        t[1u * 4u + j] = d[1u * 4u + j] + d[2u * 4u + j];
        t[2u * 4u + j] = d[2u * 4u + j] - d[1u * 4u + j];
        t[3u * 4u + j] = d[1u * 4u + j] - d[3u * 4u + j];
    });
    array<T, 16> v{};
    repeat_c<4>([&](auto i) {
        const auto base = i * 4u;
        v[base + 0u]    = t[base + 0u] - t[base + 2u];
        v[base + 1u]    = t[base + 1u] + t[base + 2u];
        v[base + 2u]    = t[base + 2u] - t[base + 1u];
        v[base + 3u]    = t[base + 1u] - t[base + 3u];
    });
    return v;
}

// G * g * G^T for F(2x2, 3x3). Returns the 4x4 U tile.
template <class T>
__device__ __attribute__((const)) array<T, 16> filter_transform(array<T, 9> g)
{
    const auto half = T{0.5};
    array<T, 12> u{};
    repeat_c<3>([&](auto j) {
        const auto g0  = g[0u * 3u + j];
        const auto g1  = g[1u * 3u + j];
        const auto g2  = g[2u * 3u + j];
        u[0u * 3u + j] = g0;
        u[1u * 3u + j] = half * (g0 + g1 + g2);
        u[2u * 3u + j] = half * (g0 - g1 + g2);
        u[3u * 3u + j] = g2;
    });
    array<T, 16> uu{};
    repeat_c<4>([&](auto i) {
        const auto u0   = u[i * 3u + 0u];
        const auto u1   = u[i * 3u + 1u];
        const auto u2   = u[i * 3u + 2u];
        const auto base = i * 4u;
        uu[base + 0u]   = u0;
        uu[base + 1u]   = half * (u0 + u1 + u2);
        uu[base + 2u]   = half * (u0 - u1 + u2);
        uu[base + 3u]   = u2;
    });
    return uu;
}

// Inline-asm fp32 row-stage transform (A^T * M). DPP fused into v_add/v_sub.
inline __device__ array<float, 8> output_transform_row_asm_f(array<float, 16> m)
{
    array<float, 8> r;
#if defined(__AMDGCN__)
    asm("v_add_f32 %[r00], %[m00], %[m10]\n"
        "v_sub_f32 %[r10], %[m10], %[m20] quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf\n"
        "v_add_f32 %[r01], %[m01], %[m11]\n"
        "v_sub_f32 %[r11], %[m11], %[m21] quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf\n"
        "v_add_f32 %[r02], %[m02], %[m12]\n"
        "v_sub_f32 %[r12], %[m12], %[m22] quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf\n"
        "v_add_f32 %[r03], %[m03], %[m13]\n"
        "v_sub_f32 %[r13], %[m13], %[m23] quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf\n"
        "v_add_f32 %[r00], %[r00], %[m20] quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf\n"
        "v_sub_f32 %[r10], %[r10], %[m30]\n"
        "v_add_f32 %[r01], %[r01], %[m21] quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf\n"
        "v_sub_f32 %[r11], %[r11], %[m31]\n"
        "v_add_f32 %[r02], %[r02], %[m22] quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf\n"
        "v_sub_f32 %[r12], %[r12], %[m32]\n"
        "v_add_f32 %[r03], %[r03], %[m23] quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf\n"
        "v_sub_f32 %[r13], %[r13], %[m33]\n"
        : [r00] "=&v"(r[0]), [r01] "=&v"(r[1]), [r02] "=&v"(r[2]), [r03] "=&v"(r[3]),
          [r10] "=&v"(r[4]), [r11] "=&v"(r[5]), [r12] "=&v"(r[6]), [r13] "=&v"(r[7])
        : [m00] "v"(m[0]), [m01] "v"(m[1]), [m02] "v"(m[2]), [m03] "v"(m[3]),
          [m10] "v"(m[4]), [m11] "v"(m[5]), [m12] "v"(m[6]), [m13] "v"(m[7]),
          [m20] "v"(m[8]), [m21] "v"(m[9]), [m22] "v"(m[10]), [m23] "v"(m[11]),
          [m30] "v"(m[12]), [m31] "v"(m[13]), [m32] "v"(m[14]), [m33] "v"(m[15]));
#else
    repeat_c<4>([&](auto j) {
        r[0u * 4u + j] = m[0u * 4u + j] + m[1u * 4u + j] + m[2u * 4u + j];
        r[1u * 4u + j] = m[1u * 4u + j] - m[2u * 4u + j] - m[3u * 4u + j];
    });
#endif
    return r;
}

// Inline-asm fp32 column-stage transform (R * A). DPP fused into v_add/v_sub.
inline __device__ array<float, 4> output_transform_col_asm_f(array<float, 8> r)
{
    array<float, 4> y;
#if defined(__AMDGCN__)
    asm("v_add_f32 %[y00], %[r00], %[r01]\n"
        "v_sub_f32 %[y01], %[r01], %[r02] quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf\n"
        "v_add_f32 %[y10], %[r10], %[r11]\n"
        "v_sub_f32 %[y11], %[r11], %[r12] quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf\n"
        "v_add_f32 %[y00], %[y00], %[r02] quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf\n"
        "v_sub_f32 %[y01], %[y01], %[r03]\n"
        "v_add_f32 %[y10], %[y10], %[r12] quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf\n"
        "v_sub_f32 %[y11], %[y11], %[r13]\n"
        : [y00] "=&v"(y[0]), [y01] "=&v"(y[1]), [y10] "=&v"(y[2]), [y11] "=&v"(y[3])
        : [r00] "v"(r[0]), [r01] "v"(r[1]), [r02] "v"(r[2]), [r03] "v"(r[3]),
          [r10] "v"(r[4]), [r11] "v"(r[5]), [r12] "v"(r[6]), [r13] "v"(r[7]));
#else
    repeat_c<2>([&](auto i) {
        y[i * 2u + 0u] = r[i * 4u + 0u] + r[i * 4u + 1u] + r[i * 4u + 2u];
        y[i * 2u + 1u] = r[i * 4u + 1u] - r[i * 4u + 2u] - r[i * 4u + 3u];
    });
#endif
    return y;
}

// fp16 row-stage with v_pk_add_f16 (VOP3P, no DPP modifier on gfx12).
inline __device__ array<half, 8> output_transform_row_asm_h(array<float, 16> m)
{
    array<half, 8> r;
    array<vec<half, 2>, 8> mp;
    repeat_c<4>([&](auto i) {
        repeat_c<2>([&](auto j) {
            vec<half, 2> p;
            p[0]           = static_cast<half>(m[i * 4u + j * 2u + 0u]);
            p[1]           = static_cast<half>(m[i * 4u + j * 2u + 1u]);
            mp[i * 2u + j] = p;
        });
    });
    array<vec<half, 2>, 4> rp;
#if defined(__AMDGCN__)
    asm("v_pk_add_f16 %[r00], %[m00], %[m10]\n"
        "v_pk_add_f16 %[r10], %[m10], %[m20] neg_lo:[0,1] neg_hi:[0,1]\n"
        "v_pk_add_f16 %[r01], %[m01], %[m11]\n"
        "v_pk_add_f16 %[r11], %[m11], %[m21] neg_lo:[0,1] neg_hi:[0,1]\n"
        "v_pk_add_f16 %[r00], %[r00], %[m20]\n"
        "v_pk_add_f16 %[r10], %[r10], %[m30] neg_lo:[0,1] neg_hi:[0,1]\n"
        "v_pk_add_f16 %[r01], %[r01], %[m21]\n"
        "v_pk_add_f16 %[r11], %[r11], %[m31] neg_lo:[0,1] neg_hi:[0,1]\n"
        : [r00] "=&v"(rp[0]), [r01] "=&v"(rp[1]), [r10] "=&v"(rp[2]), [r11] "=&v"(rp[3])
        : [m00] "v"(mp[0]), [m01] "v"(mp[1]),
          [m10] "v"(mp[2]), [m11] "v"(mp[3]),
          [m20] "v"(mp[4]), [m21] "v"(mp[5]),
          [m30] "v"(mp[6]), [m31] "v"(mp[7]));
#else
    repeat_c<2>([&](auto j) {
        rp[0u * 2u + j] = mp[0u * 2u + j] + mp[1u * 2u + j] + mp[2u * 2u + j];
        rp[1u * 2u + j] = mp[1u * 2u + j] - mp[2u * 2u + j] - mp[3u * 2u + j];
    });
#endif
    repeat_c<2>([&](auto i) {
        repeat_c<2>([&](auto j) {
            r[i * 4u + j * 2u + 0u] = rp[i * 2u + j][0];
            r[i * 4u + j * 2u + 1u] = rp[i * 2u + j][1];
        });
    });
    return r;
}

// fp16 col-stage. Generic; produces 2x2 output tile.
inline __device__ array<half, 4> output_transform_col_h(array<half, 8> r)
{
    array<half, 4> y;
    repeat_c<2>([&](auto i) {
        const auto base = i * 4u;
        y[i * 2u + 0u]  = static_cast<half>(static_cast<float>(r[base + 0u]) +
                                           static_cast<float>(r[base + 1u]) +
                                           static_cast<float>(r[base + 2u]));
        y[i * 2u + 1u]  = static_cast<half>(static_cast<float>(r[base + 1u]) -
                                           static_cast<float>(r[base + 2u]) -
                                           static_cast<float>(r[base + 3u]));
    });
    return y;
}

// ----------------------------------------------------------------------------
//   Loaders
// ----------------------------------------------------------------------------

// Read 4x4 input tile, padded with zeros for OOB positions.
template <class T, class X>
__device__ __attribute__((const)) array<T, 16>
load_tile(X x, index_int n, index_int c, diff_int r0, diff_int c0)
{
    constexpr auto xs   = typename X::shape_type{};
    constexpr auto H_in = _c<index_int{xs.lens[2]}>;
    constexpr auto W_in = _c<index_int{xs.lens[3]}>;
    array<T, 16> d{};
    repeat_c<4>([&](auto ii) {
        const diff_int hh = r0 + diff_int{ii};
        const bool h_ok   = (hh >= 0 and hh < diff_int{H_in});
        repeat_c<4>([&](auto jj) {
            const diff_int ww = c0 + diff_int{jj};
            const bool w_ok   = (ww >= 0 and ww < diff_int{W_in});
            if(h_ok and w_ok)
                d[ii * 4u + jj] =
                    x[make_array<index_int>(n, c, index_int(hh), index_int(ww))];
        });
    });
    return d;
}

// Read 3x3 filter (KCRS layout). Packed strides → single memcpy.
template <class T, class W>
__device__ __attribute__((const)) array<T, 9> load_filter(W w, index_int k, index_int c)
{
    constexpr auto ws = typename W::shape_type{};
    array<T, 9> g{};
    if constexpr(ws.strides[2] == 3u and ws.strides[3] == 1u)
    {
        const T* base = w.data() + k * ws.strides[0] + c * ws.strides[1];
        __builtin_memcpy(g.data(), base, 9u * sizeof(T));
    }
    else
    {
        repeat_c<9>([&](auto rs) {
            const auto r = rs / 3u;
            const auto s = rs % 3u;
            g[rs]        = w[make_array<index_int>(k, c, r, s)];
        });
    }
    return g;
}

} // namespace winograd

// ============================================================================
//   Wave-distributed Winograd F(2x2, 3x3) kernel  (MIOpen-aligned)
// ============================================================================
//
// MIOpen's gfx12 fp16_dot2 Winograd kernel distributes the 16 Winograd
// elements (e = 0..15) across 16 LANES within a wave: lanes 0..15 hold
// elements 0..15 for ONE (k_thr, t_thr) sub-block.  The output transform is
// then a within-wave reduction using v_pk_fma_f16 + DPP shuffles - no LDS
// round-trip.
//
// Layout we mirror:
//
//   workgroup_size = NWAVES * 64  (must be a multiple of 64)
//   element-group  = 16 lanes within a wave (4 per wave)
//   one element-group owns ONE (k_thr, t_thr) sub-block of size KT × TT_
//
//   Lane decomposition (within the wavefront of 64 lanes):
//     wave_id   = local / 64
//     lane      = local % 64
//     group_id_in_wave = lane / 16          - 0..3 (4 groups per wave)
//     in_group  = lane % 16                 - 0..15 = Winograd element index e
//
//   Each lane:
//     - holds element index e = in_group
//     - covers KT × TT_ (k, tile) outputs for the element-group's sub-block
//
//   Workgroup-level (wave_id, group_id_in_wave) → (k_thr, t_thr):
//     pos = wave_id * 4 + group_id_in_wave  (0..NGROUPS-1)
//     k_thr = pos / TT_DIV
//     t_thr = pos % TT_DIV
//   where NGROUPS = NWAVES * 4 = KT_DIV * TT_DIV.
//
//   Block tile:
//     K_BLOCK = KT_DIV * KT
//     T_BLOCK = TT_DIV * TT_
//
// LDS layout (per ring slot, fp16 / fp32):
//   u_lds[16][K_BLOCK][2]  - transformed filter U  (e × K × ch_pair)
//   v_lds[16][T_BLOCK][2]  - transformed input  V  (e × tile × ch_pair)
//
// GEMM:
//   Each lane reads U and V for ITS element only (e = in_group).  All 16
//   lanes in a group cover the 16 elements for the same (k_thr, t_thr).
//   Per channel pair, accumulates KT * TT_ values via v_dot2_f32_f16.
//
// Output transform (in-wave, no LDS exchange):
//   16 lanes hold m[0..15] for (kk, tt).  Combine via DPP:
//     - lane = j * 4 + i  →  4 lanes per quad cover rows i=0..3 of column j
//     - Within-quad DPP (quad_perm) does row-stage A^T M
//     - Cross-quad DPP (row_shr) does col-stage R A
//   Output 2x2 tile is held by lanes (j=0..1, i=0..1) per quad pair.
//
// Tuning (matches MIOpen's canonical config):
//   KT_DIV=4, TT_DIV=4, KT=8, TT_=8  → 256-thread block, K_BLOCK=T_BLOCK=32,
//                                     64 fp32 acc/thread.
template <index_int KT_DIV,
          index_int TT_DIV,
          index_int KT,
          index_int TT_,
          index_int RING,
          class Acc,
          class X,
          class W,
          class Y>
__device__ void winograd_conv_f2x3_s1_kernel(X x, W w, Y y)
{
    using winograd::dot2_acc;
    using winograd::filter_transform;
    using winograd::input_transform;
    using winograd::load_filter;
    using winograd::load_tile;
    using winograd::set_prio;

    using out_type = typename Y::type;

    constexpr index_int N_ELEM    = 16u;
    constexpr index_int NGROUPS   = KT_DIV * TT_DIV;
    constexpr index_int BLOCK     = N_ELEM * NGROUPS;
    constexpr index_int K_BLOCK   = KT_DIV * KT;
    constexpr index_int T_BLOCK   = TT_DIV * TT_;
    constexpr index_int WAVE      = 64u;
    constexpr index_int GROUPS_PER_WAVE = WAVE / N_ELEM; // = 4
    constexpr index_int NWAVES    = BLOCK / WAVE;
    static_assert(BLOCK % WAVE == 0,
                  "BLOCK must be a multiple of wave size (64)");
    static_assert(NGROUPS == NWAVES * GROUPS_PER_WAVE,
                  "NGROUPS must = NWAVES * 4 for in-wave DPP output transform");
    (void)NWAVES;

    // ---- Problem dims
    auto idx               = make_index();
    constexpr auto y_shape = typename Y::shape_type{};
    constexpr auto x_shape = typename X::shape_type{};
    constexpr auto N_      = _c<index_int{y_shape.lens[0]}>;
    constexpr auto K_      = _c<index_int{y_shape.lens[1]}>;
    constexpr auto H_out   = _c<index_int{y_shape.lens[2]}>;
    constexpr auto W_out   = _c<index_int{y_shape.lens[3]}>;
    constexpr auto C_      = _c<index_int{x_shape.lens[1]}>;

    constexpr auto t_h     = (H_out + 1u) / 2u;
    constexpr auto t_w     = (W_out + 1u) / 2u;
    constexpr auto t_pi    = t_h * t_w;
    constexpr auto total_  = N_ * t_pi;
    constexpr auto tblocks = (total_ + T_BLOCK - 1u) / T_BLOCK;

    // ---- This block's coordinates
    const index_int group   = idx.group;
    const index_int local   = idx.local;
    const index_int k_block = group / tblocks;
    const index_int t_block = group % tblocks;

    // Lane decomposition: 16 LANES within a wave hold the 16 Winograd
    // elements for ONE (k_thr, t_thr) sub-block.  4 such groups per wave.
    const index_int wave_id          = local / WAVE;
    const index_int lane             = local % WAVE;
    const index_int group_in_wave    = lane / N_ELEM;       // 0..3
    const index_int my_e             = lane % N_ELEM;       // 0..15 = element index
    const index_int pos              = wave_id * GROUPS_PER_WAVE + group_in_wave;
    const index_int my_k_div         = pos / TT_DIV;
    const index_int my_t_div         = pos % TT_DIV;

    // ---- LDS ring buffer for U/V staging.
    //
    // Layout has CH as OUTER dimension so that within one channel slab the
    // K (or tile) dimension is contiguous in memory.  Loading KT consecutive
    // K values for one element + one channel becomes a single ds_load_b128
    // (16 bytes for KT=8 fp16, or KT=4 fp32) instead of fragmented narrow
    // loads.  We then issue CH such loads (one per channel of the pair).
    //
    //   u_lds[ring][ch][16 elements][K_BLOCK]
    //   v_lds[ring][ch][16 elements][T_BLOCK]
    constexpr index_int CH = 2u;
    constexpr auto u_shape = make_shape(index_ints<RING, CH, N_ELEM, K_BLOCK>{});
    constexpr auto v_shape = make_shape(index_ints<RING, CH, N_ELEM, T_BLOCK>{});
    constexpr index_int U_N = RING * CH * N_ELEM * K_BLOCK;
    constexpr index_int V_N = RING * CH * N_ELEM * T_BLOCK;

    __shared__ uninitialized_buffer<out_type, U_N> u_smem;
    __shared__ uninitialized_buffer<out_type, V_N> v_smem;
    auto u_lds = make_tensor_view(u_smem.data(), u_shape);
    auto v_lds = make_tensor_view(v_smem.data(), v_shape);

    // ---- Per-thread acc bank (one Winograd element, KT × TT_ k/tile positions)
    array<Acc, KT * TT_> acc{};

    constexpr diff_int pad = 1;

    auto stage = [&](index_int p, index_int slot) {
        const index_int c_a = p * 2u;
        const index_int c_b = c_a + 1u;

        idx.local_stride(_c<K_BLOCK>, [&](auto kk) {
            const index_int my_k = k_block * K_BLOCK + kk;
            array<out_type, 9> g_a{};
            array<out_type, 9> g_b{};
            if(my_k < K_)
            {
                if(c_a < C_)
                    g_a = load_filter<out_type>(w, my_k, c_a);
                if(c_b < C_)
                    g_b = load_filter<out_type>(w, my_k, c_b);
            }
            const auto u_a = filter_transform(g_a);
            const auto u_b = filter_transform(g_b);
            repeat_c<16>([&](auto e) {
                u_lds[make_array<index_int>(slot, 0u, e, kk)] = u_a[e];
                u_lds[make_array<index_int>(slot, 1u, e, kk)] = u_b[e];
            });
        });

        idx.local_stride(_c<T_BLOCK>, [&](auto tt) {
            const index_int tile_g = t_block * T_BLOCK + tt;
            array<out_type, 16> d_a{};
            array<out_type, 16> d_b{};
            if(tile_g < total_)
            {
                const index_int n_    = tile_g / t_pi;
                const index_int t_img = tile_g % t_pi;
                const index_int th    = t_img / t_w;
                const index_int tw    = t_img % t_w;
                const diff_int r0     = static_cast<diff_int>(th * 2u) - pad;
                const diff_int c0     = static_cast<diff_int>(tw * 2u) - pad;
                if(c_a < C_)
                    d_a = load_tile<out_type>(x, n_, c_a, r0, c0);
                if(c_b < C_)
                    d_b = load_tile<out_type>(x, n_, c_b, r0, c0);
            }
            const auto v_a = input_transform(d_a);
            const auto v_b = input_transform(d_b);
            repeat_c<16>([&](auto e) {
                v_lds[make_array<index_int>(slot, 0u, e, tt)] = v_a[e];
                v_lds[make_array<index_int>(slot, 1u, e, tt)] = v_b[e];
            });
        });
    };

    // GEMM consume one channel pair from LDS slot.
    // With CH-outer LDS layout, KT contiguous K values for one (e, ch) are a
    // single ds_load_b128 (16 bytes for KT=4 fp32 or KT=8 fp16).  We force
    // the compiler to emit b128 by reading via vec<out_type, N> chunks.
    constexpr index_int B128_HALVES = 16u / sizeof(out_type);  // 8 fp16, 4 fp32
    static_assert(KT % B128_HALVES == 0 or KT < B128_HALVES,
                  "KT should be a multiple of b128 chunk for full vectorization");
    static_assert(TT_ % B128_HALVES == 0 or TT_ < B128_HALVES,
                  "TT should be a multiple of b128 chunk");

    auto gemm = [&](index_int slot) {
        alignas(16) array<out_type, KT> u_a;
        alignas(16) array<out_type, KT> u_b;
        alignas(16) array<out_type, TT_> v_a;
        alignas(16) array<out_type, TT_> v_b;

        // Vector-load via vec<out_type, B128_HALVES> chunks. This forces the
        // compiler's LDS load lowering to emit ds_load_b128.
        if constexpr(KT >= B128_HALVES)
        {
            constexpr index_int N_CHUNKS = KT / B128_HALVES;
            using uvec = vec<out_type, B128_HALVES>;
            const uvec* u_a_v =
                reinterpret_cast<const uvec*>(
                    &u_lds[make_array<index_int>(slot, 0u, my_e, my_k_div * KT)]);
            const uvec* u_b_v =
                reinterpret_cast<const uvec*>(
                    &u_lds[make_array<index_int>(slot, 1u, my_e, my_k_div * KT)]);
            uvec* u_a_dst = reinterpret_cast<uvec*>(u_a.data());
            uvec* u_b_dst = reinterpret_cast<uvec*>(u_b.data());
            repeat_c<N_CHUNKS>([&](auto i) {
                u_a_dst[i] = u_a_v[i];
                u_b_dst[i] = u_b_v[i];
            });
        }
        else
        {
            __builtin_memcpy(
                u_a.data(),
                &u_lds[make_array<index_int>(slot, 0u, my_e, my_k_div * KT)],
                KT * sizeof(out_type));
            __builtin_memcpy(
                u_b.data(),
                &u_lds[make_array<index_int>(slot, 1u, my_e, my_k_div * KT)],
                KT * sizeof(out_type));
        }
        if constexpr(TT_ >= B128_HALVES)
        {
            constexpr index_int N_CHUNKS = TT_ / B128_HALVES;
            using vvec = vec<out_type, B128_HALVES>;
            const vvec* v_a_v =
                reinterpret_cast<const vvec*>(
                    &v_lds[make_array<index_int>(slot, 0u, my_e, my_t_div * TT_)]);
            const vvec* v_b_v =
                reinterpret_cast<const vvec*>(
                    &v_lds[make_array<index_int>(slot, 1u, my_e, my_t_div * TT_)]);
            vvec* v_a_dst = reinterpret_cast<vvec*>(v_a.data());
            vvec* v_b_dst = reinterpret_cast<vvec*>(v_b.data());
            repeat_c<N_CHUNKS>([&](auto i) {
                v_a_dst[i] = v_a_v[i];
                v_b_dst[i] = v_b_v[i];
            });
        }
        else
        {
            __builtin_memcpy(
                v_a.data(),
                &v_lds[make_array<index_int>(slot, 0u, my_e, my_t_div * TT_)],
                TT_ * sizeof(out_type));
            __builtin_memcpy(
                v_b.data(),
                &v_lds[make_array<index_int>(slot, 1u, my_e, my_t_div * TT_)],
                TT_ * sizeof(out_type));
        }

        repeat_c<KT>([&](auto m) {
            repeat_c<TT_>([&](auto nn) {
                const auto ai = m * TT_ + nn;
                if constexpr(sizeof(out_type) == 2u)
                {
                    vec<half, 2> up;
                    vec<half, 2> vp;
                    up[0]   = u_a[m];
                    up[1]   = u_b[m];
                    vp[0]   = v_a[nn];
                    vp[1]   = v_b[nn];
                    acc[ai] = dot2_acc(up, vp, acc[ai]);
                }
                else
                {
                    acc[ai] = __builtin_fmaf(u_a[m], v_a[nn], acc[ai]);
                    acc[ai] = __builtin_fmaf(u_b[m], v_b[nn], acc[ai]);
                }
            });
        });
    };

    // Software-pipelined channel loop.
    constexpr index_int n_pairs = (C_ + 1u) / 2u;
    if constexpr(RING == 1u)
    {
        for(index_int p = 0; p < n_pairs; ++p)
        {
            stage(p, 0u);
            __syncthreads();
            set_prio<1>();
            gemm(0u);
            set_prio<0>();
            __syncthreads();
        }
    }
    else
    {
        if constexpr(n_pairs > 0u)
        {
            stage(0u, 0u);
            __syncthreads();
            set_prio<1>();
            for(index_int p = 0; p + 1u < n_pairs; ++p)
            {
                const index_int cur  = p % RING;
                const index_int next = (p + 1u) % RING;
                stage(p + 1u, next);
                gemm(cur);
                __syncthreads();
            }
            gemm((n_pairs - 1u) % RING);
            set_prio<0>();
        }
    }

    // ---- In-wave DPP-based output transform (MIOpen-style, no LDS exchange).
    //
    // Accumulator layout after GEMM:
    //   Each lane holds KT × TT_ acc[kk, tt] values for ITS element index
    //   my_e = lane % 16.  Filter_transform produces u_a[e] = U[e/4][e%4],
    //   so lane-to-element mapping is:
    //       i = my_e / 4   (Winograd row)
    //       j = my_e % 4   (Winograd column)
    //   Within each element-group of 16 lanes:
    //       lanes  0.. 3 = m[0][0..3]  (row i=0, cols j=0..3)
    //       lanes  4.. 7 = m[1][0..3]
    //       lanes  8..11 = m[2][0..3]
    //       lanes 12..15 = m[3][0..3]
    //
    // Row stage (sum/diff across i for fixed j) → CROSS-QUAD DPP (row_shl).
    //   r0 = own + row_shl:4(own) + row_shl:8(own)
    //       - lane 0: m[0][0] + m[1][0] + m[2][0] = R[0][0]  ✓
    //       - lane 1: m[0][1] + m[1][1] + m[2][1] = R[0][1]  ✓
    //       - lanes 0..3 hold R[0][j=0..3]  (other lanes garbage/unused)
    //   r1 = row_shl:4(own) - row_shl:8(own) - row_shl:12(own)
    //       - lane 0: m[1][0] - m[2][0] - m[3][0] = R[1][0]  ✓
    //       - lanes 0..3 hold R[1][j=0..3]
    //
    // Col stage (sum/diff across j for fixed i) → WITHIN-QUAD DPP (quad_perm).
    //   Applied to r0 (gives Y[0][*]) and to r1 (gives Y[1][*]).
    //   y0 = r + quad_perm[1,1,1,1](r) + quad_perm[2,2,2,2](r)
    //       - lane 0: R[i][0] + R[i][1] + R[i][2] = Y[i][0]  ✓
    //   y1 = quad_perm[1,1,1,1](r) - quad_perm[2,2,2,2](r) - quad_perm[3,3,3,3](r)
    //       - lane 0: R[i][1] - R[i][2] - R[i][3] = Y[i][1]  ✓
    //
    // Writer: lane 0 of each element-group (my_e == 0) writes the full 2×2
    // output tile {Y[0][0], Y[0][1], Y[1][0], Y[1][1]}.
    using winograd::dpp_perm_1111;
    using winograd::dpp_perm_2222;
    using winograd::dpp_perm_3333;
    using winograd::dpp_row_shl_12;
    using winograd::dpp_row_shl_4;
    using winograd::dpp_row_shl_8;

    const bool is_writer = (my_e == 0u);

    const index_int k_base_k    = k_block * K_BLOCK + my_k_div * KT;
    const index_int t_base_tile = t_block * T_BLOCK + my_t_div * TT_;

    repeat_c<KT>([&](auto m_kk) {
        repeat_c<TT_>([&](auto m_tt) {
            const index_int ai = m_kk * TT_ + m_tt;
            Acc m_val          = acc[ai];

            // --- Row stage via row_shl (cross-quad within row-of-16).
            Acc m_k1 = dpp_row_shl_4(m_val);
            Acc m_k2 = dpp_row_shl_8(m_val);
            Acc m_k3 = dpp_row_shl_12(m_val);

            Acc r0 = m_val + m_k1 + m_k2;   // lanes 0..3: R[0][j=lane]
            Acc r1 = m_k1 - m_k2 - m_k3;    // lanes 0..3: R[1][j=lane]

            // --- Col stage via quad_perm (within 4-lane quad).
            // For Y[0][*] apply to r0; for Y[1][*] apply to r1.
            Acc r0_q1 = dpp_perm_1111(r0);
            Acc r0_q2 = dpp_perm_2222(r0);
            Acc r0_q3 = dpp_perm_3333(r0);
            Acc r1_q1 = dpp_perm_1111(r1);
            Acc r1_q2 = dpp_perm_2222(r1);
            Acc r1_q3 = dpp_perm_3333(r1);

            Acc y00 = r0 + r0_q1 + r0_q2;   // lane 0: Y[0][0]
            Acc y01 = r0_q1 - r0_q2 - r0_q3; // lane 0: Y[0][1]
            Acc y10 = r1 + r1_q1 + r1_q2;   // lane 0: Y[1][0]
            Acc y11 = r1_q1 - r1_q2 - r1_q3; // lane 0: Y[1][1]

            if(not is_writer)
                return;
            const index_int my_k    = k_base_k + m_kk;
            const index_int my_tile = t_base_tile + m_tt;
            if(my_k >= K_ or my_tile >= total_)
                return;
            const index_int n_     = my_tile / t_pi;
            const index_int t_img  = my_tile % t_pi;
            const index_int th     = t_img / t_w;
            const index_int tw     = t_img % t_w;
            const index_int base_h = th * 2u;
            const index_int base_w = tw * 2u;

            // Write 2x2 output tile.  When the output W stride is 1 and both
            // (base_w, base_w+1) are in-bounds, the (y00, y01) pair sits at
            // adjacent bytes — pack into a vec<out_type, 2> store so the
            // compiler emits global_store_b64 (instead of two b32 stores).
            constexpr auto y_strides = typename Y::shape_type{}.strides;
            constexpr bool w_unit    = (y_strides[3] == 1u);
            const bool h0_ok = (base_h < H_out);
            const bool h1_ok = (base_h + 1u < H_out);
            const bool w0_ok = (base_w < W_out);
            const bool w1_ok = (base_w + 1u < W_out);

            if constexpr(w_unit)
            {
                if(h0_ok and w0_ok and w1_ok)
                {
                    vec<out_type, 2> p;
                    p[0] = static_cast<out_type>(y00);
                    p[1] = static_cast<out_type>(y01);
                    out_type* dst =
                        &y[make_array<index_int>(n_, my_k, base_h, base_w)];
                    __builtin_memcpy(dst, &p, sizeof(p));
                }
                else
                {
                    if(h0_ok and w0_ok)
                        y[make_array<index_int>(n_, my_k, base_h, base_w)] =
                            static_cast<out_type>(y00);
                    if(h0_ok and w1_ok)
                        y[make_array<index_int>(n_, my_k, base_h, base_w + 1u)] =
                            static_cast<out_type>(y01);
                }
                if(h1_ok and w0_ok and w1_ok)
                {
                    vec<out_type, 2> p;
                    p[0] = static_cast<out_type>(y10);
                    p[1] = static_cast<out_type>(y11);
                    out_type* dst =
                        &y[make_array<index_int>(n_, my_k, base_h + 1u, base_w)];
                    __builtin_memcpy(dst, &p, sizeof(p));
                }
                else
                {
                    if(h1_ok and w0_ok)
                        y[make_array<index_int>(n_, my_k, base_h + 1u, base_w)] =
                            static_cast<out_type>(y10);
                    if(h1_ok and w1_ok)
                        y[make_array<index_int>(n_, my_k, base_h + 1u, base_w + 1u)] =
                            static_cast<out_type>(y11);
                }
            }
            else
            {
                if(h0_ok and w0_ok)
                    y[make_array<index_int>(n_, my_k, base_h, base_w)] =
                        static_cast<out_type>(y00);
                if(h0_ok and w1_ok)
                    y[make_array<index_int>(n_, my_k, base_h, base_w + 1u)] =
                        static_cast<out_type>(y01);
                if(h1_ok and w0_ok)
                    y[make_array<index_int>(n_, my_k, base_h + 1u, base_w)] =
                        static_cast<out_type>(y10);
                if(h1_ok and w1_ok)
                    y[make_array<index_int>(n_, my_k, base_h + 1u, base_w + 1u)] =
                        static_cast<out_type>(y11);
            }
        });
    });
}


// fp accumulator dispatch wrapper for wave kernel.
template <index_int KT_DIV,
          index_int TT_DIV,
          index_int KT,
          index_int TT_,
          index_int RING,
          class X,
          class W,
          class Y>
__device__ void winograd_conv_f2x3_s1_kernel(X x, W w, Y y)
{
    winograd_conv_f2x3_s1_kernel<KT_DIV, TT_DIV, KT, TT_, RING, float>(x, w, y);
}

} // namespace migraphx

#endif // MIGRAPHX_GUARD_KERNELS_WINOGRAD_HPP
