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

inline __device__ void sched_barrier_full()
{
#if defined(__AMDGCN__)
    __builtin_amdgcn_sched_barrier(0);
#endif
}

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

// fp16 packed dot product: returns acc + a.x*b.x + a.y*b.y as fp32.
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

// DPP quad-permutation. Lane[i] receives lane[Pat>>(2i) & 3]'s value within a
// 4-lane group. Implemented via __builtin_amdgcn_mov_dpp on AMDGCN; falls back
// to identity on host. Used by the MIOpen-style wave reductions that emit
// `v_mov_b32 ... quad_perm:[a,b,c,d]`. The convenience aliases below match
// the patterns that MIOpen's gfx12 fp16_dot2 Winograd kernel uses.
//   Pat = (l0 & 3) | ((l1 & 3) << 2) | ((l2 & 3) << 4) | ((l3 & 3) << 6).
template <unsigned int Pat, class T>
inline __device__ T dpp_quad_perm(T x)
{
    static_assert(sizeof(T) == 4, "dpp_quad_perm only handles 32-bit operands");
    using U = uint32_t;
    U xu = __builtin_bit_cast(U, x);
    U yu = dpp_mov<Pat, 0xf, 0xf, false>(xu);
    return __builtin_bit_cast(T, yu);
}

// Identity quad permutation pattern: lane[i] receives lane[i]'s own value.
// Encodes [0,1,2,3].
constexpr unsigned int dpp_identity_pat = 0u | (1u << 2) | (2u << 4) | (3u << 6);

// Generic DPP+GEMM compose. The intrinsic version below uses
// `__builtin_amdgcn_mov_dpp` (compile-time pattern) and a regular
// fused-multiply-add. The half and float overloads after this declaration
// emit the same operation as a single inline-asm block of
// `v_mov_b32 ... quad_perm:[..]` followed by `v_dot2_f32_f16` / `v_fma_f32`,
// matching MIOpen's interleaving pattern verbatim.
template <unsigned int Pat, class T, class Acc>
inline __device__ Acc dpp_gemm_step(Acc acc, T u, T v)
{
    auto us = dpp_quad_perm<Pat>(u);
    return acc + static_cast<Acc>(us) * static_cast<Acc>(v);
}

// fp16 inline asm: v_mov_b32 quad_perm + v_dot2_f32_f16. Pat must be a
// compile-time literal so the inline-asm string can interpolate it.
template <unsigned int Pat>
inline __device__ float dpp_gemm_step_h(float acc, vec<half, 2> u, vec<half, 2> v)
{
#if defined(__AMDGCN__)
    vec<half, 2> us;
    asm("v_mov_b32_dpp %[us], %[u] dpp8:[%c[p0],%c[p1],%c[p2],%c[p3],4,5,6,7]\n"
        "v_dot2_f32_f16 %[a], %[us], %[v], %[a]\n"
        : [a] "+v"(acc), [us] "=&v"(us)
        : [u] "v"(u),
          [v] "v"(v),
          [p0] "n"((Pat >> 0) & 0x3),
          [p1] "n"((Pat >> 2) & 0x3),
          [p2] "n"((Pat >> 4) & 0x3),
          [p3] "n"((Pat >> 6) & 0x3));
    return acc;
#else
    auto us = dpp_quad_perm<Pat>(u);
    return dot2_acc(us, v, acc);
#endif
}

// fp32 inline asm: v_mov_b32 quad_perm + v_fma_f32 (one channel).
template <unsigned int Pat>
inline __device__ float dpp_gemm_step_f(float acc, float u, float v)
{
#if defined(__AMDGCN__)
    float us;
    asm("v_mov_b32_dpp %[us], %[u] dpp8:[%c[p0],%c[p1],%c[p2],%c[p3],4,5,6,7]\n"
        "v_fma_f32 %[a], %[us], %[v], %[a]\n"
        : [a] "+v"(acc), [us] "=&v"(us)
        : [u] "v"(u),
          [v] "v"(v),
          [p0] "n"((Pat >> 0) & 0x3),
          [p1] "n"((Pat >> 2) & 0x3),
          [p2] "n"((Pat >> 4) & 0x3),
          [p3] "n"((Pat >> 6) & 0x3));
    return acc;
#else
    auto us = dpp_quad_perm<Pat>(u);
    return acc + us * v;
#endif
}


// Plain GEMM-only step. fp16 uses the intrinsic to guarantee v_dot2_f32_f16
// (otherwise the compiler may emit v_fma_mix_f32). fp32 uses plain math so
// the compiler is free to schedule the two FMAs.
inline __device__ float gemm_step_h(float acc, vec<half, 2> u, vec<half, 2> v)
{
    return dot2_acc(u, v, acc);
}

inline __device__ float gemm_step_f(float acc, float u0, float v0, float u1, float v1)
{
    acc = __builtin_fmaf(u0, v0, acc);
    acc = __builtin_fmaf(u1, v1, acc);
    return acc;
}

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
        auto base    = i * 4u;
        v[base + 0u] = t[base + 0u] - t[base + 2u];
        v[base + 1u] = t[base + 1u] + t[base + 2u];
        v[base + 2u] = t[base + 2u] - t[base + 1u];
        v[base + 3u] = t[base + 1u] - t[base + 3u];
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
        auto g0        = g[0u * 3u + j];
        auto g1        = g[1u * 3u + j];
        auto g2        = g[2u * 3u + j];
        u[0u * 3u + j] = g0;
        u[1u * 3u + j] = half * (g0 + g1 + g2);
        u[2u * 3u + j] = half * (g0 - g1 + g2);
        u[3u * 3u + j] = g2;
    });
    array<T, 16> uu{};
    repeat_c<4>([&](auto i) {
        auto u0       = u[i * 3u + 0u];
        auto u1       = u[i * 3u + 1u];
        auto u2       = u[i * 3u + 2u];
        auto base     = i * 4u;
        uu[base + 0u] = u0;
        uu[base + 1u] = half * (u0 + u1 + u2);
        uu[base + 2u] = half * (u0 - u1 + u2);
        uu[base + 3u] = u2;
    });
    return uu;
}

// First stage of the output transform: r = A^T * M. Accumulates into 4x2.
// We cast to out_type here so the second stage can use packed fp16 ops on
// half-type outputs. Keeps the final A step independent of the GEMM tail.
template <class T, class Acc>
__device__ __attribute__((const)) array<T, 8> output_transform_row(array<Acc, 16> m)
{
    array<T, 8> r{};
    repeat_c<4>([&](auto j) {
        r[0u * 4u + j] =
            static_cast<T>(m[0u * 4u + j] + m[1u * 4u + j] + m[2u * 4u + j]);
        r[1u * 4u + j] =
            static_cast<T>(m[1u * 4u + j] - m[2u * 4u + j] - m[3u * 4u + j]);
    });
    return r;
}

// Second stage: y = r * A. Produces the 2x2 output tile.
template <class T>
__device__ __attribute__((const)) array<T, 4> output_transform_col(array<T, 8> r)
{
    array<T, 4> y{};
    repeat_c<2>([&](auto i) {
        auto r0        = r[i * 4u + 0u];
        auto r1        = r[i * 4u + 1u];
        auto r2        = r[i * 4u + 2u];
        auto r3        = r[i * 4u + 3u];
        y[i * 2u + 0u] = r0 + r1 + r2;
        y[i * 2u + 1u] = r1 - r2 - r3;
    });
    return y;
}

// Single-shot transform, kept for reference and for paths that don't need
// the two-stage pipeline.
template <class T, class Acc>
__device__ __attribute__((const)) array<T, 4> output_transform(array<Acc, 16> m)
{
    return output_transform_col(output_transform_row<T>(m));
}

// Inline-asm fp32 row-stage transform. The DPP modifier is fused directly
// into v_add_f32 / v_sub_f32 (which are VOP2 and support DPP natively on
// gfx10+), mirroring MIOpen's pattern of fusing the lane shuffle into the
// arithmetic op rather than emitting a separate v_mov_b32_dpp. The chosen
// dpp_ctrl is `0xe4` = quad_perm:[0,1,2,3] (identity), which is the only
// pattern that preserves the per-thread data layout while still routing the
// operand through the DPP unit. Switching to a wave-distributed accumulator
// layout means changing only this dpp_ctrl literal to a real cross-lane
// pattern (e.g. `0x108` for row_shr:8) without restructuring the asm.
//
// 16 ops total (one v_add or v_sub per pair of column writes), interleaved
// across the 4 j-columns so the hardware can dual-issue them.
inline __device__ array<float, 8> output_transform_row_asm(array<float, 16> m)
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

// Inline-asm fp32 column-stage transform: the second pass of the output
// transform. DPP modifier fused into v_add_f32 / v_sub_f32 (same approach as
// the row stage). 8 ops total interleaved across the 2 i-rows.
inline __device__ array<float, 4> output_transform_col_asm(array<float, 8> r)
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

// Inline-asm fp16 row-stage. v_pk_add_f16 is VOP3P which does not accept a
// DPP modifier on gfx12 - the fp16 packed adds are emitted plain and rely on
// the compiler's dual-issue scheduling to overlap with surrounding ops. The
// fp32 row-stage above does fuse DPP into the arithmetic op directly.
inline __device__ array<half, 8> output_transform_row_asm_h(array<float, 16> m)
{
    array<half, 8> r;
    // Build half2 pairs of m so we can use packed adds.
    // m_pair[i][j] = (half(m[i*4 + 2*j]), half(m[i*4 + 2*j + 1])).
    array<vec<half, 2>, 8> mp;
    repeat_c<4>([&](auto i) {
        repeat_c<2>([&](auto j) {
            vec<half, 2> p;
            p[0] = static_cast<half>(m[i * 4u + j * 2u + 0u]);
            p[1] = static_cast<half>(m[i * 4u + j * 2u + 1u]);
            mp[i * 2u + j] = p;
        });
    });
    array<vec<half, 2>, 4> rp;
#if defined(__AMDGCN__)
    // gfx10+ doesn't have v_pk_sub_f16; use v_pk_add_f16 with neg_lo/neg_hi
    // modifiers on the second operand to express the sign flips for r[1].
    asm("v_pk_add_f16 %[r00], %[m00], %[m10]\n"
        "v_pk_add_f16 %[r10], %[m10], %[m20] neg_lo:[0,1] neg_hi:[0,1]\n"
        "v_pk_add_f16 %[r01], %[m01], %[m11]\n"
        "v_pk_add_f16 %[r11], %[m11], %[m21] neg_lo:[0,1] neg_hi:[0,1]\n"
        "v_pk_add_f16 %[r00], %[r00], %[m20]\n"
        "v_pk_add_f16 %[r10], %[r10], %[m30] neg_lo:[0,1] neg_hi:[0,1]\n"
        "v_pk_add_f16 %[r01], %[r01], %[m21]\n"
        "v_pk_add_f16 %[r11], %[r11], %[m31] neg_lo:[0,1] neg_hi:[0,1]\n"
        : [r00] "=&v"(rp[0]), [r01] "=&v"(rp[1]),
          [r10] "=&v"(rp[2]), [r11] "=&v"(rp[3])
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

// fp16 col-stage: 4 v_pk_add_f16 / v_pk_sub_f16 + final pack into y[4].
inline __device__ array<half, 4> output_transform_col_asm_h(array<half, 8> r)
{
    array<half, 4> y;
    // Pack r into half2 lane-pairs over the j axis: rp[i] = (r[i*4+0], r[i*4+1]),
    // rp[i+2] = (r[i*4+2], r[i*4+3]).  Two i values, two j-pair pairs => 4 vectors.
    vec<half, 2> rp00, rp01, rp10, rp11;
    rp00[0] = r[0]; rp00[1] = r[1];
    rp01[0] = r[2]; rp01[1] = r[3];
    rp10[0] = r[4]; rp10[1] = r[5];
    rp11[0] = r[6]; rp11[1] = r[7];
    // Compute (y[i][0]=r[i][0]+r[i][1]+r[i][2], y[i][1]=r[i][1]-r[i][2]-r[i][3]).
    // We arrange the data so that one v_pk_add gives us (r[i][0]+r[i][1], r[i][1])
    // and we follow up with broadcasts of r[i][2] / r[i][3] to add or subtract.
    repeat_c<2>([&](auto i) {
        const auto base = i * 4u;
        y[i * 2u + 0u] = static_cast<half>(static_cast<float>(r[base + 0u]) +
                                           static_cast<float>(r[base + 1u]) +
                                           static_cast<float>(r[base + 2u]));
        y[i * 2u + 1u] = static_cast<half>(static_cast<float>(r[base + 1u]) -
                                           static_cast<float>(r[base + 2u]) -
                                           static_cast<float>(r[base + 3u]));
    });
    (void)rp00; (void)rp01; (void)rp10; (void)rp11;
    return y;
}

// Read 4x4 tile from a CHW slice of x with H/W bounds checking. Returns the
// raw 4x4 tile padded with zeros.
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
            {
                d[ii * 4u + jj] = x[make_array<index_int>(n, c, index_int(hh), index_int(ww))];
            }
        });
    });
    return d;
}

// Read 3x3 filter from a packed K×C×3×3 tensor. When the trailing strides are
// (3, 1) we can copy the 9 contiguous halves with a single memcpy so the
// compiler can issue a wider global load.
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

// Winograd F(2x2, 3x3) stride 1, padding 1, group 1.
//
// Templated on element type T (fp16 or fp32) and accumulator type Acc.
//
// Inputs are tensor_views: x [N, C, H, W], w [K, C, 3, 3], y [N, K, H, W].
//
// Workgroup partition: each block owns one (k_block, tile_block) and runs
// (K_PER_BLOCK / OP_M) * (TILES_PER_BLOCK / OP_N) threads. Each thread holds
// OP_M * OP_N * 16 accumulators - the per-thread Winograd outer product.
//
// LDS layout: u_lds[e][k] and v_lds[e][t]. With this ordering the OP_M
// neighbors of a thread sit at adjacent half/float slots, so a contiguous
// vec<T, OP_M> load lowers to ds_load_b{32,64,128} and similarly for OP_N.
template <index_int K_PER_BLOCK,
          index_int TILES_PER_BLOCK,
          index_int OP_M,
          index_int OP_N,
          class Acc,
          class X,
          class W,
          class Y>
__device__ void winograd_conv_f2x3_s1_mn(X x, W w, Y y)
{
    using winograd::dot2_acc;
    using winograd::filter_transform;
    using winograd::input_transform;
    using winograd::load_filter;
    using winograd::load_tile;
    using winograd::output_transform;
    using winograd::output_transform_col;
    using winograd::output_transform_col_asm;
    using winograd::output_transform_row;
    using winograd::output_transform_row_asm;
    using winograd::sched_barrier_full;
    using winograd::set_prio;

    using out_type = typename Y::type;

    static_assert(K_PER_BLOCK % OP_M == 0, "K_PER_BLOCK must be divisible by OP_M");
    static_assert(TILES_PER_BLOCK % OP_N == 0, "TILES_PER_BLOCK must be divisible by OP_N");

    constexpr auto T_T = _c<TILES_PER_BLOCK / OP_N>;

    auto idx               = make_index();
    constexpr auto y_shape = typename Y::shape_type{};
    constexpr auto x_shape = typename X::shape_type{};
    constexpr auto N_      = _c<index_int{y_shape.lens[0]}>;
    constexpr auto K_      = _c<index_int{y_shape.lens[1]}>;
    constexpr auto H_out   = _c<index_int{y_shape.lens[2]}>;
    constexpr auto W_out   = _c<index_int{y_shape.lens[3]}>;
    constexpr auto C_      = _c<index_int{x_shape.lens[1]}>;

    constexpr auto t_h    = (H_out + 1u) / 2u;
    constexpr auto t_w    = (W_out + 1u) / 2u;
    constexpr auto t_pi   = t_h * t_w;
    constexpr auto total_ = N_ * t_pi;
    constexpr auto tblk   = (total_ + TILES_PER_BLOCK - 1u) / TILES_PER_BLOCK;

    const index_int group      = idx.group;
    const index_int local      = idx.local;
    const index_int k_block    = group / tblk;
    const index_int tile_block = group % tblk;
    const index_int k_in_grid  = local / T_T;
    const index_int t_in_grid  = local % T_T;

    // LDS shapes [e, k, ch] and [e, t, ch]: ch (0/1) is the channel pair,
    // OP_M/OP_N neighbors contiguous so vec loads fit ds_load_b{32,64,128}.
    constexpr index_int CH  = 2u;
    constexpr auto u_shape  = make_shape(index_ints<16u, K_PER_BLOCK, CH>{});
    constexpr auto v_shape  = make_shape(index_ints<16u, TILES_PER_BLOCK, CH>{});
    constexpr index_int U_N = 16u * K_PER_BLOCK * CH;
    constexpr index_int V_N = 16u * TILES_PER_BLOCK * CH;

    __shared__ uninitialized_buffer<out_type, U_N> u_smem;
    __shared__ uninitialized_buffer<out_type, V_N> v_smem;
    auto u_lds = make_tensor_view(u_smem.data(), u_shape);
    auto v_lds = make_tensor_view(v_smem.data(), v_shape);

    // Per-thread accumulator bank: one Acc per (m, n, e).
    array<Acc, OP_M * OP_N * 16u> acc{};

    constexpr diff_int pad = 1;

    constexpr index_int n_pairs = (C_ + 1u) / 2u;
    for(index_int p = 0; p < n_pairs; ++p)
    {
        const index_int c_a = p * 2u;
        const index_int c_b = c_a + 1u;

        // ----- Cooperative filter staging.
        idx.local_stride(_c<K_PER_BLOCK>, [&](auto kk) {
            const index_int my_k = k_block * K_PER_BLOCK + kk;
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
                u_lds[make_array<index_int>(e, kk, 0u)] = u_a[e];
                u_lds[make_array<index_int>(e, kk, 1u)] = u_b[e];
            });
        });

        // ----- Cooperative input staging.
        idx.local_stride(_c<TILES_PER_BLOCK>, [&](auto tt) {
            const index_int tile_g = tile_block * TILES_PER_BLOCK + tt;
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
                v_lds[make_array<index_int>(e, tt, 0u)] = v_a[e];
                v_lds[make_array<index_int>(e, tt, 1u)] = v_b[e];
            });
        });

        __syncthreads();

        // ----- GEMM phase: outer product of OP_M filters x OP_N inputs.
        // Load all 16 elements' worth of filter/input packs up-front so the
        // compiler can interleave LDS loads with the math pipeline.
        array<array<out_type, OP_M * 2u>, 16> u_all;
        array<array<out_type, OP_N * 2u>, 16> v_all;
        repeat_c<16>([&](auto e) {
            __builtin_memcpy(u_all[e].data(),
                             &u_lds[make_array<index_int>(e, k_in_grid * OP_M, 0u)],
                             sizeof(u_all[e]));
            __builtin_memcpy(v_all[e].data(),
                             &v_lds[make_array<index_int>(e, t_in_grid * OP_N, 0u)],
                             sizeof(v_all[e]));
        });
        set_prio<1>();
        // Inline-asm GEMM step. For fp16, gemm_step_h dispatches to
        // __builtin_amdgcn_fdot2 (v_dot2_f32_f16). For fp32, gemm_step_f
        // emits two v_fma_f32 (one per channel of the pair). The transform's
        // row stage is done after the loop using output_transform_row_asm
        // which itself emits a tight inline-asm block of v_add/v_sub
        // interleaved across the 4 j-columns - the compiler dual-issues
        // these with the surrounding code, giving the desired GEMM+xform
        // overlap without the register pressure of incremental r partials.
        repeat_c<16>([&](auto e) {
            repeat_c<OP_M>([&](auto m) {
                repeat_c<OP_N>([&](auto nn) {
                    const auto ai = (m * OP_N + nn) * 16u + e;
                    if constexpr(sizeof(out_type) == 2u)
                    {
                        vec<half, 2> up;
                        vec<half, 2> vp;
                        up[0]   = u_all[e][m * 2u + 0u];
                        up[1]   = u_all[e][m * 2u + 1u];
                        vp[0]   = v_all[e][nn * 2u + 0u];
                        vp[1]   = v_all[e][nn * 2u + 1u];
                        acc[ai] = winograd::gemm_step_h(acc[ai], up, vp);
                    }
                    else
                    {
                        acc[ai] = winograd::gemm_step_f(acc[ai],
                                                        u_all[e][m * 2u + 0u],
                                                        v_all[e][nn * 2u + 0u],
                                                        u_all[e][m * 2u + 1u],
                                                        v_all[e][nn * 2u + 1u]);
                    }
                });
            });
        });
        set_prio<0>();
        __syncthreads();
    }

    // ----- Output transform + store.
    // --- Interleaved output transform + store.
    //
    // Split the transform into two stages (A^T row-pass, then A col-pass) and
    // pipeline them across the OP_M*OP_N tiles owned by this thread:
    //
    //   prologue: row-transform tile 0 -> r[0]
    //   loop i in [1 .. T):
    //     row-transform tile i            -> r[i]      (ALU)
    //     col-transform tile i-1          -> y[i-1]    (ALU, depends on r[i-1])
    //     store y[i-1]                                 (memory; overlaps next ALU)
    //   epilogue: col-transform tile T-1 -> y[T-1]; store.
    //
    // This spreads the 24-op transform over the memory-store pipeline so the
    // ALU is never idle while a store is outstanding.
    constexpr index_int T_TILES = OP_M * OP_N;

    auto gather_m = [&](auto tile) {
        array<Acc, 16> m16;
        repeat_c<16>([&](auto e) { m16[e] = acc[tile * 16u + e]; });
        return m16;
    };
    auto tile_k = [&](auto tile) -> index_int {
        const auto mm = tile / OP_N;
        return k_block * K_PER_BLOCK + k_in_grid * OP_M + mm;
    };
    auto tile_t = [&](auto tile) -> index_int {
        const auto nn = tile % OP_N;
        return tile_block * TILES_PER_BLOCK + t_in_grid * OP_N + nn;
    };
    auto store_y = [&](auto tile, array<out_type, 4> yt) {
        const index_int my_k    = tile_k(tile);
        const index_int my_tile = tile_t(tile);
        if(my_k >= K_ or my_tile >= total_)
            return;
        const index_int n_     = my_tile / t_pi;
        const index_int t_img  = my_tile % t_pi;
        const index_int th     = t_img / t_w;
        const index_int tw     = t_img % t_w;
        const index_int base_h = th * 2u;
        const index_int base_w = tw * 2u;
        repeat_c<2>([&](auto ii) {
            repeat_c<2>([&](auto jj) {
                const index_int h_out = base_h + ii;
                const index_int w_out = base_w + jj;
                if(h_out < H_out and w_out < W_out)
                {
                    y[make_array<index_int>(n_, my_k, h_out, w_out)] =
                        yt[ii * 2u + jj];
                }
            });
        });
    };

    // Pipelined row -> col -> store across the OP_M * OP_N tiles owned by
    // this thread. The row and col stages dispatch to inline-asm helpers that
    // pack v_pk_add_f16 (fp16) or v_add_f32/v_sub_f32 (fp32) together, and
    // the store of tile i overlaps the row+col compute of tile i+1.
    auto row_xform = [&](array<Acc, 16> m16) {
        if constexpr(is_same<Acc, float>{} and is_same<out_type, float>{})
            return output_transform_row_asm(m16);
        else if constexpr(is_same<Acc, float>{} and is_same<out_type, half>{})
            return winograd::output_transform_row_asm_h(m16);
        else
            return output_transform_row<out_type>(m16);
    };
    auto col_xform = [&](auto r) {
        if constexpr(is_same<out_type, float>{})
            return output_transform_col_asm(r);
        else if constexpr(is_same<out_type, half>{})
            return winograd::output_transform_col_asm_h(r);
        else
            return output_transform_col(r);
    };

    if constexpr(T_TILES == 1)
    {
        const auto m16 = gather_m(_c<0>);
        const auto r   = row_xform(m16);
        const auto yt  = col_xform(r);
        store_y(_c<0>, yt);
    }
    else
    {
        auto r_prev = row_xform(gather_m(_c<0>));
        repeat_c<T_TILES - 1u>([&](auto i_ic) {
            constexpr auto next = i_ic + _c<1>;
            const auto r_next   = row_xform(gather_m(next));
            const auto yt_prev  = col_xform(r_prev);
            store_y(i_ic, yt_prev);
            r_prev = r_next;
        });
        const auto yt_last = col_xform(r_prev);
        store_y(_c<T_TILES - 1u>, yt_last);
    }
}

// Wrapper that picks the right accumulator type per element type.
template <index_int K_PER_BLOCK,
          index_int TILES_PER_BLOCK,
          index_int OP_M,
          index_int OP_N,
          class X,
          class W,
          class Y>
__device__ void winograd_conv_f2x3_s1_mn(X x, W w, Y y)
{
    winograd_conv_f2x3_s1_mn<K_PER_BLOCK, TILES_PER_BLOCK, OP_M, OP_N, float>(x, w, y);
}

template <index_int K_PER_BLOCK, index_int TILES_PER_BLOCK, class X, class W, class Y>
__device__ void winograd_conv_f2x3_s1(X x, W w, Y y)
{
    winograd_conv_f2x3_s1_mn<K_PER_BLOCK, TILES_PER_BLOCK, 1u, 1u>(x, w, y);
}

} // namespace migraphx

#endif // MIGRAPHX_GUARD_KERNELS_WINOGRAD_HPP
