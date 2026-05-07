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
template <class T>
inline __device__ T dpp_perm_2211(T x)
{
    return dpp_quad_perm<0x55u | (2u << 4) | (2u << 6)>(x);
}
template <class T>
inline __device__ T dpp_perm_1111(T x)
{
    return dpp_quad_perm<0x55u>(x);
}
template <class T>
inline __device__ T dpp_perm_2222(T x)
{
    return dpp_quad_perm<0xAAu>(x);
}
template <class T>
inline __device__ T dpp_perm_3333(T x)
{
    return dpp_quad_perm<0xFFu>(x);
}
template <class T>
inline __device__ T dpp_perm_0001(T x)
{
    return dpp_quad_perm<0x40u>(x);
}
template <class T>
inline __device__ T dpp_perm_0021(T x)
{
    return dpp_quad_perm<0x60u>(x);
}
#endif

// row_shl:N via inline asm: lane[i] reads lane[i+N] within row of 16.
// MIOpen-exact `v_mov_b32 vDst, vSrc row_shl:N row_mask:0xf bank_mask:0xf`.
#define MIGRAPHX_WINOGRAD_ROW_SHL(NAME, N)                                     \
    template <class T>                                                         \
    inline __device__ T NAME(T x)                                              \
    {                                                                          \
        static_assert(sizeof(T) == 4, "row_shl only handles 32-bit operands"); \
        T y;                                                                   \
        asm("v_mov_b32 %[y], %[x] row_shl:" #N " row_mask:0xf bank_mask:0xf"   \
            : [y] "=v"(y)                                                      \
            : [x] "v"(x));                                                     \
        return y;                                                              \
    }

#if defined(__AMDGCN__)
MIGRAPHX_WINOGRAD_ROW_SHL(dpp_row_shl_4, 4)
MIGRAPHX_WINOGRAD_ROW_SHL(dpp_row_shl_8, 8)
MIGRAPHX_WINOGRAD_ROW_SHL(dpp_row_shl_12, 12)
#else
template <class T>
inline __device__ T dpp_row_shl_4(T x)
{
    return x;
}
template <class T>
inline __device__ T dpp_row_shl_8(T x)
{
    return x;
}
template <class T>
inline __device__ T dpp_row_shl_12(T x)
{
    return x;
}
#endif

// Full F(2x2, 3x3) inverse transform (A^T M A) computed in-wave, producing
// y00, y01, y10, y11 in the corner lane of each element-group. All DPP +
// arithmetic intermediates live in a single inline-asm block scoped to one
// (kk, tt) output iteration, so the compiler serialises VGPR usage and cannot
// hoist temporaries across iterations — crucial for keeping VGPR count down
// on fully-unrolled output loops.
//
// Transform:
//   Row stage: m_val → r0, r1 (3 row_shl, 2 sum, 2 sub)
//   Col stage: r0 → y00, y01; r1 → y10, y11 (6 quad_perm, 4 sum, 4 sub)
// 20 DPP+ALU ops total, 7 scratch VGPRs declared locally.
inline __device__ void
winograd_output_transform_f(float m_val, float& y00, float& y01, float& y10, float& y11)
{
#if defined(__AMDGCN__)
    float mk1, mk2, mk3, r0, r1, tq1, tq2, tq3;
    asm("v_mov_b32 %[mk1], %[mv] row_shl:4 row_mask:0xf bank_mask:0xf\n"
        "v_mov_b32 %[mk2], %[mv] row_shl:8 row_mask:0xf bank_mask:0xf\n"
        "v_mov_b32 %[mk3], %[mv] row_shl:12 row_mask:0xf bank_mask:0xf\n"
        "v_add_f32 %[r0], %[mv], %[mk1]\n"
        "v_add_f32 %[r0], %[r0], %[mk2]\n"
        "v_sub_f32 %[r1], %[mk1], %[mk2]\n"
        "v_sub_f32 %[r1], %[r1], %[mk3]\n"
        "v_mov_b32 %[tq1], %[r0] quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf\n"
        "v_mov_b32 %[tq2], %[r0] quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf\n"
        "v_mov_b32 %[tq3], %[r0] quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf\n"
        "v_add_f32 %[y00], %[r0], %[tq1]\n"
        "v_add_f32 %[y00], %[y00], %[tq2]\n"
        "v_sub_f32 %[y01], %[tq1], %[tq2]\n"
        "v_sub_f32 %[y01], %[y01], %[tq3]\n"
        "v_mov_b32 %[tq1], %[r1] quad_perm:[1,1,1,1] row_mask:0xf bank_mask:0xf\n"
        "v_mov_b32 %[tq2], %[r1] quad_perm:[2,2,2,2] row_mask:0xf bank_mask:0xf\n"
        "v_mov_b32 %[tq3], %[r1] quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf\n"
        "v_add_f32 %[y10], %[r1], %[tq1]\n"
        "v_add_f32 %[y10], %[y10], %[tq2]\n"
        "v_sub_f32 %[y11], %[tq1], %[tq2]\n"
        "v_sub_f32 %[y11], %[y11], %[tq3]\n"
        : [y00] "=&v"(y00),
          [y01] "=&v"(y01),
          [y10] "=&v"(y10),
          [y11] "=&v"(y11),
          [mk1] "=&v"(mk1),
          [mk2] "=&v"(mk2),
          [mk3] "=&v"(mk3),
          [r0] "=&v"(r0),
          [r1] "=&v"(r1),
          [tq1] "=&v"(tq1),
          [tq2] "=&v"(tq2),
          [tq3] "=&v"(tq3)
        : [mv] "v"(m_val));
#else
    (void)m_val;
    y00 = m_val;
    y01 = m_val;
    y10 = m_val;
    y11 = m_val;
#endif
}

// ----------------------------------------------------------------------------
//   Winograd F(2x2, 3x3) input transform - cooperative 4-lane DPP variant
// ----------------------------------------------------------------------------

// MIOpen-style 4-lane-per-tile DPP-cooperative input transform B^T * d * B
// for F(2x2, 3x3), packed across two channels A/B (low/high half for fp16,
// adjacent VGPRs for fp32).
//
// Layout assumption (input):
//   Each 4-lane quad cooperates on ONE 4x4 input tile.
//   Lane j ∈ {0, 1, 2, 3} within a quad holds COLUMN j of the input.
//   v[i] for i = 0..3 holds the 4 ROW values of column j (channels packed).
//
// Output (in-place):
//   v[i] holds V_alt[i][j] for the lane's column j, where V_alt has 6 sign
//   flips vs standard V = B^T*d*B at positions e ∈ {3, 7, 11, 12, 13, 14}.
//
// The matching offline U pretransform negates the same 6 positions, so the
// dot product U_alt · V_alt = U · V (standard), leaving the inverse
// transform unaffected.
//
// `sign` carries the per-lane v181 vector replicated across each channel:
// +1.0 for lane 1 (col j=1), -1.0 for lanes 0/2/3 (cols j=0/2/3).
//
// Algorithm (matches MIOpen exactly for fp16; fp32 mirrors the same shape):
//   Row stage: t = B^T * d, computed in-place across v[0..3]:
//     v0 = v0 - v2          # t[0] = d[0] - d[2]
//     v3 = v3 - v1          # = -t[3] (sign absorbed into v[3])
//     v2 = v2 - v1          # t[2] = d[2] - d[1]
//     v1 = 2*v1 + v2_new    # = v1 + v2_orig = t[1] (one FMA, MIOpen trick)
//   Col stage: cross-lane DPP quad_perm:[2,2,1,1] then FMA with sign s:
//     For each VGPR v[i], lane[k] reads lane[shuffled_k]:
//       lane 0 ← lane 2;  lane 1 ← lane 2;  lane 2 ← lane 1;  lane 3 ← lane 1
//     v[i] = v[i] + shuffled[v[i]] * s

// fp16 specialization: row stage as plain vector ops (compiler emits
// v_pk_add/sub_f16) followed by per-VGPR DPP shuffle + v_pk_fma_f16.
// Avoiding inline asm for the row stage lets the compiler interleave the
// individual ALU ops with surrounding GEMM dots.
inline __device__ void input_transform_packed_dpp(array<vec<half, 2>, 4>& v, vec<half, 2> sign)
{
    // Row stage (read original values, write back; compiler emits 4 v_pk_*).
    const vec<half, 2> v0o = v[0];
    const vec<half, 2> v1o = v[1];
    const vec<half, 2> v2o = v[2];
    const vec<half, 2> v3o = v[3];
    v[0]                   = v0o - v2o; // t[0]
    v[1]                   = v1o + v2o; // t[1]
    v[2]                   = v2o - v1o; // t[2]
    v[3]                   = v3o - v1o; // -t[3]
    // Col stage: per-VGPR DPP shuffle + FMA with per-lane sign.
    repeat_c<4>([&](auto i) {
        vec<half, 2> shuf = v[i];
        // 32-bit DPP shuffle on the packed half2 register.
        using U = uint32_t;
        U sh    = dpp_perm_2211(__builtin_bit_cast(U, shuf));
        shuf    = __builtin_bit_cast(vec<half, 2>, sh);
        v[i]    = shuf * sign + v[i];
    });
}

// fp32 specialization: row stage is plain scalar FMAs; col stage uses
// v_mov_b32 quad_perm:[2,2,1,1] DPP shuffle followed by v_fma_f32.
// Each channel is processed independently (no fp32 packed instruction).
//
// Implemented in C++ with explicit per-channel ops + inline-asm DPP shuffle
// helpers - clean enough that the compiler emits 8 v_fma_f32 row + 8 v_mov
// + 8 v_fma_f32 col = 24 ALU ops per cooperating lane.
inline __device__ void input_transform_packed_dpp(array<vec<float, 2>, 4>& v, vec<float, 2> sign)
{
    // Row stage: read original values then write back, no in-place hazard.
    const vec<float, 2> v0_orig = v[0];
    const vec<float, 2> v1_orig = v[1];
    const vec<float, 2> v2_orig = v[2];
    const vec<float, 2> v3_orig = v[3];
    v[0]                        = v0_orig - v2_orig; // t[0] = d[0] - d[2]
    v[1]                        = v1_orig + v2_orig; // t[1] = d[1] + d[2]
    v[2]                        = v2_orig - v1_orig; // t[2] = d[2] - d[1]
    v[3]                        = v3_orig - v1_orig; // -t[3] = d[3] - d[1]

    // Col stage: per-VGPR DPP shuffle + FMA with per-lane sign.
    repeat_c<4>([&](auto i) {
        const float shuf_a = dpp_perm_2211(v[i][0]);
        const float shuf_b = dpp_perm_2211(v[i][1]);
        v[i][0]            = v[i][0] + sign[0] * shuf_a;
        v[i][1]            = v[i][1] + sign[1] * shuf_b;
    });
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
        : [r00] "=&v"(r[0]),
          [r01] "=&v"(r[1]),
          [r02] "=&v"(r[2]),
          [r03] "=&v"(r[3]),
          [r10] "=&v"(r[4]),
          [r11] "=&v"(r[5]),
          [r12] "=&v"(r[6]),
          [r13] "=&v"(r[7])
        : [m00] "v"(m[0]),
          [m01] "v"(m[1]),
          [m02] "v"(m[2]),
          [m03] "v"(m[3]),
          [m10] "v"(m[4]),
          [m11] "v"(m[5]),
          [m12] "v"(m[6]),
          [m13] "v"(m[7]),
          [m20] "v"(m[8]),
          [m21] "v"(m[9]),
          [m22] "v"(m[10]),
          [m23] "v"(m[11]),
          [m30] "v"(m[12]),
          [m31] "v"(m[13]),
          [m32] "v"(m[14]),
          [m33] "v"(m[15]));
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
        : [r00] "v"(r[0]),
          [r01] "v"(r[1]),
          [r02] "v"(r[2]),
          [r03] "v"(r[3]),
          [r10] "v"(r[4]),
          [r11] "v"(r[5]),
          [r12] "v"(r[6]),
          [r13] "v"(r[7]));
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
        : [m00] "v"(mp[0]),
          [m01] "v"(mp[1]),
          [m10] "v"(mp[2]),
          [m11] "v"(mp[3]),
          [m20] "v"(mp[4]),
          [m21] "v"(mp[5]),
          [m30] "v"(mp[6]),
          [m31] "v"(mp[7]));
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
        y[i * 2u + 0u] =
            static_cast<half>(static_cast<float>(r[base + 0u]) + static_cast<float>(r[base + 1u]) +
                              static_cast<float>(r[base + 2u]));
        y[i * 2u + 1u] =
            static_cast<half>(static_cast<float>(r[base + 1u]) - static_cast<float>(r[base + 2u]) -
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
                d[ii * 4u + jj] = x[make_array<index_int>(n, c, index_int(hh), index_int(ww))];
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

// Read 16-element pre-transformed filter (W shape is [K, C, 4, 4], populated
// offline by prefuse_ops::pretransform_filter_literal with U = G * g * G^T).
// Single 16-byte / 32-byte memcpy for the common packed-stride case.
template <class T, class W>
__device__ __attribute__((const)) array<T, 16> load_filter_xformed(W w, index_int k, index_int c)
{
    constexpr auto ws = typename W::shape_type{};
    array<T, 16> u;
    if constexpr(ws.strides[2] == 4u and ws.strides[3] == 1u)
    {
        const T* base = w.data() + k * ws.strides[0] + c * ws.strides[1];
        __builtin_memcpy(u.data(), base, 16u * sizeof(T));
    }
    else
    {
        repeat_c<16>([&](auto e) {
            const auto i = e / 4u;
            const auto j = e % 4u;
            u[e]         = w[make_array<index_int>(k, c, i, j)];
        });
    }
    return u;
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
    using winograd::input_transform_packed_dpp;
    using winograd::load_filter_xformed;
    using winograd::set_prio;

    using out_type = typename Y::type;

    constexpr index_int N_ELEM          = 16u;
    constexpr index_int NGROUPS         = KT_DIV * TT_DIV;
    constexpr index_int BLOCK           = N_ELEM * NGROUPS;
    constexpr index_int K_BLOCK         = KT_DIV * KT;
    constexpr index_int T_BLOCK         = TT_DIV * TT_;
    constexpr index_int WAVE            = 64u;
    constexpr index_int GROUPS_PER_WAVE = WAVE / N_ELEM; // = 4
    constexpr index_int NWAVES          = BLOCK / WAVE;
    static_assert(BLOCK % WAVE == 0, "BLOCK must be a multiple of wave size (64)");
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
    const index_int wave_id       = local / WAVE;
    const index_int lane          = local % WAVE;
    const index_int group_in_wave = lane / N_ELEM; // 0..3
    const index_int my_e          = lane % N_ELEM; // 0..15 = element index
    const index_int pos           = wave_id * GROUPS_PER_WAVE + group_in_wave;
    const index_int my_k_div      = pos / TT_DIV;
    const index_int my_t_div      = pos % TT_DIV;

    // ---- LDS ring buffer for U/V staging.
    //
    // Layout has CH as INNERMOST dim, packing (ch_a, ch_b) at every (e, k)
    // slot.  For fp16 each slot is a vec<half,2> directly consumable by
    // v_dot2_f32_f16 - no v_perm_b32 unpacking needed between the LDS read
    // and the dot product.  For fp32 each slot is two adjacent floats used
    // by paired FMAs.  Mirrors MIOpen's hi/lo packed buffer-load pattern.
    //
    //   u_lds[ring][16 elements][K_PAD][CH]
    //   v_lds[ring][16 elements][T_PAD][CH]
    //
    // K_PAD = K_BLOCK + 1 (likewise T_PAD) so the per-element row size is
    // an odd multiple of the LDS bank period.  Without padding all 16 lanes
    // in an element-group would hit the same 4 banks during ds_load_b128 (a
    // 16-way conflict); the +1 stride spreads them across distinct banks.
    //
    // A single ds_load_b128 returns 16 bytes = 4 packed (k, ch) pairs for
    // fp16 (or 2 packed pairs for fp32), which load straight into the dot
    // operand registers.
    constexpr index_int CH    = 2u;
    constexpr index_int K_PAD = K_BLOCK + 1u;
    constexpr index_int T_PAD = T_BLOCK + 1u;
    constexpr auto u_shape    = make_shape(index_ints<RING, N_ELEM, K_PAD, CH>{});
    constexpr auto v_shape    = make_shape(index_ints<RING, N_ELEM, T_PAD, CH>{});
    constexpr index_int U_N   = RING * CH * N_ELEM * K_PAD;
    constexpr index_int V_N   = RING * CH * N_ELEM * T_PAD;

    __shared__ uninitialized_buffer<out_type, U_N> u_smem;
    __shared__ uninitialized_buffer<out_type, V_N> v_smem;
    auto u_lds = make_tensor_view(u_smem.data(), u_shape);
    auto v_lds = make_tensor_view(v_smem.data(), v_shape);

    // ---- Per-thread acc bank (one Winograd element, KT × TT_ k/tile positions)
    array<Acc, KT * TT_> acc{};

    constexpr diff_int pad = 1;

    // ---- Per-lane sign vector for the 4-lane DPP cooperative input transform.
    //
    // The col stage of the cooperative transform applies
    //   v[i]' = shuffled[v[i]] * sign + v[i]
    // with quad_perm:[2,2,1,1].  For the result to equal V_alt[i][j] (with 6
    // sign-flipped positions vs the canonical V), `sign` must hold:
    //   lane 0 in quad: -1   (j=0:  V[i][0] = t[i][0] - t[i][2])
    //   lane 1 in quad: +1   (j=1:  V[i][1] = t[i][1] + t[i][2])
    //   lane 2 in quad: -1   (j=2:  V[i][2] = t[i][2] - t[i][1])
    //   lane 3 in quad: -1   (j=3:  V[i][3] = -t[i][1] + t[i][3]; sign absorbed
    //                              into v[3] = -t[3] from the row stage)
    //
    // Both channel slots are set to the same value so the FMA applies the
    // same per-lane sign to both packed channels.
    vec<out_type, CH> in_xform_sign;
    {
        const out_type s = ((local & 3u) == 1u) ? out_type{1} : static_cast<out_type>(-1.0f);
        in_xform_sign[0] = s;
        in_xform_sign[1] = s;
    }

    auto stage = [&](index_int p, index_int slot) {
        const index_int c_a = p * 2u;
        const index_int c_b = c_a + 1u;

        // Filter is PRE-TRANSFORMED offline (prefuse_ops::pretransform_filter_literal):
        //   w shape is [K, C, 4, 4] containing U = G * g * G^T per (k, c).
        //
        // Distribute global loads UNIFORMLY across all BLOCK threads (matches
        // MIOpen's pattern of every-thread loading). Per channel pair the
        // total filter data is K_BLOCK * 16 * CH halves; each thread handles
        // one (kk, ch) filter chunk's slice of E_PER_THREAD elements.
        //
        // Layout: thread `local` covers position p = local within the linear
        // span of K_BLOCK*CH filter chunks, then loads E_PER_THREAD
        // contiguous transformed elements from that chunk.
        constexpr index_int FILT_CHUNKS  = K_BLOCK * CH; // 64 chunks
        constexpr index_int E_PER_THREAD = (16u * FILT_CHUNKS + BLOCK - 1u) / BLOCK;
        // For canonical (BLOCK=256, FILT_CHUNKS=64): E_PER_THREAD = 4 (one
        // chunk requires 16/E_PER_THREAD = 4 threads, total 256 threads).
        constexpr index_int THREADS_PER_CHUNK = (16u + E_PER_THREAD - 1u) / E_PER_THREAD;
        // chunk_id covers (kk, ch); slice_id covers element offset within chunk.
        const index_int chunk_id = local / THREADS_PER_CHUNK;
        const index_int slice_id = local % THREADS_PER_CHUNK;
        if(chunk_id < FILT_CHUNKS)
        {
            const index_int kk     = chunk_id / CH;
            const index_int ch_idx = chunk_id % CH;
            const index_int my_k = k_block * K_BLOCK + kk;
            const index_int c      = (ch_idx == 0u) ? c_a : c_b;
            const index_int e0     = slice_id * E_PER_THREAD;
            array<out_type, E_PER_THREAD> chunk{};
            if(my_k < K_ and c < C_ and e0 < 16u)
            {
                // Pre-transformed filter is [K, C, 4, 4] with packed strides.
                // Read E_PER_THREAD contiguous halves at (k, c, e0).
                constexpr auto ws = typename W::shape_type{};
                if constexpr(ws.strides[2] == 4u and ws.strides[3] == 1u)
                {
                    const out_type* base = w.data() + my_k * ws.strides[0] + c * ws.strides[1] + e0;
                    __builtin_memcpy(chunk.data(), base, sizeof(chunk));
                }
                else
                {
                    repeat_c<E_PER_THREAD>([&](auto i) {
                        const index_int e = e0 + i;
                        if(e < 16u)
                            chunk[i] = w[make_array<index_int>(my_k, c, e / 4u, e % 4u)];
                    });
                }
            }
            // Write to LDS: each thread's E_PER_THREAD halves go into its
            // (kk, ch) chunk at element positions [e0, e0+E_PER_THREAD).
            repeat_c<E_PER_THREAD>([&](auto i) {
                const index_int e = e0 + i;
                if(e < 16u)
                    u_lds[make_array<index_int>(slot, e, kk, ch_idx)] = chunk[i];
            });
        }

        // ---- 4-lane DPP-cooperative input transform (MIOpen-style).
        //
        // Lane decomposition (within the workgroup of BLOCK threads):
        //   col_in_quad = local & 3        - column j ∈ 0..3 within 4-lane quad
        //   quad_id     = local / 4        - quad index 0..(BLOCK/4 - 1)
        //
        // Each quad cooperates on ONE tile.  Lane j holds COLUMN j of the
        // 4x4 input, with 4 packed channel-pair values (one per row i=0..3).
        //
        // After the cooperative transform, lane j has v[i] = V_alt[i][j] for
        // i = 0..3, both channels packed.  Each lane writes 4 entries to LDS
        // at element positions e = i * 4 + j.
        //
        // BLOCK / 4 quads available; if T_BLOCK <= BLOCK/4, only the first
        // T_BLOCK quads are engaged for input staging.  Remaining lanes still
        // participate in the per-thread filter staging above.
        {
            const index_int col_in_quad = local & 3u;
            const index_int quad_id     = local >> 2u;
            if(quad_id < T_BLOCK)
            {
                const index_int tile_g = t_block * T_BLOCK + quad_id;
                array<vec<out_type, CH>, 4> v;
                repeat_c<4>([&](auto i) {
                    v[i][0] = out_type{};
                    v[i][1] = out_type{};
                });
                if(tile_g < total_)
                {
                    const index_int n_    = tile_g / t_pi;
                    const index_int t_img = tile_g % t_pi;
                    const index_int th    = t_img / t_w;
                    const index_int tw    = t_img % t_w;
                    const diff_int r0     = static_cast<diff_int>(th * 2u) - pad;
                    const diff_int c0     = static_cast<diff_int>(tw * 2u) - pad;
                    const diff_int ww     = c0 + static_cast<diff_int>(col_in_quad);
                    constexpr auto xs     = typename X::shape_type{};
                    constexpr auto H_in   = _c<index_int{xs.lens[2]}>;
                    constexpr auto W_in   = _c<index_int{xs.lens[3]}>;
                    const bool w_ok       = (ww >= 0 and ww < diff_int{W_in});
                    const bool a_ok       = (c_a < C_);
                    const bool b_ok       = (c_b < C_);
                    if(w_ok)
                    {
                        repeat_c<4>([&](auto i) {
                            const diff_int hh = r0 + diff_int{i};
                            if(hh >= 0 and hh < diff_int{H_in})
                            {
                                if(a_ok)
                                    v[i][0] = x[make_array<index_int>(
                                        n_, c_a, index_int(hh), index_int(ww))];
                                if(b_ok)
                                    v[i][1] = x[make_array<index_int>(
                                        n_, c_b, index_int(hh), index_int(ww))];
                            }
                        });
                    }
                }
                input_transform_packed_dpp(v, in_xform_sign);
                repeat_c<4>([&](auto i) {
                    const index_int e = i * 4u + col_in_quad;
                    __builtin_memcpy(
                        &v_lds[make_array<index_int>(slot, e, quad_id, 0u)], &v[i], sizeof(v[i]));
                });
            }
        }
    };

    // GEMM consume one channel pair from LDS slot.
    //
    // Operand layout:
    //   u_pair[k] = (u_a[k], u_b[k])   - packed CH-wide per K value for ITS
    //                                    element (e = my_e).
    //   v_pair[t] = (v_a[t], v_b[t])   - packed CH-wide per tile value.
    //
    // Because LDS has CH as innermost dim, each ds_load_b128 returns 16 bytes
    // worth of contiguous (k, ch) halves already in v_dot2_f32_f16 operand
    // order - no v_perm_b32 needed between the load and the dot.
    constexpr index_int B128_HALVES = 16u / sizeof(out_type); // 8 fp16, 4 fp32
    constexpr index_int B128_PAIRS  = B128_HALVES / CH;       // 4 fp16, 2 fp32
    static_assert(KT % B128_PAIRS == 0 or KT < B128_PAIRS,
                  "KT should be a multiple of b128 pair-chunk for full vectorization");
    static_assert(TT_ % B128_PAIRS == 0 or TT_ < B128_PAIRS,
                  "TT should be a multiple of b128 pair-chunk");

    // Helpers to load one LDS slot's worth of u_pair/v_pair into VGPRs.
    auto load_slot_u = [&](index_int slot, array<out_type, KT * CH>& u_pair) {
        if constexpr(KT >= B128_PAIRS)
        {
            constexpr index_int N_CHUNKS = KT / B128_PAIRS;
            using uvec                   = vec<out_type, B128_HALVES>;
            const uvec* u_v              = reinterpret_cast<const uvec*>(
                &u_lds[make_array<index_int>(slot, my_e, my_k_div * KT, 0u)]);
            uvec* u_dst = reinterpret_cast<uvec*>(u_pair.data());
            repeat_c<N_CHUNKS>([&](auto i) { u_dst[i] = u_v[i]; });
        }
        else
        {
            __builtin_memcpy(u_pair.data(),
                             &u_lds[make_array<index_int>(slot, my_e, my_k_div * KT, 0u)],
                             KT * CH * sizeof(out_type));
        }
    };
    auto load_slot_v = [&](index_int slot, array<out_type, TT_ * CH>& v_pair) {
        if constexpr(TT_ >= B128_PAIRS)
        {
            constexpr index_int N_CHUNKS = TT_ / B128_PAIRS;
            using vvec                   = vec<out_type, B128_HALVES>;
            const vvec* v_v              = reinterpret_cast<const vvec*>(
                &v_lds[make_array<index_int>(slot, my_e, my_t_div * TT_, 0u)]);
            vvec* v_dst = reinterpret_cast<vvec*>(v_pair.data());
            repeat_c<N_CHUNKS>([&](auto i) { v_dst[i] = v_v[i]; });
        }
        else
        {
            __builtin_memcpy(v_pair.data(),
                             &v_lds[make_array<index_int>(slot, my_e, my_t_div * TT_, 0u)],
                             TT_ * CH * sizeof(out_type));
        }
    };

    // Run KT*TT_ dots accumulating into acc, consuming u_pair/v_pair.
    //
    // MIOpen's GEMM layout (matching v4-v67 register order):
    //   T-outer / K-inner.  acc[t * KT + k].
    //   For each tile t (V[t] in v86-v93), broadcast across all K (U[0..7] in
    //   v70-v77).  This matches MIOpen's exact instruction stream:
    //     v_dot2_f32_f16 v4,  v70, v86, v4   # acc(t=0,k=0)
    //     v_dot2_f32_f16 v5,  v71, v86, v5   # acc(t=0,k=1)
    //     ...
    //     v_dot2_f32_f16 v11, v77, v86, v11  # acc(t=0,k=7)
    //     v_dot2_f32_f16 v12, v70, v87, v12  # acc(t=1,k=0)
    //     ...
    auto dot_block = [&](const array<out_type, KT * CH>& u_pair,
                         const array<out_type, TT_ * CH>& v_pair) {
        repeat_c<TT_>([&](auto nn) {
            repeat_c<KT>([&](auto m) {
                const auto ai = nn * KT + m;
                if constexpr(sizeof(out_type) == 2u)
                {
                    vec<half, 2> up;
                    vec<half, 2> vp;
                    up[0]   = u_pair[m * CH + 0u];
                    up[1]   = u_pair[m * CH + 1u];
                    vp[0]   = v_pair[nn * CH + 0u];
                    vp[1]   = v_pair[nn * CH + 1u];
                    acc[ai] = dot2_acc(up, vp, acc[ai]);
                }
                else
                {
                    acc[ai] = __builtin_fmaf(u_pair[m * CH + 0u], v_pair[nn * CH + 0u], acc[ai]);
                    acc[ai] = __builtin_fmaf(u_pair[m * CH + 1u], v_pair[nn * CH + 1u], acc[ai]);
                }
            });
        });
    };

    auto gemm = [&](index_int slot) {
        alignas(16) array<out_type, KT * CH> u_pair;  // interleaved (u_a, u_b)
        alignas(16) array<out_type, TT_ * CH> v_pair; // interleaved (v_a, v_b)
        load_slot_u(slot, u_pair);
        load_slot_v(slot, v_pair);
        // Barrier between LDS loads and dots - empirically the best perf:
        // without it the compiler sometimes hoists dots before the load
        // arrives, then waits on dscnt mid-stream creating bubbles.
        winograd::sched_barrier_full();
        dot_block(u_pair, v_pair);
    };

    // Software-pipelined channel loop. Mirrors MIOpen's pipeline:
    //
    //   pre:    issue buffer_loads for iter 0, transform + ds_store iter 0
    //   loop p in [0, n_pairs - 1):
    //     issue buffer_loads for iter p+1   <- hide global-load latency
    //                                          behind the dot-product chain
    //     gemm(p)                            <- 64+ v_dot / v_fma per thread
    //     transform + ds_store iter p+1     <- hide ALU+ds_store latency
    //                                          behind the next sync
    //     sync
    //   post:   gemm(n_pairs - 1)
    //
    // Wave priority stays at 1 from before the loop through the final dot
    // product, then drops back to 0 for the output transform.
    constexpr index_int n_pairs = (C_ + 1u) / 2u;
    set_prio<1>();
    if constexpr(RING == 1u)
    {
        for(index_int p = 0; p < n_pairs; ++p)
        {
            stage(p, 0u);
            __syncthreads();
            gemm(0u);
            __syncthreads();
        }
    }
    else
    {
        // RING >= 2: software-pipelined channel loop with gemm() FIRST in the
        // body so the compiler can interleave gemm's dot-product chain with
        // stage's input/filter transform ALU ops (they have no LDS aliasing
        // because gemm reads slot cur = p%RING and stage writes slot next =
        // (p+1)%RING). The single __syncthreads at end of each body protects
        // the next iter's gemm from this iter's stage writes.
        if constexpr(n_pairs > 0u)
        {
            stage(0u, 0u);
            __syncthreads();
            for(index_int p = 0; p + 1u < n_pairs; ++p)
            {
                const index_int cur  = p % RING;
                const index_int next = (p + 1u) % RING;
                gemm(cur);
                stage(p + 1u, next);
                __syncthreads();
            }
            gemm((n_pairs - 1u) % RING);
        }
    }
    set_prio<0>();

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
            const index_int ai = m_tt * KT + m_kk; // T outer / K inner
            Acc m_val          = acc[ai];

            // --- Row stage via row_shl (cross-quad within row-of-16).
            Acc m_k1 = dpp_row_shl_4(m_val);
            Acc m_k2 = dpp_row_shl_8(m_val);
            Acc m_k3 = dpp_row_shl_12(m_val);

            Acc r0 = m_val + m_k1 + m_k2; // lanes 0..3: R[0][j=lane]
            Acc r1 = m_k1 - m_k2 - m_k3;  // lanes 0..3: R[1][j=lane]

            // --- Col stage via quad_perm (within 4-lane quad).
            Acc r0_q1 = dpp_perm_1111(r0);
            Acc r0_q2 = dpp_perm_2222(r0);
            Acc r0_q3 = dpp_perm_3333(r0);
            Acc r1_q1 = dpp_perm_1111(r1);
            Acc r1_q2 = dpp_perm_2222(r1);
            Acc r1_q3 = dpp_perm_3333(r1);

            Acc y00 = r0 + r0_q1 + r0_q2;    // lane 0: Y[0][0]
            Acc y01 = r0_q1 - r0_q2 - r0_q3; // lane 0: Y[0][1]
            Acc y10 = r1 + r1_q1 + r1_q2;    // lane 0: Y[1][0]
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
            const bool h0_ok         = (base_h < H_out);
            const bool h1_ok         = (base_h + 1u < H_out);
            const bool w0_ok         = (base_w < W_out);
            const bool w1_ok         = (base_w + 1u < W_out);

            if constexpr(w_unit)
            {
                if(h0_ok and w0_ok and w1_ok)
                {
                    vec<out_type, 2> p;
                    p[0]          = static_cast<out_type>(y00);
                    p[1]          = static_cast<out_type>(y01);
                    out_type* dst = &y[make_array<index_int>(n_, my_k, base_h, base_w)];
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
                    p[0]          = static_cast<out_type>(y10);
                    p[1]          = static_cast<out_type>(y11);
                    out_type* dst = &y[make_array<index_int>(n_, my_k, base_h + 1u, base_w)];
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
                    y[make_array<index_int>(n_, my_k, base_h, base_w)] = static_cast<out_type>(y00);
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
