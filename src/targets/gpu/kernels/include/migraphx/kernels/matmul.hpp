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

#ifndef MIGRAPHX_GUARD_KERNELS_MATMUL_HPP
#define MIGRAPHX_GUARD_KERNELS_MATMUL_HPP

#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/shape.hpp>
#include <migraphx/kernels/vec.hpp>
#include <migraphx/kernels/types.hpp>
#include <migraphx/kernels/type_traits.hpp>
#include <migraphx/kernels/bit_cast.hpp>
#include <migraphx/kernels/array.hpp>
#include <migraphx/kernels/functional.hpp>
#include <migraphx/kernels/math.hpp>

namespace migraphx {
namespace matmul_detail {

constexpr index_int lanes = MIGRAPHX_WAVEFRONTSIZE;

// The lane's fragment of a BxB accumulator
template <class Acc, index_int B>
using acc_frag = vec<Acc, B * B / lanes>;

// One matrix instruction, D[M][N] += A[M][K] B[K][N] accumulating in Acc:
// specialized for every instruction the architecture has, so a shape is
// supported exactly when the specialization is complete
template <class Acc, class T, index_int M, index_int N, index_int K>
struct mma_intrinsic;

template <class Acc, class T, index_int M, index_int N, index_int K, class = void>
struct has_mma_intrinsic : false_type
{
};

template <class Acc, class T, index_int M, index_int N, index_int K>
struct has_mma_intrinsic<Acc, T, M, N, K, void_t<decltype(sizeof(mma_intrinsic<Acc, T, M, N, K>))>>
    : true_type
{
};

// NOLINTNEXTLINE
#define MIGRAPHX_MMA_INTRINSIC(acc, type, m, n, k, ...)        \
    template <>                                                \
    struct mma_intrinsic<acc, type, m, n, k>                   \
    {                                                          \
        template <class Frag, class Acc>                       \
        __device__ Acc operator()(Frag a, Frag b, Acc c) const \
        {                                                      \
            return __VA_ARGS__;                                \
        }                                                      \
    };

// The lane layout of the architecture's instructions:
//   frag_k<B, KI>()     the k elements a lane holds of A and of B
//   frag_k0(kpl, lg)    the first of them for lane group lg
//   frag_row<B>(r, lg)  the row of accumulator element r held by lane group lg
// Every layout gives lane l row l % B of A and column l % B of B and D
#if defined(__GFX12__) and MIGRAPHX_WAVEFRONTSIZE == 32
// RDNA4: WMMA 16x16x16, each lane half holds 8 k
MIGRAPHX_MMA_INTRINSIC(
    float, half, 16, 16, 16, __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c))
MIGRAPHX_MMA_INTRINSIC(
    float, bf16, 16, 16, 16, __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c))

template <index_int B, index_int KI>
constexpr index_int frag_k()
{
    return KI * B / lanes;
}

__device__ constexpr index_int frag_k0(index_int kpl, index_int lg) { return kpl * lg; }

template <index_int B>
__device__ constexpr index_int frag_row(index_int r, index_int lg)
{
    return 8 * lg + r;
}
#elif defined(__GFX11__) and MIGRAPHX_WAVEFRONTSIZE == 32
// RDNA3: WMMA 16x16x16, both lane halves hold all 16 k, D rows interleave
MIGRAPHX_MMA_INTRINSIC(float, half, 16, 16, 16, __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c))
MIGRAPHX_MMA_INTRINSIC(float,
                       bf16,
                       16,
                       16,
                       16,
                       __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(bit_cast<short>(a),
                                                                   bit_cast<short>(b),
                                                                   c))

template <index_int B, index_int KI>
constexpr index_int frag_k()
{
    return KI;
}

__device__ constexpr index_int frag_k0(index_int, index_int) { return 0; }

template <index_int B>
__device__ constexpr index_int frag_row(index_int r, index_int lg)
{
    return 2 * r + lg;
}
#elif (defined(__gfx908__) or defined(__gfx90a__) or defined(__gfx940__) or defined(__gfx941__) or \
       defined(__gfx942__) or defined(__gfx950__)) and                                             \
    MIGRAPHX_WAVEFRONTSIZE == 64
// CDNA: MFMA 16x16 and 32x32 blocks, each lane quarter (half) holds its
// share of k; bf16 is 16x16x16/32x32x8 from gfx90a on, half of that on
// gfx908, and gfx950 doubles the f16 and bf16 k
MIGRAPHX_MMA_INTRINSIC(
    float, float, 16, 16, 4, __builtin_amdgcn_mfma_f32_16x16x4f32(a, b, c, 0, 0, 0))
MIGRAPHX_MMA_INTRINSIC(
    float, float, 32, 32, 2, __builtin_amdgcn_mfma_f32_32x32x2f32(a, b, c, 0, 0, 0))
MIGRAPHX_MMA_INTRINSIC(
    float, half, 16, 16, 16, __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, c, 0, 0, 0))
MIGRAPHX_MMA_INTRINSIC(
    float, half, 32, 32, 8, __builtin_amdgcn_mfma_f32_32x32x8f16(a, b, c, 0, 0, 0))
#if defined(__gfx908__)
MIGRAPHX_MMA_INTRINSIC(
    float,
    bf16,
    16,
    16,
    8,
    __builtin_amdgcn_mfma_f32_16x16x8bf16(bit_cast<short>(a), bit_cast<short>(b), c, 0, 0, 0))
MIGRAPHX_MMA_INTRINSIC(
    float,
    bf16,
    32,
    32,
    4,
    __builtin_amdgcn_mfma_f32_32x32x4bf16(bit_cast<short>(a), bit_cast<short>(b), c, 0, 0, 0))
#else
MIGRAPHX_MMA_INTRINSIC(
    float,
    bf16,
    16,
    16,
    16,
    __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(bit_cast<short>(a), bit_cast<short>(b), c, 0, 0, 0))
MIGRAPHX_MMA_INTRINSIC(
    float,
    bf16,
    32,
    32,
    8,
    __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(bit_cast<short>(a), bit_cast<short>(b), c, 0, 0, 0))
#endif
#if defined(__gfx950__)
MIGRAPHX_MMA_INTRINSIC(
    float, half, 16, 16, 32, __builtin_amdgcn_mfma_f32_16x16x32_f16(a, b, c, 0, 0, 0))
MIGRAPHX_MMA_INTRINSIC(
    float, half, 32, 32, 16, __builtin_amdgcn_mfma_f32_32x32x16_f16(a, b, c, 0, 0, 0))
MIGRAPHX_MMA_INTRINSIC(
    float, bf16, 16, 16, 32, __builtin_amdgcn_mfma_f32_16x16x32_bf16(a, b, c, 0, 0, 0))
MIGRAPHX_MMA_INTRINSIC(
    float, bf16, 32, 32, 16, __builtin_amdgcn_mfma_f32_32x32x16_bf16(a, b, c, 0, 0, 0))
#endif

template <index_int B, index_int KI>
constexpr index_int frag_k()
{
    return KI * B / lanes;
}

__device__ constexpr index_int frag_k0(index_int kpl, index_int lg) { return kpl * lg; }

// each lane group holds 4 consecutive rows in turn
template <index_int B>
__device__ constexpr index_int frag_row(index_int r, index_int lg)
{
    return (r / 4) * (4 * lanes / B) + 4 * lg + r % 4;
}
#else
// no matrix instruction: matmul takes the FMA path and never instantiates these
template <index_int B, index_int KI>
constexpr index_int frag_k()
{
    return 1;
}

__device__ constexpr index_int frag_k0(index_int, index_int) { return 0; }

template <index_int B>
__device__ constexpr index_int frag_row(index_int r, index_int)
{
    return r;
}
#endif

// The largest instruction k of a BxB block of T accumulating in Acc, at most
// KI, that divides K; 0 when there is none
template <class Acc, class T, index_int B, index_int K, index_int KI = 64>
constexpr index_int mma_k()
{
    if constexpr(KI == 0)
        return 0;
    else if constexpr(has_mma_intrinsic<Acc, T, B, B, KI>{} and K % KI == 0)
        return KI;
    else
        return mma_k<Acc, T, B, K, KI / 2>();
}

// acc += A[i0.., k0..] B[k0.., j0..] for one BxB block and one instruction
// of KI k
template <class Acc, class T, index_int B, index_int KI, class A, class Bl>
__device__ void mma_block(
    A a, Bl b, index_int i0, index_int j0, index_int k0, acc_frag<Acc, B>& acc, index_int lane)
{
    static_assert(B * B % lanes == 0, "the accumulator must split evenly over the wave");
    static_assert(KI * B % lanes == 0, "k must split evenly over the lane groups");
    constexpr index_int kpl = frag_k<B, KI>();
    using frag              = conditional_t<(kpl > 1), vec<T, kpl>, T>;
    const index_int li      = lane % B;
    const index_int k       = k0 + frag_k0(kpl, lane / B);
    frag av{};
    frag bv{};
    if constexpr(kpl > 1)
    {
        for(index_int r = 0; r < kpl; r++)
        {
            av[r] = a(i0 + li, k + r);
            bv[r] = b(k + r, j0 + li);
        }
    }
    else
    {
        av = a(i0 + li, k);
        bv = b(k, j0 + li);
    }
    acc = mma_intrinsic<Acc, T, B, B, KI>{}(av, bv, acc);
}

// c(x, i, j) for the lane's fragment of block (i0, j0)
template <class Acc, index_int B, class C>
__device__ void
store_block(const acc_frag<Acc, B>& acc, index_int i0, index_int j0, C c, index_int lane)
{
    const index_int li = lane % B;
    const index_int lg = lane / B;
    for(index_int r = 0; r < vec_size<acc_frag<Acc, B>>(); r++)
        c(acc[r], i0 + frag_row<B>(r, lg), j0 + li);
}

// Without a matrix instruction each lane owns tiles of tm consecutive rows
// by tn columns and walks k with FMAs: tm rows when the lanes cover the
// matrix exactly, so a lane's loads of A along its rows stay adjacent, else
// one row per tile strided over the lanes. Tile t covers rows
// (t / (N / tn)) * tm.. and columns (t % (N / tn)) * tn.., so a lane's tiles
// share their columns of B when N / tn divides the wave
template <class Acc, index_int M, index_int N>
struct fma_tiling
{
    static constexpr index_int tn        = N % 4 == 0 ? 4 : (N % 2 == 0 ? 2 : 1);
    static constexpr index_int col_tiles = N / tn;
    static constexpr index_int exact     = (M * N) % (lanes * tn) == 0 ? (M * N) / (lanes * tn) : 0;
    static constexpr index_int tm       = (exact > 0 and exact <= 4 and M % exact == 0) ? exact : 1;
    static constexpr index_int tiles    = (M / tm) * col_tiles;
    static constexpr index_int per_lane = ceil_div(tiles, lanes);
    using tile_acc                      = array<array<Acc, tn>, tm>;

    __device__ static index_int row0(index_int t) { return (t / col_tiles) * tm; }
    __device__ static index_int col0(index_int t) { return (t % col_tiles) * tn; }

    // acc plus the rank-1 update from column k of A and row k of B
    template <class A, class B>
    __device__ static tile_acc
    accumulate(tile_acc acc, const A& a, const B& b, index_int t, index_int k)
    {
        const index_int i0 = row0(t);
        const index_int j0 = col0(t);
        array<Acc, tn> bv{};
        repeat_c<tn>([&](auto jj) { bv[jj] = b(k, j0 + jj); });
        repeat_c<tm>([&](auto ii) {
            const Acc av = a(i0 + ii, k);
            repeat_c<tn>([&](auto jj) { acc[ii][jj] += av * bv[jj]; });
        });
        return acc;
    }

    template <class C>
    __device__ static void store(const tile_acc& acc, const C& c, index_int t)
    {
        repeat_c<tm>([&](auto ii) {
            repeat_c<tn>([&](auto jj) { c(acc[ii][jj], row0(t) + ii, col0(t) + jj); });
        });
    }
};

template <class Acc, class T, index_int M, index_int N, index_int K, class A, class B, class C>
__device__ void matmul_fma(A a, B b, C c, index_int lane)
{
    using tiling = fma_tiling<Acc, M, N>;
    array<typename tiling::tile_acc, tiling::per_lane> acc{};
    for(index_int k = 0; k < K; k++)
    {
        repeat_c<tiling::per_lane>([&](auto s) {
            const index_int t = lane + s * lanes;
            if(t < tiling::tiles)
                acc[s] = tiling::accumulate(acc[s], a, b, t, k);
        });
    }
    repeat_c<tiling::per_lane>([&](auto s) {
        const index_int t = lane + s * lanes;
        if(t < tiling::tiles)
            tiling::store(acc[s], c, t);
    });
}

} // namespace matmul_detail

// The matrix instruction block matmul<T, M, N, K> runs on: the largest that
// tiles M and N with an instruction k dividing K, 0 for the FMA path
template <class T, index_int M, index_int N, index_int K>
constexpr index_int matmul_block()
{
    using acc = typename acc_type<T>::type;
    if constexpr(M % 32 == 0 and N % 32 == 0 and matmul_detail::mma_k<acc, T, 32, K>() > 0)
        return 32;
    else if constexpr(M % 16 == 0 and N % 16 == 0 and matmul_detail::mma_k<acc, T, 16, K>() > 0)
        return 16;
    else
        return 0;
}

// C[M][N] = sum_k A[M][K] B[K][N] by one wave: every lane calls it, a(i, k)
// and b(k, j) load the operands as T, c(x, i, j) receives each sum in
// acc_type<T> (float, double for double). Blocks of a shape the
// architecture's matrix instructions cover use them, anything else is
// per-lane FMAs. Index the loaders through rank-1 views so adjacent loads
// merge.
template <class T, index_int M, index_int N, index_int K, class A, class B, class C>
__device__ void matmul(A a, B b, C c)
{
    using acc              = typename acc_type<T>::type;
    const index_int lane   = make_index().local_wave();
    constexpr index_int bs = matmul_block<T, M, N, K>();
    if constexpr(bs > 0)
    {
        constexpr index_int ki = matmul_detail::mma_k<acc, T, bs, K>();
        for(index_int i0 = 0; i0 < M; i0 += bs)
        {
            for(index_int j0 = 0; j0 < N; j0 += bs)
            {
                matmul_detail::acc_frag<acc, bs> accf{};
                for(index_int k0 = 0; k0 < K; k0 += ki)
                    matmul_detail::mma_block<acc, T, bs, ki>(a, b, i0, j0, k0, accf, lane);
                matmul_detail::store_block<acc, bs>(accf, i0, j0, c, lane);
            }
        }
    }
    else
    {
        matmul_detail::matmul_fma<acc, T, M, N, K>(a, b, c, lane);
    }
}

} // namespace migraphx
#endif
