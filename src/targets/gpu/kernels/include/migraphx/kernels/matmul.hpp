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
#include <migraphx/kernels/vec.hpp>
#include <migraphx/kernels/types.hpp>
#include <migraphx/kernels/type_traits.hpp>
#include <migraphx/kernels/bit_cast.hpp>
#include <migraphx/kernels/array.hpp>
#include <migraphx/kernels/functional.hpp>

namespace migraphx {
namespace matmul_detail {

// The k advanced by one instruction
template <class T>
constexpr index_int mma_k()
{
    return is_same<T, float>{} ? 4 : 16;
}

// The 16x16 matrix instruction of the architecture: which types it takes,
// the lane's fragment of the accumulator, the row of the block fragment
// element r holds for lane half (quarter on wave64) lh, and the block
// product acc += A[i0.., k0..] B[k0.., j0..] over one instruction of k.
// Every layout gives lane l row l % 16 of A and column l % 16 of B and D;
// the lane halves split k
#if defined(__GFX12__)
// RDNA4: wave32 WMMA, each lane half holds 8 of the 16 k
template <class T>
constexpr bool has_mma()
{
    return MIGRAPHX_WAVEFRONTSIZE == 32 and (is_same<T, half>{} or is_same<T, bf16>{});
}

using acc_frag = vec<float, 8>;

__device__ constexpr index_int frag_row(index_int r, index_int lh) { return 8 * lh + r; }

template <class T, class A, class B>
__device__ void
mma_block(A a, B b, index_int i0, index_int j0, index_int k0, acc_frag& acc, index_int lane)
{
    const index_int li = lane % 16;
    const index_int lh = lane / 16;
    vec<T, 8> av{};
    vec<T, 8> bv{};
    for(index_int r = 0; r < 8; r++)
    {
        av[r] = a(i0 + li, k0 + 8 * lh + r);
        bv[r] = b(k0 + 8 * lh + r, j0 + li);
    }
    if constexpr(is_same<T, half>{})
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(av, bv, acc);
    else
        acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(av, bv, acc);
}
#elif defined(__GFX11__)
// RDNA3: wave32 WMMA, both lane halves hold all 16 k, D rows interleave
template <class T>
constexpr bool has_mma()
{
    return MIGRAPHX_WAVEFRONTSIZE == 32 and (is_same<T, half>{} or is_same<T, bf16>{});
}

using acc_frag = vec<float, 8>;

__device__ constexpr index_int frag_row(index_int r, index_int lh) { return 2 * r + lh; }

template <class T, class A, class B>
__device__ void
mma_block(A a, B b, index_int i0, index_int j0, index_int k0, acc_frag& acc, index_int lane)
{
    const index_int li = lane % 16;
    vec<T, 16> av{};
    vec<T, 16> bv{};
    for(index_int r = 0; r < 16; r++)
    {
        av[r] = a(i0 + li, k0 + r);
        bv[r] = b(k0 + r, j0 + li);
    }
    if constexpr(is_same<T, half>{})
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(av, bv, acc);
    else
        acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
            bit_cast<short>(av), bit_cast<short>(bv), acc);
}
#elif defined(__gfx908__) or defined(__gfx90a__) or defined(__gfx940__) or defined(__gfx941__) or \
    defined(__gfx942__) or defined(__gfx950__)
// CDNA: wave64 MFMA, each lane quarter holds 4 of the 16 k (1 of 4 for f32)
template <class T>
constexpr bool has_mma()
{
#if defined(__gfx908__)
    return MIGRAPHX_WAVEFRONTSIZE == 64 and (is_same<T, half>{} or is_same<T, float>{});
#else
    return MIGRAPHX_WAVEFRONTSIZE == 64 and
           (is_same<T, half>{} or is_same<T, bf16>{} or is_same<T, float>{});
#endif
}

using acc_frag = vec<float, 4>;

__device__ constexpr index_int frag_row(index_int r, index_int lh) { return 4 * lh + r; }

template <class T, class A, class B>
__device__ void
mma_block(A a, B b, index_int i0, index_int j0, index_int k0, acc_frag& acc, index_int lane)
{
    const index_int li = lane % 16;
    const index_int lh = lane / 16;
    if constexpr(is_same<T, float>{})
    {
        const float av = a(i0 + li, k0 + lh);
        const float bv = b(k0 + lh, j0 + li);
        acc            = __builtin_amdgcn_mfma_f32_16x16x4f32(av, bv, acc, 0, 0, 0);
    }
    else
    {
        vec<T, 4> av{};
        vec<T, 4> bv{};
        for(index_int r = 0; r < 4; r++)
        {
            av[r] = a(i0 + li, k0 + 4 * lh + r);
            bv[r] = b(k0 + 4 * lh + r, j0 + li);
        }
        if constexpr(is_same<T, half>{})
            acc = __builtin_amdgcn_mfma_f32_16x16x16f16(av, bv, acc, 0, 0, 0);
        else
            acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
                bit_cast<short>(av), bit_cast<short>(bv), acc, 0, 0, 0);
    }
}
#else
// no matrix instruction: matmul takes the FMA path and never instantiates these
template <class T>
constexpr bool has_mma()
{
    return false;
}

using acc_frag = vec<float, 8>;

__device__ constexpr index_int frag_row(index_int r, index_int lh) { return 8 * lh + r; }

template <class T, class A, class B>
__device__ void mma_block(A, B, index_int, index_int, index_int, acc_frag&, index_int)
{
}
#endif

// c(x, i, j) for the lane's fragment of block (i0, j0)
template <class C>
__device__ void store_block(const acc_frag& acc, index_int i0, index_int j0, C c, index_int lane)
{
    const index_int li = lane % 16;
    const index_int lh = lane / 16;
    for(index_int r = 0; r < vec_size<acc_frag>(); r++)
        c(acc[r], i0 + frag_row(r, lh), j0 + li);
}

// Without a matrix instruction each lane owns tiles of tm consecutive rows
// by tn columns and walks k with FMAs: tm rows when the lanes cover the
// matrix exactly, so a lane's loads of A along its rows stay adjacent, else
// one row per tile strided over the lanes. Tile t covers rows
// (t / (N / tn)) * tm.. and columns (t % (N / tn)) * tn.., so the tiles of
// a lane share their columns of B
template <class Acc, index_int M, index_int N>
struct fma_tiling
{
    static constexpr index_int lanes     = MIGRAPHX_WAVEFRONTSIZE;
    static constexpr index_int tn        = N % 4 == 0 ? 4 : (N % 2 == 0 ? 2 : 1);
    static constexpr index_int col_tiles = N / tn;
    static constexpr index_int exact     = (M * N) % (lanes * tn) == 0 ? (M * N) / (lanes * tn) : 0;
    static constexpr index_int tm       = (exact > 0 and exact <= 4 and M % exact == 0) ? exact : 1;
    static constexpr index_int tiles    = (M / tm) * col_tiles;
    static constexpr index_int per_lane = (tiles + lanes - 1) / lanes;
    using tile_acc                      = array<array<Acc, tn>, tm>;

    __device__ static index_int row0(index_int t) { return (t / col_tiles) * tm; }
    __device__ static index_int col0(index_int t) { return (t % col_tiles) * tn; }

    // the tile after column k of A and row k of B
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

template <class T, index_int M, index_int N, index_int K, class A, class B, class C>
__device__ void matmul_fma(A a, B b, C c, index_int lane)
{
    using acc_type = conditional_t<is_same<T, double>{}, double, float>;
    using tiling   = fma_tiling<acc_type, M, N>;
    array<typename tiling::tile_acc, tiling::per_lane> acc{};
    for(index_int k = 0; k < K; k++)
    {
        repeat_c<tiling::per_lane>([&](auto s) {
            const index_int t = lane + s * tiling::lanes;
            if(t < tiling::tiles)
                acc[s] = tiling::accumulate(acc[s], a, b, t, k);
        });
    }
    repeat_c<tiling::per_lane>([&](auto s) {
        const index_int t = lane + s * tiling::lanes;
        if(t < tiling::tiles)
            tiling::store(acc[s], c, t);
    });
}

} // namespace matmul_detail

// Whether matmul<T, M, N, K> runs on the matrix instruction
template <class T, index_int M, index_int N, index_int K>
constexpr bool matmul_uses_mma()
{
    return matmul_detail::has_mma<T>() and M % 16 == 0 and N % 16 == 0 and
           K % matmul_detail::mma_k<T>() == 0;
}

// C[M][N] = sum_k A[M][K] B[K][N] by one wave: every lane calls it, a(i, k)
// and b(k, j) load the operands as T, c(x, i, j) receives each sum in float
// (double for double). 16x16 blocks of a shape the architecture's matrix
// instruction covers use it, anything else is per-lane FMAs. Index the
// loaders through rank-1 views so adjacent loads merge.
template <class T, index_int M, index_int N, index_int K, class A, class B, class C>
__device__ void matmul(A a, B b, C c)
{
    const index_int lane = make_index().local_wave();
    if constexpr(matmul_uses_mma<T, M, N, K>())
    {
        for(index_int i0 = 0; i0 < M; i0 += 16)
        {
            for(index_int j0 = 0; j0 < N; j0 += 16)
            {
                matmul_detail::acc_frag acc{};
                for(index_int k0 = 0; k0 < K; k0 += matmul_detail::mma_k<T>())
                    matmul_detail::mma_block<T>(a, b, i0, j0, k0, acc, lane);
                matmul_detail::store_block(acc, i0, j0, c, lane);
            }
        }
    }
    else
    {
        matmul_detail::matmul_fma<T, M, N, K>(a, b, c, lane);
    }
}

} // namespace migraphx
#endif
