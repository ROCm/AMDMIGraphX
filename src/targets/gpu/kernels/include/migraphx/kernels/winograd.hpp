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
#include <migraphx/kernels/array.hpp>
#include <migraphx/kernels/functional.hpp>
#include <migraphx/kernels/tensor_view.hpp>

namespace migraphx {
namespace winograd {

// Winograd F(2x2,3x3) constants
constexpr auto Alpha   = _c<4>;
constexpr auto Alpha2  = _c<16>;
constexpr auto OutTile = _c<2>;

// 2D tile shape for 4x4 Winograd domain
inline constexpr auto tile_shape() { return make_shape(index_ints<4, 4>{}); }

// =============================================================================
// Transforms — generic over element type T
// Return arrays by value, use generate_array to avoid zero-init overhead.
// =============================================================================

template <class T>
__device__ auto input_xform(const array<T, 16>& d) -> array<T, 16>
{
    constexpr auto ts = tile_shape();
    auto tmp = generate_array<T>(_c<16>, [&](auto el) {
        constexpr auto j   = el % Alpha;
        constexpr auto row = el / Alpha;
        auto d0 = d[ts.index({_c<0>, j})];
        auto d1 = d[ts.index({_c<1>, j})];
        auto d2 = d[ts.index({_c<2>, j})];
        auto d3 = d[ts.index({_c<3>, j})];
        if constexpr(row == 0)
            return d0 - d2;
        else if constexpr(row == 1)
            return d1 + d2;
        else if constexpr(row == 2)
            return d2 - d1;
        else
            return d1 - d3;
    });
    return generate_array<T>(_c<16>, [&](auto el) {
        constexpr auto col = el % Alpha;
        constexpr auto row = el / Alpha;
        auto a  = tmp[ts.index({row, _c<0>})];
        auto b  = tmp[ts.index({row, _c<1>})];
        auto c  = tmp[ts.index({row, _c<2>})];
        auto dd = tmp[ts.index({row, _c<3>})];
        if constexpr(col == 0)
            return a - c;
        else if constexpr(col == 1)
            return b + c;
        else if constexpr(col == 2)
            return c - b;
        else
            return b - dd;
    });
}

template <class T>
__device__ auto filter_xform(const array<T, 9>& g) -> array<T, 16>
{
    constexpr auto gs = make_shape(index_ints<4, 3>{});
    auto tmp = generate_array<T>(_c<12>, [&](auto el) {
        constexpr auto j   = el % _c<3>;
        constexpr auto row = el / _c<3>;
        auto g0 = g[j];
        auto g1 = g[_c<3> + j];
        auto g2 = g[_c<6> + j];
        auto s  = (g0 + g2) * T(0.5);
        auto d  = g1 * T(0.5);
        if constexpr(row == 0)
            return g0;
        else if constexpr(row == 1)
            return s + d;
        else if constexpr(row == 2)
            return s - d;
        else
            return g2;
    });
    return generate_array<T>(_c<16>, [&](auto el) {
        constexpr auto col = el % Alpha;
        constexpr auto row = el / Alpha;
        auto t0 = tmp[gs.index({row, _c<0>})];
        auto t1 = tmp[gs.index({row, _c<1>})];
        auto t2 = tmp[gs.index({row, _c<2>})];
        auto s  = (t0 + t2) * T(0.5);
        auto d  = t1 * T(0.5);
        if constexpr(col == 0)
            return t0;
        else if constexpr(col == 1)
            return s + d;
        else if constexpr(col == 2)
            return s - d;
        else
            return t2;
    });
}

template <class T>
__device__ auto output_xform(const array<T, 16>& M) -> array<T, 4>
{
    constexpr auto ts = tile_shape();
    auto tmp = generate_array<T>(_c<8>, [&](auto el) {
        constexpr auto j   = el % Alpha;
        constexpr auto row = el / Alpha;
        auto m0 = M[ts.index({_c<0>, j})];
        auto m1 = M[ts.index({_c<1>, j})];
        auto m2 = M[ts.index({_c<2>, j})];
        auto m3 = M[ts.index({_c<3>, j})];
        if constexpr(row == 0)
            return m0 + m1 + m2;
        else
            return m1 - m2 - m3;
    });
    constexpr auto ts2 = make_shape(index_ints<2, 4>{});
    return generate_array<T>(_c<4>, [&](auto el) {
        constexpr auto col = el % OutTile;
        constexpr auto row = el / OutTile;
        auto a = tmp[ts2.index({row, _c<0>})];
        auto b = tmp[ts2.index({row, _c<1>})];
        auto c = tmp[ts2.index({row, _c<2>})];
        auto d = tmp[ts2.index({row, _c<3>})];
        if constexpr(col == 0)
            return a + b + c;
        else
            return b - c - d;
    });
}

// Filter precompute kernel (graph-level constant folding)
template <class Input, class Output>
__device__ void filter_precompute(Input weight, Output workspace)
{
    using T    = typename Input::type;
    auto total = weight.get_shape().lens[0] * weight.get_shape().lens[1];

    make_index().global_stride(total, [&](auto id) {
        auto g = generate_array<T>(_c<9>, [&](auto p) { return weight[id * 9 + p]; });
        auto U = filter_xform(g);
        repeat_c<Alpha2>([&](auto p) { workspace[id * Alpha2 + p] = U[p]; });
    });
}

// =============================================================================
// GEMM-based Winograd F(2x2,3x3) convolution
//
// Template parameters (CamelCase):
//   GroupCount, TilesPerWg, KPerWg, ChunkC  — workgroup tiling
//   Pretransformed — weight layout flag
//   TTile, KTile   — per-thread GEMM outer product tile
// =============================================================================

template <index_int GroupCount,
          index_int TilesPerWg,
          index_int KPerWg,
          index_int ChunkC,
          bool Pretransformed,
          index_int TTile = 2,
          index_int KTile = 2,
          class Input,
          class Weight,
          class Output,
          class LDS>
__device__ void conv(Input input, Weight weight, Output output, LDS& lds_buf)
{
    using T = typename Input::type;

    // Compile-time dimensions from tensor_view shape types
    constexpr auto in_lens   = typename Input::shape_type{}.lens;
    constexpr auto channels  = _c<in_lens[1]>;
    constexpr auto height    = _c<in_lens[2]>;
    constexpr auto width     = _c<in_lens[3]>;
    constexpr auto n_filters = _c<typename Output::shape_type{}.lens[1]>;

    constexpr auto ThreadsN   = _c<KPerWg / KTile>;
    constexpr auto TilesH     = (height + OutTile - _c<1>) / OutTile;
    constexpr auto TilesW     = (width + OutTile - _c<1>) / OutTile;
    constexpr auto TotalTiles = TilesH * TilesW;
    constexpr auto CPerGrp    = channels / _c<GroupCount>;
    constexpr auto KPerGrp    = n_filters / _c<GroupCount>;
    constexpr auto TileGrps   = (TotalTiles + _c<TilesPerWg> - _c<1>) / _c<TilesPerWg>;
    constexpr auto KGrps      = (n_filters + _c<KPerWg> - _c<1>) / _c<KPerWg>;
    constexpr auto VPlane     = _c<TilesPerWg * ChunkC>;

    // Interior tile range (compile-time)
    constexpr auto IntTrLo    = _c<1>;
    constexpr auto IntTrHi    = (height >= _c<4>) ? (height - _c<3>) / OutTile : _c<0>;
    constexpr auto IntTcLo    = _c<1>;
    constexpr auto IntTcHi    = (width >= _c<4>) ? (width - _c<3>) / OutTile : _c<0>;
    constexpr auto HasInterior = bool_constant<(IntTrHi >= IntTrLo and IntTcHi >= IntTcLo)>{};

    // LDS as 2D tensor_views: V[pos, tile*chunk+cc], U[pos, cc*kpw+kl]
    constexpr auto v_lds_shape = make_shape(index_ints<Alpha2, TilesPerWg * ChunkC>{});
    constexpr auto u_lds_shape = make_shape(index_ints<Alpha2, ChunkC * KPerWg>{});
    auto lds_v = make_tensor_view(lds_buf.data(), v_lds_shape);
    auto lds_u = make_tensor_view(lds_buf.data() + Alpha2 * VPlane, u_lds_shape);

    // Workgroup decomposition
    auto idx     = make_index();
    auto tid     = idx.local;
    auto wg      = idx.group;
    auto ntg     = wg / KGrps;
    auto k_grp   = wg % KGrps;
    auto n_val   = ntg / TileGrps;
    auto tg      = ntg % TileGrps;
    auto t_base  = tg * _c<TilesPerWg>;
    auto k_base  = k_grp * _c<KPerWg>;
    auto k_actual =
        (k_base + _c<KPerWg> > n_filters) ? (n_filters - k_base) : index_int{KPerWg};
    auto group_id = k_base / KPerGrp;
    auto c_base   = group_id * CPerGrp;

    auto thread_m = tid / ThreadsN;
    auto thread_n = tid % ThreadsN;
    auto my_t0    = thread_m * _c<TTile>;
    auto my_k0    = thread_n * _c<KTile>;

    // Accumulators: zero-initialized
    array<T, TTile * KTile * Alpha2> acc{};

    for(index_int c_chunk = 0; c_chunk < CPerGrp; c_chunk += ChunkC)
    {
        index_int csz = CPerGrp - c_chunk;
        if(csz > ChunkC)
            csz = ChunkC;

        // === Phase 1a: Input tile B^T*d*B transform → LDS ===
        {
            index_int total = _c<TilesPerWg> * csz;
            idx.local_stride(total, [&](auto i) {
                auto tl     = i / csz;
                auto cc     = i % csz;
                auto tg_idx = t_base + tl;

                array<T, 16> V{};
                if(tg_idx < TotalTiles)
                {
                    auto ic  = c_base + c_chunk + cc;
                    auto tr  = tg_idx / TilesW;
                    auto tc  = tg_idx % TilesW;
                    auto ih0 = static_cast<diff_int>(tr * OutTile) - 1;
                    auto iw0 = static_cast<diff_int>(tc * OutTile) - 1;

                    auto load_tile = [&]() {
                        if constexpr(HasInterior)
                        {
                            if(tr >= IntTrLo and tr <= IntTrHi and
                               tc >= IntTcLo and tc <= IntTcHi)
                            {
                                auto base =
                                    ((n_val * channels + ic) * height +
                                     static_cast<index_int>(ih0)) *
                                        width +
                                    static_cast<index_int>(iw0);
                                return generate_array<T>(_c<16>, [&](auto el) {
                                    return input[base + el / Alpha * width + el % Alpha];
                                });
                            }
                        }
                        return generate_array<T>(_c<16>, [&](auto el) {
                            auto ih = ih0 + static_cast<diff_int>(el / Alpha);
                            auto ih_ok =
                                ih >= 0 and ih < static_cast<diff_int>(height);
                            auto rb = ((n_val * channels + ic) * height +
                                       static_cast<index_int>(ih)) *
                                      width;
                            auto iw = iw0 + static_cast<diff_int>(el % Alpha);
                            return (ih_ok and iw >= 0 and
                                    iw < static_cast<diff_int>(width))
                                       ? input[rb + static_cast<index_int>(iw)]
                                       : T(0);
                        });
                    };

                    __builtin_amdgcn_sched_barrier(1 << 4);
                    V = input_xform(load_tile());
                }
                repeat_c<Alpha2>([&](auto p) {
                    lds_v[make_array<index_int>(p, tl * _c<ChunkC> + cc)] = V[p];
                });
            });
        }

        // === Phase 1b: Filter transform → LDS ===
        if constexpr(Pretransformed)
        {
            index_int total = csz * k_actual;
            idx.local_stride(total, [&](auto i) {
                auto cc  = i / k_actual;
                auto kl  = i % k_actual;
                auto kk  = k_base + kl;
                auto src = (kk * CPerGrp + c_chunk + cc) * Alpha2;
                repeat_c<Alpha2>([&](auto p) {
                    lds_u[make_array<index_int>(p, cc * _c<KPerWg> + kl)] = weight[src + p];
                });
            });
        }
        else
        {
            index_int total = csz * k_actual;
            idx.local_stride(total, [&](auto i) {
                auto cc    = i / k_actual;
                auto kl    = i % k_actual;
                auto kk    = k_base + kl;
                auto w_off = (kk * CPerGrp + c_chunk + cc) * _c<9>;
                auto g     = generate_array<T>(_c<9>, [&](auto p) {
                    return weight[w_off + p];
                });
                auto U = filter_xform(g);
                repeat_c<Alpha2>([&](auto p) {
                    lds_u[make_array<index_int>(p, cc * _c<KPerWg> + kl)] = U[p];
                });
            });
        }

        __builtin_amdgcn_sched_barrier(1 << 7);
        __syncthreads();
        __builtin_amdgcn_sched_barrier(0);

        // === Phase 2: Tiled GEMM from LDS ===
        __builtin_amdgcn_s_setprio(1);
        for(index_int cc = 0; cc < csz; cc++)
        {
            repeat_c<Alpha2>([&](auto p) {
                auto v = generate_array<T>(_c<TTile>, [&](auto tm) {
                    return lds_v[make_array<index_int>(p, (my_t0 + tm) * _c<ChunkC> + cc)];
                });
                auto u = generate_array<T>(_c<KTile>, [&](auto tn) {
                    return lds_u[make_array<index_int>(p, cc * _c<KPerWg> + my_k0 + tn)];
                });
                repeat_c<TTile>([&](auto tm) {
                    repeat_c<KTile>([&](auto tn) {
                        acc[(tm * _c<KTile> + tn) * Alpha2 + p] =
                            __builtin_fmaf(v[tm], u[tn],
                                           acc[(tm * _c<KTile> + tn) * Alpha2 + p]);
                    });
                });
            });
        }
        __builtin_amdgcn_s_setprio(0);
        __syncthreads();
    }

    // === Phase 3: Output transform and store ===
    repeat_c<TTile>([&](auto tm) {
        auto tile_idx = t_base + my_t0 + tm;
        if(tile_idx >= TotalTiles)
            return;
        auto tr  = tile_idx / TilesW;
        auto tc  = tile_idx % TilesW;
        auto oh0 = tr * OutTile;
        auto ow0 = tc * OutTile;
        repeat_c<KTile>([&](auto tn) {
            auto kk = k_base + my_k0 + tn;
            if(kk >= n_filters)
                return;
            auto M = generate_array<T>(_c<16>, [&](auto p) {
                return acc[(tm * _c<KTile> + tn) * Alpha2 + p];
            });
            auto Y = output_xform(M);
            repeat_c<OutTile>([&](auto oi) {
                auto oh = oh0 + oi;
                if(oh >= height)
                    return;
                auto row =
                    ((n_val * n_filters + kk) * height + oh) * width;
                repeat_c<OutTile>([&](auto oj) {
                    auto ow = ow0 + oj;
                    if(ow >= width)
                        return;
                    output[row + ow] = Y[oi * OutTile + oj];
                });
            });
        });
    });
}

} // namespace winograd
} // namespace migraphx

#endif // MIGRAPHX_GUARD_KERNELS_WINOGRAD_HPP
