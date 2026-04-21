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

// A^T * M * A producing the 2x2 output tile.
template <class T, class Acc>
__device__ __attribute__((const)) array<T, 4> output_transform(array<Acc, 16> m)
{
    array<Acc, 8> r{};
    repeat_c<4>([&](auto j) {
        r[0u * 4u + j] = m[0u * 4u + j] + m[1u * 4u + j] + m[2u * 4u + j];
        r[1u * 4u + j] = m[1u * 4u + j] - m[2u * 4u + j] - m[3u * 4u + j];
    });
    array<T, 4> y{};
    repeat_c<2>([&](auto i) {
        auto r0        = r[i * 4u + 0u];
        auto r1        = r[i * 4u + 1u];
        auto r2        = r[i * 4u + 2u];
        auto r3        = r[i * 4u + 3u];
        y[i * 2u + 0u] = static_cast<T>(r0 + r1 + r2);
        y[i * 2u + 1u] = static_cast<T>(r1 - r2 - r3);
    });
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
    (void)0;

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
        repeat_c<16>([&](auto e) {
            repeat_c<OP_M>([&](auto m) {
                repeat_c<OP_N>([&](auto nn) {
                    const auto ai = (m * OP_N + nn) * 16u + e;
                    if constexpr(sizeof(out_type) == 2u)
                    {
                        // Fuse both channels of the pair into one v_dot2.
                        vec<half, 2> up;
                        vec<half, 2> vp;
                        up[0]   = u_all[e][m * 2u + 0u];
                        up[1]   = u_all[e][m * 2u + 1u];
                        vp[0]   = v_all[e][nn * 2u + 0u];
                        vp[1]   = v_all[e][nn * 2u + 1u];
                        acc[ai] = dot2_acc(vp, up, acc[ai]);
                    }
                    else
                    {
                        acc[ai] += static_cast<Acc>(u_all[e][m * 2u + 0u]) *
                                   static_cast<Acc>(v_all[e][nn * 2u + 0u]);
                        acc[ai] += static_cast<Acc>(u_all[e][m * 2u + 1u]) *
                                   static_cast<Acc>(v_all[e][nn * 2u + 1u]);
                    }
                });
            });
        });
        set_prio<0>();
        __syncthreads();
    }

    // ----- Output transform + store.
    repeat_c<OP_M>([&](auto m) {
        const index_int my_k = k_block * K_PER_BLOCK + k_in_grid * OP_M + m;
        if(my_k >= K_)
            return;
        repeat_c<OP_N>([&](auto nn) {
            const index_int my_tile = tile_block * TILES_PER_BLOCK + t_in_grid * OP_N + nn;
            if(my_tile >= total_)
                return;
            const index_int n_     = my_tile / t_pi;
            const index_int t_img  = my_tile % t_pi;
            const index_int th     = t_img / t_w;
            const index_int tw     = t_img % t_w;
            const index_int base_h = th * 2u;
            const index_int base_w = tw * 2u;

            array<Acc, 16> m16;
            repeat_c<16>([&](auto e) { m16[e] = acc[(m * OP_N + nn) * 16u + e]; });
            const auto yt = output_transform<out_type>(m16);
            repeat_c<2>([&](auto ii) {
                repeat_c<2>([&](auto jj) {
                    const index_int h_out = base_h + ii;
                    const index_int w_out = base_w + jj;
                    if(h_out < H_out and w_out < W_out)
                    {
                        y[make_array<index_int>(n_, my_k, h_out, w_out)] = yt[ii * 2u + jj];
                    }
                });
            });
        });
    });
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
