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

#ifndef MIGRAPHX_GUARD_KERNELS_CHUNK_SCAN_HPP
#define MIGRAPHX_GUARD_KERNELS_CHUNK_SCAN_HPP

#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/tensor_view.hpp>
#include <migraphx/kernels/uninitialized_buffer.hpp>
#include <migraphx/kernels/vec.hpp>
#include <migraphx/kernels/functional.hpp>
#include <migraphx/kernels/slice.hpp>

namespace migraphx {

// The matrix axes of the view Shape as a tensor_view: at the leading index
// lead, offset elements into the base's buffer
template <class Shape, class Base, class Lead>
__device__ auto matrix_slice(Base base, Lead lead, index_int offset)
{
    constexpr auto s   = Shape{};
    constexpr auto n   = s.lens.size();
    constexpr auto mat = make_shape(index_ints<s.lens[n - 2], s.lens[n - 1]>{},
                                    index_ints<s.strides[n - 2], s.strides[n - 1]>{});
    return base.with(base.data() + s.index(lead) + offset, mat);
}

// Row r and column c of a matrix view. Indexing a rank-1 slice keeps the
// address affine in the loop variable, which lets adjacent loads merge
template <class View>
__device__ auto row(View m, index_int r)
{
    return slice_tensor(m, array<index_int, 2>{r, 0}, slice_axes<1>());
}

template <class View>
__device__ auto column(View m, index_int c)
{
    return slice_tensor(m, array<index_int, 2>{0, c}, slice_axes<0>());
}

// N columns of a row as one tile
template <class T, index_int N, class View>
__device__ auto load_cols(View m, index_int r, index_int c)
{
    auto mr = row(m, r);
    if constexpr(N == 1)
    {
        return mr[c];
    }
    else
    {
        vec<T, N> v{};
        for(index_int j = 0; j < N; j++)
            v[j] = mr[c + j];
        return v;
    }
}

template <index_int N, class View, class V>
__device__ void store_cols(View m, index_int r, index_int c, V v)
{
    auto mr = row(m, r);
    if constexpr(N == 1)
        mr[c] = v;
    else
        for(index_int j = 0; j < N; j++)
            mr[c + j] = v[j];
}

// Walk the chunks of a linear recurrence with the state tile for one problem
// and one dv block resident in LDS. One workgroup per (problem, dv block):
// each chunk reads the state with q_i (and w_i under the read module), writes
// u into LDS, then updates the state with kt_i u (or the precomputed update)
// through the step module. Each thread owns a tr x tv (read) or td x tv
// (update) register tile of tv-wide column vectors. The kernel arguments are
// packed in args and found by the role indices in Settings; see
// gpu::chunk_scan for the roles.
template <class Settings, class Step, class Read, class Out, class Args>
struct chunk_scanner
{
    template <index_int I>
    using arg_t      = decltype(declval<Args>()(arg_c<I>()));
    using value_type = typename arg_t<Settings::q_index>::type; // the operands' type accumulates
    static constexpr index_int chunks = Settings::chunks;
    static constexpr index_int rows   = Settings::rows;
    static constexpr index_int dk     = Settings::dk;
    static constexpr index_int dv     = Settings::dv;
    static constexpr index_int dvb    = Settings::dvb;
    static constexpr index_int tv     = Settings::tv;
    static constexpr index_int tr     = Settings::tr;
    static constexpr index_int td     = Settings::td;
    static constexpr index_int nvt    = dvb / tv;
    using tile_t                      = conditional_t<(tv > 1), vec<value_type, tv>, value_type>;
    // the leading axes index the problems; multi() of this shape gives the
    // leading index with zeros in the matrix axes
    static constexpr auto so           = get_shape_c<arg_t<Settings::state_out_index>>{};
    static constexpr auto lead         = make_shape(transform_i(
        so.lens, [](auto x, auto i) { return i < decltype(so.lens){}.size() - 2 ? x : 1; }));
    static constexpr auto grid         = make_shape(index_ints<lead.elements(), dv / dvb>{});
    static constexpr auto s_tiles      = make_shape(index_ints<dk, nvt>{});
    static constexpr auto u_tiles      = make_shape(index_ints<rows, nvt>{});
    static constexpr auto read_tiles   = make_shape(index_ints<rows / tr, nvt>{});
    static constexpr auto update_tiles = make_shape(index_ints<dk / td, nvt>{});
    // the state and u tiles in LDS, as dk x nvt and rows x nvt column vectors
    using s_view = decltype(make_tensor_view<lds_memory_tag>(declval<tile_t*>(), s_tiles));
    using u_view = decltype(make_tensor_view<lds_memory_tag>(declval<tile_t*>(), u_tiles));

    Step step;
    Read read;
    Out out_fn;
    Args args;
    index idx;
    decltype(lead.multi(0)) base;
    index_int col0;
    s_view s;
    u_view u;

    // the matrix of argument I at this problem, offset into its base
    template <index_int I, class Shape>
    __device__ auto mat(index_int offset) const
    {
        return matrix_slice<Shape>(args(arg_c<I>()), base, offset);
    }

    // the chunk i matrix of a family argument
    template <index_int I, class Shape>
    __device__ auto chunk(index_int i, index_int offset, index_int delta) const
    {
        return mat<I, Shape>(offset + i * delta);
    }

    __device__ void load_state() const
    {
        auto s0m = mat<Settings::s0_index, get_shape_c<arg_t<Settings::s0_index>>>(0);
        idx.local_stride(s_tiles.elements(), [&](auto e) {
            auto t = s_tiles.multi(e);
            if constexpr(Settings::has_state)
                s[t] = load_cols<value_type, tv>(s0m, t[0], col0 + t[1] * tv);
            else
                s[t] = tile_t{};
        });
    }

    __device__ void store_state() const
    {
        auto som = mat<Settings::state_out_index, decltype(so)>(0);
        idx.local_stride(s_tiles.elements(), [&](auto e) {
            auto t = s_tiles.multi(e);
            store_cols<tv>(som, t[0], col0 + t[1] * tv, s[t]);
        });
    }

    // o_i = out(pu0_i, q_i S) and u = read(u0_i, w_i S) for chunk i
    __device__ void read_chunk(index_int i, bool live) const
    {
        auto wi = chunk<Settings::w_index, decltype(Settings::w_shape)>(
            i, Settings::w_offset, Settings::w_delta);
        auto qi = chunk<Settings::q_index, decltype(Settings::q_shape)>(
            i, Settings::q_offset, Settings::q_delta);
        auto u0i = chunk<Settings::u0_index, decltype(Settings::u0_shape)>(
            i, Settings::u0_offset, Settings::u0_delta);
        auto pu0i = chunk<Settings::pu0_index, decltype(Settings::pu0_shape)>(
            i, Settings::pu0_offset, Settings::pu0_delta);
        auto outm = mat<Settings::output_index, get_shape_c<arg_t<Settings::output_index>>>(0);
        idx.local_stride(read_tiles.elements(), [&](auto e) {
            auto t        = read_tiles.multi(e);
            const auto r0 = t[0] * tr;
            const auto vt = t[1];
            const auto c0 = col0 + vt * tv;
            array<tile_t, tr> qs{};
            array<tile_t, tr> ws{};
            if(live)
            {
                auto scol = column(s, vt);
                for(index_int k = 0; k < dk; k++)
                {
                    const tile_t sk = scol[k];
                    repeat_c<tr>([&](auto a) {
                        const index_int r = r0 + a;
                        qs[a] += row(qi, r)[k] * sk;
                        if constexpr(Settings::has_read)
                            ws[a] += row(wi, r)[k] * sk;
                    });
                }
            }
            repeat_c<tr>([&](auto a) {
                const index_int r = r0 + a;
                if constexpr(Settings::has_update_dot)
                {
                    auto uv = load_cols<value_type, tv>(u0i, r, c0);
                    if constexpr(Settings::has_read)
                    {
                        if(live)
                            uv = vec_transform(uv, ws[a])(read);
                    }
                    column(u, vt)[r] = uv;
                }
                tile_t ov = qs[a];
                if constexpr(Settings::has_out)
                {
                    auto pv = load_cols<value_type, tv>(pu0i, r, c0);
                    ov      = live ? tile_t{vec_transform(pv, qs[a])(out_fn)} : pv;
                }
                store_cols<tv>(outm, i * rows + r, c0, ov);
            });
        });
    }

    // S = step(S, [gamma_i], kt_i u or upd_i) for chunk i
    __device__ void update_chunk(index_int i, bool live) const
    {
        auto kti = chunk<Settings::kt_index, decltype(Settings::kt_shape)>(
            i, Settings::kt_offset, Settings::kt_delta);
        auto updi = chunk<Settings::upd_index, decltype(Settings::upd_shape)>(
            i, Settings::upd_offset, Settings::upd_delta);
        auto gi = chunk<Settings::gamma_index, decltype(Settings::gamma_shape)>(
            i, Settings::gamma_offset, Settings::gamma_delta);
        idx.local_stride(update_tiles.elements(), [&](auto e) {
            auto t        = update_tiles.multi(e);
            const auto d0 = t[0] * td;
            const auto vt = t[1];
            const auto c0 = col0 + vt * tv;
            array<tile_t, td> up{};
            if constexpr(Settings::has_update_dot)
            {
                auto ucol = column(u, vt);
                for(index_int r = 0; r < rows; r++)
                {
                    const tile_t ur = ucol[r];
                    repeat_c<td>([&](auto a) {
                        const index_int d = d0 + a;
                        up[a] += column(kti, r)[d] * ur;
                    });
                }
            }
            else
            {
                repeat_c<td>([&](auto a) { up[a] = load_cols<value_type, tv>(updi, d0 + a, c0); });
            }
            auto scol = column(s, vt);
            repeat_c<td>([&](auto a) {
                const index_int d = d0 + a;
                auto& sv          = scol[d];
                if(live)
                {
                    if constexpr(Settings::has_gamma)
                        sv = vec_transform(sv, load_cols<value_type, tv>(gi, d, c0), up[a])(step);
                    else
                        sv = vec_transform(sv, up[a])(step);
                }
                else
                {
                    sv = up[a];
                }
            });
        });
    }

    __device__ void run() const
    {
        load_state();
        __syncthreads();
        for(index_int i = 0; i < chunks; i++)
        {
            // chunk 0 of a zero start skips the reads and the transition
            const bool live = Settings::has_state or i > 0;
            read_chunk(i, live);
            __syncthreads();
            update_chunk(i, live);
            __syncthreads();
        }
        store_state();
    }
};

template <class Settings, class Step, class Read, class Out, class... Ts>
__device__ void chunk_scan(Step step, Read read, Out out_fn, Ts... xs)
{
    using scanner = chunk_scanner<Settings, Step, Read, Out, decltype(pack(xs...))>;
    __shared__ uninitialized_buffer<typename scanner::tile_t, scanner::s_tiles.elements()> s_buf;
    __shared__ uninitialized_buffer<typename scanner::tile_t, scanner::u_tiles.elements()> u_buf;
    auto idx = make_index();
    auto pb  = scanner::grid.multi(idx.group);
    scanner{step,
            read,
            out_fn,
            pack(xs...),
            idx,
            scanner::lead.multi(pb[0]),
            pb[1] * scanner::dvb,
            make_tensor_view<lds_memory_tag>(s_buf.data(), scanner::s_tiles),
            make_tensor_view<lds_memory_tag>(u_buf.data(), scanner::u_tiles)}
        .run();
}

} // namespace migraphx
#endif
