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
#include <migraphx/kernels/matmul.hpp>

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

// Walk the chunks of a linear recurrence with the state tile for one problem
// and one dv block resident in LDS. One workgroup per (problem, dv block):
// each chunk reads the state with q_i (and w_i under the read module), writes
// u into LDS, then updates the state with kt_i u (or the precomputed update)
// through the step module. The products are wave-level matmuls over 16x16
// blocks (per-lane FMAs where the architecture has no matrix instruction),
// with half operands accumulating in float. The kernel arguments are packed
// in args and found by the role indices in Settings; see gpu::chunk_scan for
// the roles.
template <class Settings, class Step, class Read, class Out, class Args>
struct chunk_scanner
{
    template <index_int I>
    using arg_t      = decltype(declval<Args>()(arg_c<I>()));
    using value_type = typename arg_t<Settings::q_index>::type;
    // half and bf16 operands accumulate in float, like the gemms they replace
    using acc_type = conditional_t<is_same<value_type, float>{} or is_same<value_type, double>{},
                                   value_type,
                                   float>;
    static constexpr index_int chunks = Settings::chunks;
    static constexpr index_int rows   = Settings::rows;
    static constexpr index_int dk     = Settings::dk;
    static constexpr index_int dv     = Settings::dv;
    static constexpr index_int dvb    = Settings::dvb;
    // the blocks of the products: 16 wide where the axis allows it, and 32
    // rows of the update so a lane's FMA tile without a matrix instruction
    // reaches 4x4
    static constexpr index_int bm = rows % 16 == 0 ? 16 : rows;
    static constexpr index_int bn = dvb % 16 == 0 ? 16 : dvb;
    static constexpr index_int bd = dk % 32 == 0 ? 32 : (dk % 16 == 0 ? 16 : dk);
    // the leading axes index the problems; multi() of this shape gives the
    // leading index with zeros in the matrix axes
    static constexpr auto so            = get_shape_c<arg_t<Settings::state_out_index>>{};
    static constexpr auto lead          = make_shape(transform_i(
        so.lens, [](auto x, auto i) { return i < decltype(so.lens){}.size() - 2 ? x : 1; }));
    static constexpr auto grid          = make_shape(index_ints<lead.elements(), dv / dvb>{});
    static constexpr auto s_tile        = make_shape(index_ints<dk, dvb>{});
    static constexpr auto u_tile        = make_shape(index_ints<rows, dvb>{});
    static constexpr auto read_blocks   = make_shape(index_ints<rows / bm, dvb / bn>{});
    static constexpr auto update_blocks = make_shape(index_ints<dk / bd, dvb / bn>{});
    // the state and u tiles in LDS
    using s_view = decltype(make_tensor_view<lds_memory_tag>(declval<acc_type*>(), s_tile));
    using u_view = decltype(make_tensor_view<lds_memory_tag>(declval<acc_type*>(), u_tile));

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

    // f(block) for the blocks of a phase, strided over the waves
    template <class Blocks, class F>
    __device__ void for_blocks(Blocks blocks, F f) const
    {
        for(index_int b = idx.wave(); b < blocks.elements(); b += idx.nwave())
            f(blocks.multi(b));
    }

    __device__ void load_state() const
    {
        auto s0m = mat<Settings::s0_index, get_shape_c<arg_t<Settings::s0_index>>>(0);
        idx.local_stride(s_tile.elements(), [&](auto e) {
            auto t = s_tile.multi(e);
            if constexpr(Settings::has_state)
                s[t] = row(s0m, t[0])[col0 + t[1]];
            else
                s[t] = 0;
        });
    }

    __device__ void store_state() const
    {
        auto som = mat<Settings::state_out_index, decltype(so)>(0);
        idx.local_stride(s_tile.elements(), [&](auto e) {
            auto t                      = s_tile.multi(e);
            row(som, t[0])[col0 + t[1]] = s[t];
        });
    }

    // chunk 0 of a zero start reads nothing: u = u0 and o = pu0, or zero
    __device__ void copy_chunk(index_int i) const
    {
        auto u0i = chunk<Settings::u0_index, decltype(Settings::u0_shape)>(
            i, Settings::u0_offset, Settings::u0_delta);
        auto pu0i = chunk<Settings::pu0_index, decltype(Settings::pu0_shape)>(
            i, Settings::pu0_offset, Settings::pu0_delta);
        auto outm = mat<Settings::output_index, get_shape_c<arg_t<Settings::output_index>>>(0);
        idx.local_stride(u_tile.elements(), [&](auto e) {
            auto t       = u_tile.multi(e);
            const auto r = t[0];
            const auto c = col0 + t[1];
            if constexpr(Settings::has_update_dot)
                u[t] = row(u0i, r)[c];
            acc_type ov = 0;
            if constexpr(Settings::has_out)
                ov = row(pu0i, r)[c];
            row(outm, i * rows + r)[c] = ov;
        });
    }

    // o_i = out(pu0_i, q_i S) and u = read(u0_i, w_i S) for chunk i
    __device__ void read_chunk(index_int i) const
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
        if constexpr(Settings::has_update_dot and not Settings::has_read)
        {
            idx.local_stride(u_tile.elements(), [&](auto e) {
                auto t = u_tile.multi(e);
                u[t]   = row(u0i, t[0])[col0 + t[1]];
            });
        }
        for_blocks(read_blocks, [&](auto blk) {
            const index_int r0 = blk[0] * bm;
            const index_int v0 = blk[1] * bn;
            const index_int c0 = col0 + v0;
            auto store_out     = [&](acc_type x, index_int r, index_int j) {
                acc_type ov = x;
                if constexpr(Settings::has_out)
                    ov = out_fn(row(pu0i, r)[c0 + j], x);
                row(outm, i * rows + r)[c0 + j] = ov;
            };
            matmul<value_type, stacked, bn, dk>(
                [&](auto r, auto k) {
                    if constexpr(Settings::has_read)
                    {
                        if(is_w(r))
                            return row(wi, r0 + src_row(r))[k];
                    }
                    return row(qi, r0 + src_row(r))[k];
                },
                [&](auto k, auto j) { return row(s, k)[v0 + j]; },
                [&](auto x, auto r, auto j) {
                    if constexpr(Settings::has_read)
                    {
                        if(is_w(r))
                        {
                            const index_int ur = r0 + src_row(r);
                            row(u, ur)[v0 + j] = read(row(u0i, ur)[c0 + j], x);
                            return;
                        }
                    }
                    store_out(x, r0 + src_row(r), j);
                });
        });
    }

    // Under the read module the q and w blocks stack as one operand sharing
    // the state tile: groups of g rows alternate between them, 16 so a
    // matrix instruction block stays in one, else 1 so a lane's tile holds
    // a q row and its w row
    static constexpr index_int stacked = Settings::has_read ? 2 * bm : bm;
    static constexpr index_int g       = matmul_uses_mma<value_type, stacked, bn, dk>() ? 16 : 1;
    __device__ static bool is_w(index_int r) { return (r / g) % 2 == 1; }
    __device__ static index_int src_row(index_int r)
    {
        if constexpr(Settings::has_read)
            return (r / (2 * g)) * g + r % g;
        else
            return r;
    }

    // S = step(S, [gamma_i], up) for one state element, or up on a zero start
    template <class GI>
    __device__ void
    transition(acc_type& sv, GI gi, index_int d, index_int c, acc_type up, bool live) const
    {
        if(not live)
            sv = up;
        else if constexpr(Settings::has_gamma)
            sv = step(sv, row(gi, d)[c], up);
        else
            sv = step(sv, up);
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
        if constexpr(Settings::has_update_dot)
        {
            for_blocks(update_blocks, [&](auto blk) {
                const index_int d0 = blk[0] * bd;
                const index_int v0 = blk[1] * bn;
                matmul<value_type, bd, bn, rows>(
                    [&](auto d, auto r) { return column(kti, r)[d0 + d]; },
                    [&](auto r, auto j) { return row(u, r)[v0 + j]; },
                    [&](auto x, auto d, auto j) {
                        transition(row(s, d0 + d)[v0 + j], gi, d0 + d, col0 + v0 + j, x, live);
                    });
            });
        }
        else
        {
            idx.local_stride(s_tile.elements(), [&](auto e) {
                auto t = s_tile.multi(e);
                transition(s[t], gi, t[0], col0 + t[1], row(updi, t[0])[col0 + t[1]], live);
            });
        }
    }

    __device__ void run() const
    {
        load_state();
        __syncthreads();
        for(index_int i = 0; i < chunks; i++)
        {
            // chunk 0 of a zero start skips the reads and the transition
            const bool live = Settings::has_state or i > 0;
            if(live)
                read_chunk(i);
            else
                copy_chunk(i);
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
    // 16-byte aligned so adjacent lane loads of a row merge
    __shared__ uninitialized_buffer<typename scanner::acc_type, scanner::s_tile.elements(), 16>
        s_buf;
    __shared__ uninitialized_buffer<typename scanner::acc_type, scanner::u_tile.elements(), 16>
        u_buf;
    auto idx = make_index();
    auto pb  = scanner::grid.multi(idx.group);
    scanner{step,
            read,
            out_fn,
            pack(xs...),
            idx,
            scanner::lead.multi(pb[0]),
            pb[1] * scanner::dvb,
            make_tensor_view<lds_memory_tag>(s_buf.data(), scanner::s_tile),
            make_tensor_view<lds_memory_tag>(u_buf.data(), scanner::u_tile)}
        .run();
}

} // namespace migraphx
#endif
