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
#include <migraphx/kernels/functional.hpp>
#include <migraphx/kernels/slice.hpp>
#include <migraphx/kernels/matmul.hpp>

namespace migraphx {

// Walk the chunks of a linear recurrence with the state tile of one problem
// and one dv block resident in LDS, one workgroup per (problem, dv block).
// Each chunk reads the state with its output matrix (and its read matrix
// under the read module), keeps u in LDS, then steps the state with its
// update matrix times u (or the precomputed update). The products are
// wave-level matmuls. See gpu::chunk_scan for the roles Settings indexes.
template <class Settings, class Step, class Read, class Out, class Args>
struct chunk_scanner
{
    // The last two axes of view as a tensor_view, positioned at the leading
    // index at
    template <class View, class Index>
    __device__ static auto matrix_slice(View view, Index at)
    {
        constexpr auto vs  = get_shape_c<View>{};
        constexpr auto n   = vs.lens.size();
        constexpr auto mat = make_shape(index_ints<vs.lens[n - 2], vs.lens[n - 1]>{},
                                        index_ints<vs.strides[n - 2], vs.strides[n - 1]>{});
        return view.with(view.data() + vs.index(at), mat);
    }

    // Row r and column c of a matrix view. Indexing a rank-1 slice keeps the
    // address affine in the loop variable, which lets adjacent loads merge
    template <class View>
    __device__ static auto row(View m, index_int r)
    {
        return slice_tensor(m, array<index_int, 2>{r, 0}, slice_axes<1>());
    }

    template <class View>
    __device__ static auto column(View m, index_int c)
    {
        return slice_tensor(m, array<index_int, 2>{0, c}, slice_axes<0>());
    }

    template <index_int I>
    using arg_t                       = decltype(declval<Args>()(arg_c<I>()));
    using value_type                  = typename arg_t<Settings::output_matrix::index>::type;
    using acc_t                       = typename acc_type<value_type>::type;
    static constexpr index_int chunks = Settings::chunks;
    static constexpr index_int rows   = Settings::rows;
    static constexpr index_int dk     = Settings::dk;
    static constexpr index_int dv     = Settings::dv;
    static constexpr index_int dvb    = Settings::dvb;
    static_assert(dv % dvb == 0, "the dv block must divide dv");
    // the blocks of the products: 16 wide where the axis allows it, and 32
    // rows of the update so a 32-lane wave's FMA tiles reach 4x4 without a
    // matrix instruction
    static constexpr index_int bm = rows % 16 == 0 ? 16 : rows;
    static constexpr index_int bn = dvb % 16 == 0 ? 16 : dvb;
    static constexpr index_int bd = dk % 32 == 0 ? 32 : (dk % 16 == 0 ? 16 : dk);
    // Under the read module the output and read matrix blocks stack as one
    // operand sharing the state tile: groups of g rows alternate between
    // them, the output block then the read block under a matrix instruction,
    // else single rows so a lane's tile holds an output row and its read row
    static constexpr index_int stacked = Settings::has_read ? 2 * bm : bm;
    static constexpr index_int g       = matmul_block<value_type, stacked, bn, dk>() > 0 ? bm : 1;
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
    static_assert((s_tile.elements() + u_tile.elements()) * sizeof(acc_t) <= Settings::max_lds,
                  "the state and u tiles must fit in LDS");
    using s_view = decltype(make_tensor_view<lds_memory_tag>(declval<acc_t*>(), s_tile));
    using u_view = decltype(make_tensor_view<lds_memory_tag>(declval<acc_t*>(), u_tile));

    Step step;
    Read read;
    Out out_fn;
    Args args;
    index idx;
    decltype(lead.multi(0)) base;
    index_int col0;
    s_view s;
    u_view u;

    // the matrix of argument I at this problem
    template <index_int I>
    __device__ auto mat() const
    {
        return matrix_slice(args(arg_c<I>()), base);
    }

    // the chunk i matrix of a family, whose argument stacks the chunks ahead
    // of the matrix axes: (lead..., chunks, m, n). A family a zero start
    // never reads at chunk 0 holds the chunks from 1 on.
    template <class Family>
    __device__ auto chunk(index_int i) const
    {
        constexpr auto fs     = get_shape_c<arg_t<Family::index>>{};
        constexpr index_int n = fs.lens.size();
        static_assert(n >= 3, "a family stacks its chunks ahead of the matrix axes");
        array<index_int, n> at{};
        for(index_int k = 0; k + 3 < n; k++)
            at[k] = base[k];
        at[n - 3] = i - (chunks - fs.lens[n - 3]);
        return matrix_slice(args(arg_c<Family::index>()), at);
    }

    // a stacked row as (group, output or read matrix, row within the group)
    static constexpr auto stack = make_shape(index_ints<bm / g, 2, g>{});
    __device__ static bool is_read_row(index_int r) { return stack.multi(r)[1] == 1; }
    __device__ static index_int src_row(index_int r)
    {
        if constexpr(Settings::has_read)
        {
            const auto m = stack.multi(r);
            return m[0] * g + m[2];
        }
        else
        {
            return r;
        }
    }

    __device__ void load_state() const
    {
        idx.local_stride(s_tile.elements(), [&](auto e) {
            auto t = s_tile.multi(e);
            if constexpr(Settings::has_state)
            {
                auto s0m = mat<Settings::s0_index>();
                s[t] = row(s0m, t[0])[col0 + t[1]];
            }
            else
                s[t] = 0;
        });
    }

    __device__ void store_state() const
    {
        auto som = mat<Settings::state_out_index>();
        idx.local_stride(s_tile.elements(), [&](auto e) {
            auto t                      = s_tile.multi(e);
            row(som, t[0])[col0 + t[1]] = s[t];
        });
    }

    // chunk 0 of a zero start reads nothing: u = update_input, o = output_base
    // (zero without an out module)
    __device__ void copy_chunk(index_int i) const
    {
        auto update_input = chunk<typename Settings::update_input>(i);
        auto output_base  = chunk<typename Settings::output_base>(i);
        auto output       = mat<Settings::output_index>();
        idx.local_stride(u_tile.elements(), [&](auto e) {
            auto t       = u_tile.multi(e);
            const auto r = t[0];
            const auto c = col0 + t[1];
            if constexpr(Settings::has_update_dot)
                u[t] = row(update_input, r)[c];
            acc_t ov = 0;
            if constexpr(Settings::has_out)
                ov = row(output_base, r)[c];
            row(output, i * rows + r)[c] = ov;
        });
    }

    // o_i = out(output_base_i, output_matrix_i S) and
    // u = read(update_input_i, read_matrix_i S) for chunk i
    __device__ void read_chunk(index_int i) const
    {
        auto read_matrix   = chunk<typename Settings::read_matrix>(i);
        auto output_matrix = chunk<typename Settings::output_matrix>(i);
        auto update_input  = chunk<typename Settings::update_input>(i);
        auto output_base   = chunk<typename Settings::output_base>(i);
        auto output        = mat<Settings::output_index>();
        if constexpr(Settings::has_update_dot and not Settings::has_read)
        {
            idx.local_stride(u_tile.elements(), [&](auto e) {
                auto t = u_tile.multi(e);
                u[t]   = row(update_input, t[0])[col0 + t[1]];
            });
        }
        idx.wave_stride(read_blocks.elements(), [&](auto b) {
            auto blk           = read_blocks.multi(b);
            const index_int r0 = blk[0] * bm;
            const index_int v0 = blk[1] * bn;
            const index_int c0 = col0 + v0;
            matmul<value_type, stacked, bn, dk>(
                [&](auto r, auto k) {
                    if constexpr(Settings::has_read)
                    {
                        if(is_read_row(r))
                            return row(read_matrix, r0 + src_row(r))[k];
                    }
                    return row(output_matrix, r0 + src_row(r))[k];
                },
                [&](auto k, auto j) { return row(s, k)[v0 + j]; },
                [&](auto x, auto r, auto j) {
                    const index_int sr = r0 + src_row(r);
                    if constexpr(Settings::has_read)
                    {
                        if(is_read_row(r))
                        {
                            row(u, sr)[v0 + j] = read(row(update_input, sr)[c0 + j], x);
                            return;
                        }
                    }
                    acc_t ov = x;
                    if constexpr(Settings::has_out)
                        ov = out_fn(row(output_base, sr)[c0 + j], x);
                    row(output, i * rows + sr)[c0 + j] = ov;
                });
        });
    }

    // S = step(S, [decay], up) for one state element, or up on a zero start;
    // decay is read only under the decay role
    template <class Decay>
    __device__ void transition(acc_t& sv, acc_t up, bool live, Decay decay) const
    {
        if(not live)
            sv = up;
        else if constexpr(Settings::has_decay)
            sv = step(sv, decay(), up);
        else
            sv = step(sv, up);
    }

    // S = step(S, [decay_i], update_matrix_i u or state_update_i) for chunk i
    __device__ void update_chunk(index_int i, bool live) const
    {
        auto update_matrix = chunk<typename Settings::update_matrix>(i);
        auto state_update  = chunk<typename Settings::state_update>(i);
        auto decay         = chunk<typename Settings::decay>(i);
        if constexpr(Settings::has_update_dot)
        {
            idx.wave_stride(update_blocks.elements(), [&](auto b) {
                auto blk           = update_blocks.multi(b);
                const index_int d0 = blk[0] * bd;
                const index_int v0 = blk[1] * bn;
                matmul<value_type, bd, bn, rows>(
                    [&](auto d, auto r) { return column(update_matrix, r)[d0 + d]; },
                    [&](auto r, auto j) { return row(u, r)[v0 + j]; },
                    [&](auto x, auto d, auto j) {
                        transition(row(s, d0 + d)[v0 + j], x, live, [&] {
                            return row(decay, d0 + d)[col0 + v0 + j];
                        });
                    });
            });
        }
        else
        {
            idx.local_stride(s_tile.elements(), [&](auto e) {
                auto t = s_tile.multi(e);
                transition(s[t], row(state_update, t[0])[col0 + t[1]], live, [&] {
                    return row(decay, t[0])[col0 + t[1]];
                });
            });
        }
    }

    __device__ void run() const
    {
        load_state();
        __syncthreads();
        for(index_int i = 0; i < chunks; i++)
        {
            // chunk 0 of a zero start skips the reads and the step: S = update
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
    __shared__ uninitialized_buffer<typename scanner::acc_t, scanner::s_tile.elements(), 16> s_buf;
    __shared__ uninitialized_buffer<typename scanner::acc_t, scanner::u_tile.elements(), 16> u_buf;
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
