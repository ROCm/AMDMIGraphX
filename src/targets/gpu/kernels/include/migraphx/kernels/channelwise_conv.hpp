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
 *
 */
#ifndef MIGRAPHX_GUARD_KERNELS_CHANNELWISE_CONV_HPP
#define MIGRAPHX_GUARD_KERNELS_CHANNELWISE_CONV_HPP

#include <migraphx/kernels/spatial_tiler.hpp>
#include <migraphx/kernels/algorithm.hpp>
#include <migraphx/kernels/copy.hpp>

namespace migraphx {

// Accumulate ChannelVec channels of NRows consecutive output rows starting at out_multi.
// Channels move as vectors, and a filter column's taps reuse the overlapping halo rows from
// registers instead of re-reading them from LDS. Values stay in the input type so half
// inputs use mixed-precision FMAs with fp32 accumulation.
template <index_int NRows, index_int ChannelVec, class Halo, class Weights, class Pos>
__device__ auto channelwise_conv_accumulate(Halo x_ch, Weights wregs, Pos out_multi)
{
    using type               = typename Halo::type;
    constexpr auto k_shape   = get_shape_c<Weights>{};
    constexpr index_int kh   = k_shape.lens[2];
    constexpr auto col_shape = make_shape(return_array_c([] {
        auto result = get_shape_c<Weights>{}.lens;
        result[0]   = 1;
        result[2]   = 1;
        return result;
    }));
    constexpr auto acc_shape = make_shape(index_ints<NRows, ChannelVec>{});
    constexpr auto col_rows  = make_shape(index_ints<NRows + kh - 1, ChannelVec>{});
    array<float, acc_shape.elements()> acc{};
    repeat(col_shape.elements(), [&](auto j) {
        auto col_multi = col_shape.multi(j);
        array<type, col_rows.elements()> col;
        repeat(_c<NRows + kh - 1>, [&](auto i) {
            auto pos = out_multi + col_multi;
            pos[2] += i;
            auto values = load_channels<ChannelVec>(x_ch, pos);
            repeat(_c<ChannelVec>,
                   [&](auto v) { col[col_rows.index(array<index_int, 2>{i, v})] = values[v]; });
        });
        repeat(_c<kh>, [&](auto t) {
            repeat(_c<ChannelVec>, [&](auto v) {
                auto k_multi = col_multi;
                k_multi[0]   = v;
                k_multi[2]   = t;
                auto wt      = static_cast<float>(wregs[k_multi]);
                repeat(_c<NRows>, [&](auto r) {
                    acc[acc_shape.index(array<index_int, 2>{r, v})] +=
                        static_cast<float>(col[col_rows.index(array<index_int, 2>{r + t, v})]) * wt;
                });
            });
        });
    });
    return acc;
}

// Apply the fused pointwise op to row r of the accumulators and store its channel vector
template <index_int ChannelVec, class F, class Output, class Pack, class Pos, class Acc, class R>
__device__ void
channelwise_conv_store(F f, Output out_ch, Pack xs_pack, Pos pos, const Acc& acc, R r)
{
    using type = typename Output::type;
    constexpr auto acc_shape =
        make_shape(index_ints<decltype(acc.size()){} / ChannelVec, ChannelVec>{});
    xs_pack([&](auto... xs) {
        pack(load_channels<ChannelVec>(xs, pos)...)([&](auto... xv) {
            array<type, ChannelVec> out;
            repeat(_c<ChannelVec>, [&](auto v) {
                out[v] =
                    f(static_cast<type>(acc[acc_shape.index(array<index_int, 2>{r, v})]), xv[v]...);
            });
            store_channels<ChannelVec>(out_ch, pos, out);
        });
    });
}

template <class TileLens,
          index_int NTiles,
          index_int ChannelTile = 1,
          index_int NRows       = 1,
          index_int ChannelVec  = 1,
          class Padding,
          class F,
          class Output,
          class Input,
          class Weights,
          class... Inputs>
__device__ void
channelwise_conv(TileLens, Padding, F f, Output output, Input x, Weights w, Inputs... inputs)
{
    auto idx   = make_index();
    auto tiler = make_spatial_tiler<NTiles, ChannelTile, NRows, ChannelVec>(
        idx, TileLens{}, get_shape_c<Output>{}, Padding{});

    __shared__ decltype(tiler.template shared_allocate<Input>()) smem;

    auto x_ch    = tiler.copy(x, smem);
    auto w_ch    = tiler.slice_weights(w);
    auto out_ch  = tiler.slice(output);
    auto xs_pack = pack(tiler.slice(inputs)...);

    using type = typename Output::type;
    array<type, decltype(w_ch.get_shape().elements()){}> wregs_arr;
    auto wregs = make_tensor_view(wregs_arr.begin(), make_packed_shape(w_ch.get_shape()));
    copy(w_ch.begin(), w_ch.end(), wregs.begin());

    __syncthreads();

    tiler.for_each_run([&](auto out_pos, auto out_multi) {
        auto acc = channelwise_conv_accumulate<NRows, ChannelVec>(x_ch, wregs, out_multi);
        repeat(_c<NRows>, [&](auto r) {
            auto pos = out_pos;
            pos[2] += r;
            if constexpr(decltype(tiler)::is_padded())
            {
                if(not tiler.contains(pos))
                    return;
            }
            channelwise_conv_store<ChannelVec>(f, out_ch, xs_pack, pos, acc, r);
        });
    });
}

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_CHANNELWISE_CONV_HPP
