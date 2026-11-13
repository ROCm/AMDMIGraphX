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
#ifndef MIGRAPHX_GUARD_KERNELS_TILED_CONV_HPP
#define MIGRAPHX_GUARD_KERNELS_TILED_CONV_HPP

#include <migraphx/kernels/spatial_tiler.hpp>

namespace migraphx {

// Input extent one output reads per spatial dim: the dilated filter. dilation has a step for
// every dim of the (K, C, k...) weights.
template <class Weights, class Dilation>
constexpr auto conv_reach(Dilation)
{
    constexpr index_int nspatial = get_shape_c<Weights>{}.lens.size() - 2;
    return generate_const_array<index_int>(_c<nspatial>, [](auto i) {
        constexpr auto k = get_shape_c<Weights>{}.lens;
        return index_c<(k[i + 2] - 1) * Dilation{}[i + 2] + 1>;
    });
}

// Halo rows a run of NRows output rows reads with Taps filter rows: rows step by S and taps
// by D
template <index_int NRows, index_int Taps, index_int S, index_int D>
constexpr index_int run_halo_rows()
{
    return (NRows - 1) * S + (Taps - 1) * D + 1;
}

// Whether halo row i of a run feeds some output row r and filter tap t: i = r*S + t*D
template <index_int NRows, index_int Taps, index_int S, index_int D>
constexpr bool halo_row_used(index_int i)
{
    for(index_int r = 0; r < NRows and r * S <= i; r++)
    {
        auto offset = i - r * S;
        if(offset % D == 0 and offset / D < Taps)
            return true;
    }
    return false;
}

// Apply the fused pointwise op to row r of the accumulators and store its channel vector.
// The conv result is passed to f at the conv's own precision (ConvType) so the post-op
// matches the reference; f's result converts to the output type at the store.
template <index_int ChannelVec,
          class ConvType,
          class F,
          class Output,
          class Pack,
          class Pos,
          class Acc,
          class R>
__device__ void tiled_conv_store(F f, Output out_ch, Pack xs_pack, Pos pos, const Acc& acc, R r)
{
    using type = typename Output::type;
    constexpr auto acc_shape =
        make_shape(index_ints<decltype(acc.size()){} / ChannelVec, ChannelVec>{});
    xs_pack([&](auto... xs) {
        pack(load_channels<ChannelVec>(xs, pos)...)([&](auto... xv) {
            array<type, ChannelVec> out;
            repeat(_c<ChannelVec>, [&](auto v) {
                out[v] = f(static_cast<ConvType>(acc[acc_shape.index(array<index_int, 2>{r, v})]),
                           xv[v]...);
            });
            store_channels<ChannelVec>(out_ch, pos, out);
        });
    });
}

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_TILED_CONV_HPP
