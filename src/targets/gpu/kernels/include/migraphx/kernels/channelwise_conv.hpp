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

template <class TileLens,
          index_int NTiles,
          index_int ChannelTile = 1,
          index_int NRows       = 1,
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
    auto tiler = make_spatial_tiler<NTiles, ChannelTile, NRows>(
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

    // Each lane computes NRows consecutive output rows, so a filter column's taps reuse
    // the overlapping halo rows from registers instead of re-reading them from LDS.
    constexpr auto k_shape   = get_shape_c<decltype(wregs)>{};
    constexpr index_int kh   = k_shape.lens[2];
    constexpr auto col_shape = make_shape(return_array_c([] {
        auto result = get_shape_c<decltype(wregs)>{}.lens;
        result[2]   = 1;
        return result;
    }));
    tiler.for_each_run([&](auto out_pos, auto out_multi) {
        array<float, NRows> acc{};
        repeat(col_shape.elements(), [&](auto j) {
            auto col_multi = col_shape.multi(j);
            array<float, NRows + kh - 1> col;
            repeat(_c<NRows + kh - 1>, [&](auto i) {
                auto pos = out_multi + col_multi;
                pos[2] += i;
                col[i] = static_cast<float>(x_ch[pos]);
            });
            repeat(_c<kh>, [&](auto t) {
                auto k_multi = col_multi;
                k_multi[2]   = t;
                auto wt      = static_cast<float>(wregs[k_multi]);
                repeat(_c<NRows>, [&](auto r) { acc[r] += col[r + t] * wt; });
            });
        });
        repeat(_c<NRows>, [&](auto r) {
            auto pos = out_pos;
            pos[2] += r;
            if constexpr(decltype(tiler)::is_padded())
            {
                if(not tiler.contains(pos))
                    return;
            }
            xs_pack([&](auto... xs) { out_ch[pos] = f(static_cast<type>(acc[r]), xs[pos]...); });
        });
    });
}

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_CHANNELWISE_CONV_HPP
