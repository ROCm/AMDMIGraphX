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

template <class TileLens, index_int NTiles, class Output, class Input, class Weights>
__device__ void channelwise_conv(TileLens, Output output, Input x, Weights w)
{
    auto idx   = make_index();
    auto tiler = make_spatial_tiler<NTiles>(idx, TileLens{}, get_shape_c<Output>{});

    __shared__ decltype(tiler.template shared_allocate<Input>()) smem;

    auto x_ch   = tiler.copy(x, smem);
    auto w_ch   = tiler.slice(w);
    auto out_ch = tiler.slice(output);

    using T = typename Output::type;
    array<T, decltype(w_ch.get_shape().elements()){}> wregs_arr;
    auto wregs = make_tensor_view(wregs_arr.begin(), make_packed_shape(w_ch.get_shape()));
    copy(w_ch.begin(), w_ch.end(), wregs.begin());

    __syncthreads();

    tiler.for_each([&](auto out_pos, auto out_multi) {
        T acc = 0;
        repeat(wregs.get_shape().elements(), [&](auto ki) {
            auto k_multi = wregs.get_shape().multi(ki);
            acc += x_ch[out_multi + k_multi] * wregs[k_multi];
        });
        out_ch[out_pos] = acc;
    });
}

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_CHANNELWISE_CONV_HPP
