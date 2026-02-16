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

#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/ops.hpp>
#include <migraphx/kernels/reduce.hpp>
#include <migraphx/kernels/pooling.hpp>

namespace migraphx {

template <class Algo, class KernelLens, class Output, class Input1, class Input2>
__device__ void channelwise_conv(KernelLens kernel_lens, Output output, Input1 x, Input2 w)
{
    constexpr index_int NS           = array_size(KernelLens{});
    constexpr index_int NDIM         = 2 + 2 * NS;
    constexpr index_int kernel_total = KernelLens{}.product();

    pooling_reduce<Algo, 1>(output, [&](auto out_idx, auto r) {
        auto result = r.reduce(op::sum{}, 0, [&](auto ki) {
            auto kmulti    = kernel_lens.multi(ki);
            auto bcast_idx = generate_array<index_int>(
                _c<NDIM>, [&](auto d) -> index_int {
                    if constexpr(d < 2)
                        return out_idx[d];
                    else if constexpr(d < 2 + NS)
                        return kmulti[d - _c<2>];
                    else
                        return out_idx[d - _c<NS>] + kmulti[d - _c<2 + NS>];
                });
            return x[bcast_idx] * w[bcast_idx];
        })(reduce::make_indices(_c<kernel_total>));
        return result;
    });
}

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_CHANNELWISE_CONV_HPP
