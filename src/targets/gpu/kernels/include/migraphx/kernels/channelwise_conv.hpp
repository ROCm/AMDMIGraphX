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
#include <migraphx/kernels/array.hpp>

namespace migraphx {

template <class KernelLens, class SpatialLens, class Output, class Input1, class Input2>
__device__ void channelwise_conv(KernelLens kernel_lens,
                                 SpatialLens,
                                 Output output,
                                 Input1 x,
                                 Input2 w)
{
    constexpr index_int NS            = array_size(KernelLens{});
    constexpr index_int kernel_total  = KernelLens{}.product();
    constexpr index_int spatial_total = SpatialLens{}.product();
    constexpr index_int product_total = kernel_total * spatial_total;

    constexpr auto out_spatial_lens = return_array_c([] {
        constexpr auto kl          = KernelLens{};
        constexpr auto sl          = SpatialLens{};
        constexpr index_int ns     = array_size(KernelLens{});
        array<index_int, ns> result;
        for(index_int i = 0; i < ns; i++)
            result[i] = sl[i] - kl[i] + 1;
        return result;
    });
    constexpr index_int out_spatial_total = out_spatial_lens.product();

    constexpr auto prod_lens = return_array_c([] {
        constexpr auto kl              = KernelLens{};
        constexpr auto sl              = SpatialLens{};
        constexpr index_int ns         = array_size(KernelLens{});
        array<index_int, 2 * ns> result;
        for(index_int i = 0; i < ns; i++)
            result[i] = kl[i];
        for(index_int i = 0; i < ns; i++)
            result[ns + i] = sl[i];
        return result;
    });
    constexpr auto smem_shape = make_shape(prod_lens);

    using T = typename Output::type;
    __shared__ T smem[product_total];

    auto idx = make_index();

    index_int C = output.get_shape().lens[1];
    auto n      = idx.group / C;
    auto c      = idx.group % C;

    // Phase 1: elementwise multiply into shared memory
    for(index_int i = idx.local; i < product_total; i += idx.nlocal())
    {
        auto prod_multi = prod_lens.multi(i);
        auto bcast_idx =
            generate_array<index_int>(_c<2 + 2 * NS>, [&](auto d) -> index_int {
                if constexpr(d == 0)
                    return n;
                else if constexpr(d == 1)
                    return c;
                else
                    return prod_multi[d - _c<2>];
            });
        smem[i] = x[bcast_idx] * w[bcast_idx];
    }

    __syncthreads();

    auto smem_view = make_tensor_view(&smem[0], smem_shape);

    // Phase 2: sliding window reduce from shared memory
    for(index_int j = idx.local; j < out_spatial_total; j += idx.nlocal())
    {
        auto out_spatial = out_spatial_lens.multi(j);
        T acc            = 0;
        for(index_int ki = 0; ki < kernel_total; ki++)
        {
            auto k_multi  = kernel_lens.multi(ki);
            auto smem_idx = generate_array<index_int>(_c<2 * NS>, [&](auto d) -> index_int {
                if constexpr(d < NS)
                    return k_multi[d];
                else
                    return out_spatial[d - _c<NS>] + k_multi[d - _c<NS>];
            });
            acc += smem_view[smem_idx];
        }

        auto out_idx = generate_array<index_int>(_c<2 + NS>, [&](auto d) -> index_int {
            if constexpr(d == 0)
                return n;
            else if constexpr(d == 1)
                return c;
            else
                return out_spatial[d - _c<2>];
        });
        output[out_idx] = acc;
    }
}

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_CHANNELWISE_CONV_HPP
