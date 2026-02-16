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
#include <migraphx/kernels/slice.hpp>
#include <migraphx/kernels/uninitialized_buffer.hpp>

namespace migraphx {

template <class KernelLens, class SpatialLens, class Output, class Input1, class Input2>
__device__ void
channelwise_conv(KernelLens kernel_lens, SpatialLens, Output output, Input1 x, Input2 w)
{
    constexpr index_int NS            = array_size(KernelLens{});
    constexpr index_int kernel_total  = KernelLens{}.product();
    constexpr index_int spatial_total = SpatialLens{}.product();
    constexpr index_int product_total = kernel_total * spatial_total;

    constexpr auto out_spatial_lens = return_array_c([] {
        constexpr auto kl      = KernelLens{};
        constexpr auto sl      = SpatialLens{};
        constexpr index_int ns = array_size(KernelLens{});
        array<index_int, ns> result;
        for(index_int i = 0; i < ns; i++)
            result[i] = sl[i] - kl[i] + 1;
        return result;
    });
    constexpr index_int out_spatial_total = out_spatial_lens.product();

    constexpr auto prod_lens = return_array_c([] {
        constexpr auto kl      = KernelLens{};
        constexpr auto sl      = SpatialLens{};
        constexpr index_int ns = array_size(KernelLens{});
        array<index_int, 2 * ns> result;
        for(index_int i = 0; i < ns; i++)
            result[i] = kl[i];
        for(index_int i = 0; i < ns; i++)
            result[ns + i] = sl[i];
        return result;
    });
    constexpr auto prod_strides = calculate_strides(prod_lens);

    using T = typename Output::type;
    __shared__ uninitialized_buffer<T, product_total> smem;

    auto idx            = make_index();
    auto keep_non_batch = [](auto, auto i, auto) { return i >= 2; };

    slice_schedule<per_block>(idx, keep_non_batch)(x, w, output)(
        [&](auto x_ch, auto w_ch, auto out_ch) {
            // Phase 1: elementwise multiply into shared memory
            idx.local_stride(_c<product_total>, [&](auto i) {
                auto prod_multi = prod_lens.multi(i);
                auto ch_idx =
                    generate_array<index_int>(_c<2 + 2 * NS>, [&](auto d) -> index_int {
                        if constexpr(d < 2)
                            return 0;
                        else
                            return prod_multi[d - _c<2>];
                    });
                smem[i] = x_ch[ch_idx] * w_ch[ch_idx];
            });

            __syncthreads();

            // Phase 2: sliding window reduce from shared memory
            idx.local_stride(_c<out_spatial_total>, [&](auto j) {
                auto out_spatial = out_spatial_lens.multi(j);
                T acc            = 0;
                for(index_int ki = 0; ki < kernel_total; ki++)
                {
                    auto k_multi  = kernel_lens.multi(ki);
                    auto smem_idx = generate_array<index_int>(
                        _c<2 * NS>, [&](auto d) -> index_int {
                            if constexpr(d < NS)
                                return k_multi[d];
                            else
                                return out_spatial[d - _c<NS>] + k_multi[d - _c<NS>];
                        });
                    acc += smem[smem_idx.dot(prod_strides)];
                }

                auto out_idx =
                    generate_array<index_int>(_c<2 + NS>, [&](auto d) -> index_int {
                        if constexpr(d < 2)
                            return 0;
                        else
                            return out_spatial[d - _c<2>];
                    });
                out_ch[out_idx] = acc;
            });
        });
}

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_CHANNELWISE_CONV_HPP
