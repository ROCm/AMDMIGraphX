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
#include <migraphx/kernels/algorithm.hpp>
#include <migraphx/kernels/slice.hpp>
#include <migraphx/kernels/copy.hpp>
#include <migraphx/kernels/reduce.hpp>
#include <migraphx/kernels/uninitialized_buffer.hpp>

namespace migraphx {

template <class Output, class F>
__device__ void per_block_pooling_reduce(index idx, Output output, F f)
{
    constexpr auto nelements = get_shape_c<Output>{}.elements();
    idx.local_stride(nelements, [&](auto i) {
        auto out_idx = get_shape_c<Output>{}.multi(i);
        auto slicer  = [](auto input) { return reduce_slice<decltype(output)>(input, 0); };
        auto r       = reduce::lane::make(idx, slicer);
        r.outer([&] { output[out_idx] = f(out_idx, r); });
    });
}

template <class KernelLens, class SpatialLens, class Output, class Input, class Weights>
__device__ void
channelwise_conv(KernelLens, SpatialLens, Output output, Input x, Weights w)
{
    constexpr index_int kernel_total  = KernelLens{}.product();
    constexpr index_int spatial_total = SpatialLens{}.product();

    constexpr index_int N     = get_shape_c<Output>{}.lens[0];
    constexpr index_int C_out = get_shape_c<Output>{}.lens[1];
    constexpr index_int C_in  = get_shape_c<Input>{}.lens[1];

    constexpr auto smem_shape  = make_packed_shape(make_slice(get_shape_c<Input>{},
        [](auto, auto i, auto) { return i >= 2; }));
    constexpr auto wregs_shape = make_packed_shape(make_slice(get_shape_c<Weights>{},
        [](auto, auto i, auto) { return i >= 2; }));

    constexpr auto out_nc = make_shape(index_ints<N, C_out>{});
    constexpr auto co_cin = make_shape(index_ints<C_out / C_in, C_in>{});
    constexpr auto in_nc  = make_shape(index_ints<N, C_in>{});

    using T = typename Output::type;
    __shared__ uninitialized_buffer<T, spatial_total> smem;

    auto idx          = make_index();
    auto keep_spatial = [](auto, auto i, auto) { return i >= 2; };

    slice_schedule<per_block>(idx, keep_spatial)(output)([&](auto out_ch) {
        auto nc_multi = out_nc.multi(idx.group);
        auto n        = nc_multi[0];
        auto co       = nc_multi[1];
        auto c_in     = co_cin.multi(co)[1];

        auto x_ch = slice_tensor(x, in_nc.index(make_array(n, c_in)), keep_spatial);
        auto w_ch = slice_tensor(w, co, keep_spatial);

        // Phase 1: copy input channel into shared memory
        auto smem_input = make_tensor_view(smem.data(), smem_shape);
        local_tensor_copy(idx, x_ch, smem_input);

        // Phase 2: copy weights into registers
        array<T, kernel_total> wregs_arr;
        auto wregs = make_tensor_view(wregs_arr.begin(), wregs_shape);
        copy(w_ch.begin(), w_ch.end(), wregs.begin());

        __syncthreads();

        // Phase 3: sliding window multiply-reduce
        per_block_pooling_reduce(idx, out_ch, [&](auto out_idx, auto r) {
            return r.reduce(op::sum{}, T{0}, [&](auto ki) {
                auto k_multi = wregs_shape.multi(ki);
                return smem_input[out_idx + k_multi] * wregs[k_multi];
            })(reduce::make_indices(_c<kernel_total>));
        });
    });
}

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_CHANNELWISE_CONV_HPP
