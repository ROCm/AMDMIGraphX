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
#include <migraphx/kernels/pooling.hpp>

namespace migraphx {

template <class Pos, class Lens>
constexpr bool in_bounds(Pos pos, Lens lens)
{
    for(index_int d = 0; d < pos.size(); d++)
    {
        if(pos[d] >= lens[d])
            return false;
    }
    return true;
}

// TileLens:   thread block tile (determines block_size = product)
// OutputLens: output region per block (>= TileLens, multiple along W)
template <class TileLens, class OutputLens, class Output, class Input, class Weights>
__device__ void channelwise_conv(TileLens, OutputLens, Output output, Input x, Weights w)
{
    auto keep_spatial = [](auto, auto i, auto) { return i >= 2; };

    constexpr index_int N     = get_shape_c<Output>{}.lens[0];
    constexpr index_int C_out = get_shape_c<Output>{}.lens[1];
    constexpr index_int C_in  = get_shape_c<Input>{}.lens[1];

    constexpr auto spatial_lens = make_slice(get_shape_c<Input>{}, keep_spatial).lens;
    constexpr auto kernel_lens  = make_slice(get_shape_c<Weights>{}, keep_spatial).lens;
    constexpr auto wregs_shape =
        make_packed_shape(make_slice(get_shape_c<Weights>{}, keep_spatial));

    constexpr index_int kernel_total = kernel_lens.product();

    constexpr auto out_nc = make_shape(index_ints<N, C_out>{});
    constexpr auto co_cin = make_shape(index_ints<C_out / C_in, C_in>{});
    constexpr auto in_nc  = make_shape(index_ints<N, C_in>{});

    // Full-rank output region per block
    constexpr auto output_lens = join(index_ints<1, 1>{}, OutputLens{});
    constexpr auto halo_lens =
        transform(output_lens, kernel_lens, [](auto o, auto k) { return o + k - 1; });
    constexpr auto out_spatial_lens =
        transform(spatial_lens, kernel_lens, [](auto s, auto k) { return s - k + 1; });
    constexpr auto tiles_per_dim =
        transform(out_spatial_lens, output_lens, [](auto o, auto t) { return (o + t - 1) / t; });

    constexpr auto output_shape      = make_shape(output_lens);
    constexpr auto halo_shape        = make_shape(halo_lens);
    constexpr index_int halo_total   = halo_lens.product();
    constexpr index_int output_total = output_lens.product();

    // Block shape: [N, C_out, tiles_h, tiles_w]
    constexpr auto block_lens  = return_array_c([] {
        constexpr auto tpd     = decltype(tiles_per_dim){};
        constexpr index_int nd = tpd.size();
        array<index_int, nd> result;
        for(index_int i = 0; i < nd; i++)
            result[i] = tpd[i];
        result[0] = N;
        result[1] = C_out;
        return result;
    });
    constexpr auto block_shape = make_shape(block_lens);

    using T = typename Output::type;
    __shared__ uninitialized_buffer<T, halo_total> smem;

    auto idx = make_index();

    // Decompose block index
    auto block_multi = block_shape.multi(idx.group);
    auto n           = block_multi[0];
    auto co          = block_multi[1];
    auto c_in        = co_cin.multi(co)[1];

    auto x_ch   = slice_tensor(x, in_nc.index(make_array(n, c_in)), keep_spatial);
    auto w_ch   = slice_tensor(w, co, keep_spatial);
    auto out_ch = slice_tensor(output, out_nc.index(make_array(n, co)), keep_spatial);

    constexpr index_int NDIM = spatial_lens.size();
    auto tile_origin         = generate_array<index_int>(_c<NDIM>, [&](auto d) -> index_int {
        if constexpr(d < 2)
            return 0;
        else
            return block_multi[d] * output_lens[d];
    });

    // Phase 1: load halo tile into shared memory with bounds checking
    auto smem_view = make_tensor_view(smem.data(), halo_shape);
    idx.local_stride(_c<halo_total>, [&](auto i) {
        auto halo_multi = halo_shape.multi(index_int{i});
        auto src_pos    = tile_origin + halo_multi;
        smem[i]         = in_bounds(src_pos, spatial_lens) ? T{x_ch[src_pos]} : T{0};
    });

    // Phase 2: copy weights into registers
    array<T, kernel_total> wregs_arr;
    auto wregs = make_tensor_view(wregs_arr.begin(), wregs_shape);
    copy(w_ch.begin(), w_ch.end(), wregs.begin());

    __syncthreads();

    // Phase 3: compute output region (each thread handles output_total / block_size elements)
    idx.local_stride(_c<output_total>, [&](auto j) {
        auto out_multi = output_shape.multi(index_int{j});
        auto out_pos   = tile_origin + out_multi;
        if(not in_bounds(out_pos, out_spatial_lens))
            return;

        T acc = 0;
        for(index_int ki = 0; ki < kernel_total; ki++)
        {
            auto k_multi = wregs_shape.multi(ki);
            acc += smem_view[out_multi + k_multi] * wregs[k_multi];
        }

        out_ch[out_pos] = acc;
    });
}

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_CHANNELWISE_CONV_HPP
