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
#ifndef MIGRAPHX_GUARD_KERNELS_SPATIAL_TILER_HPP
#define MIGRAPHX_GUARD_KERNELS_SPATIAL_TILER_HPP

#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/algorithm.hpp>
#include <migraphx/kernels/slice.hpp>
#include <migraphx/kernels/copy.hpp>
#include <migraphx/kernels/uninitialized_buffer.hpp>

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

template <index_int NTiles, class TileLens, class OutputShape>
struct spatial_tiler
{
    static constexpr auto keep_spatial = [](auto, auto i, auto) { return i >= 2; };

    // Full-rank tile lens: [1, 1, TileH, TileW]
    static constexpr auto tile_lens = join(index_ints<1, 1>{}, TileLens{});

    // Output region per block: tile with last dim scaled by NTiles
    static constexpr auto output_lens = return_array_c([] {
        auto result       = decltype(tile_lens){};
        constexpr auto nd = result.size();
        array<index_int, nd> r;
        for(index_int i = 0; i < nd; i++)
            r[i] = result[i];
        r[nd - 1] *= NTiles;
        return r;
    });

    static constexpr auto out_spatial_lens = make_slice(OutputShape{}, keep_spatial).lens;

    static constexpr auto tiles_per_dim =
        transform(out_spatial_lens, output_lens, [](auto o, auto t) { return (o + t - 1) / t; });

    static constexpr auto block_lens  = return_array_c([] {
        constexpr auto tpd     = decltype(tiles_per_dim){};
        constexpr index_int nd = tpd.size();
        constexpr auto olens   = OutputShape{}.lens;
        array<index_int, nd> result;
        for(index_int i = 0; i < nd; i++)
            result[i] = tpd[i];
        result[0] = olens[0];
        result[1] = olens[1];
        return result;
    });
    static constexpr auto block_shape = make_shape(block_lens);

    static constexpr auto output_shape      = make_shape(output_lens);
    static constexpr index_int output_total = output_lens.product();
    static constexpr index_int tiles_total  = tiles_per_dim.product();
    static constexpr index_int NDIM         = out_spatial_lens.size();

    static constexpr bool is_padded = [] {
        return (out_spatial_lens != tiles_per_dim * output_lens);
    }();

    index idx;
    array<index_int, NDIM> tile_origin;

    // Compute halo lens for a given input shape: output_lens + (input_spatial - output_spatial)
    template <class InputShape>
    static constexpr auto halo_lens_for()
    {
        constexpr auto input_spatial = make_slice(InputShape{}, keep_spatial).lens;
        constexpr auto halo_extra =
            transform(input_spatial, out_spatial_lens, [](auto is, auto os) { return is - os; });
        return transform(output_lens, halo_extra, [](auto o, auto h) { return o + h; });
    }

    // Type for shared memory allocation
    template <class Input>
    __device__ auto shared_allocate() const
    {
        using T                          = typename Input::type;
        constexpr auto hl                = halo_lens_for<get_shape_c<Input>>();
        constexpr index_int halo_total_v = hl.product();
        return uninitialized_buffer<T, halo_total_v>{};
    }

    // Slice a tensor to per-channel spatial view
    template <class Tensor>
    __device__ auto slice(Tensor t) const
    {
        constexpr auto n_ch = nslices(get_shape_c<Tensor>{}, keep_spatial);
        return slice_tensor(t, (idx.group / tiles_total) % index_int{n_ch}, keep_spatial);
    }

    // Copy input halo tile into shared memory, return tensor_view over smem
    template <class Input, class Smem>
    __device__ auto copy(Input input, Smem& smem) const
    {
        using T                          = typename Input::type;
        constexpr auto hl                = halo_lens_for<get_shape_c<Input>>();
        constexpr auto halo_shape        = make_shape(hl);
        constexpr index_int halo_total_v = hl.product();
        constexpr auto input_spatial     = make_slice(get_shape_c<Input>{}, keep_spatial).lens;

        constexpr auto n_out  = nslices(OutputShape{}, keep_spatial);
        constexpr auto n_in   = nslices(get_shape_c<Input>{}, keep_spatial);
        constexpr auto groups = n_out / n_in;
        auto channel_idx      = idx.group / tiles_total;
        auto input_ch =
            slice_tensor(input, (channel_idx / index_int{groups}) % index_int{n_in}, keep_spatial);

        idx.local_stride(_c<halo_total_v>, [&](auto i) {
            auto halo_multi = halo_shape.multi(index_int{i});
            auto src_pos    = tile_origin + halo_multi;
            if constexpr(is_padded)
                smem[i] = in_bounds(src_pos, input_spatial) ? T{input_ch[src_pos]} : T{0};
            else
                smem[i] = input_ch[src_pos];
        });

        return make_tensor_view(smem.data(), halo_shape);
    }

    // Iterate over output tile positions with bounds checking
    template <class F>
    __device__ void for_each(F f) const
    {
        idx.local_stride(_c<output_total>, [&](auto j) {
            auto out_multi = output_shape.multi(index_int{j});
            auto out_pos   = tile_origin + out_multi;
            if constexpr(is_padded)
            {
                if(not in_bounds(out_pos, out_spatial_lens))
                    return;
            }
            f(out_pos, out_multi);
        });
    }
};

template <index_int NTiles, class TileLens, class OutputShape>
__device__ auto make_spatial_tiler(index idx, TileLens, OutputShape)
{
    using tiler_type = spatial_tiler<NTiles, TileLens, OutputShape>;

    auto block_multi = tiler_type::block_shape.multi(idx.group);
    auto tile_origin = generate_array<index_int>(_c<tiler_type::NDIM>, [&](auto d) -> index_int {
        if constexpr(d < 2)
            return 0;
        else
            return block_multi[d] * tiler_type::output_lens[d];
    });

    return tiler_type{idx, tile_origin};
}

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_SPATIAL_TILER_HPP
