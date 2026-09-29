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
#include <migraphx/kernels/permutation.hpp>
#include <migraphx/kernels/uninitialized_buffer.hpp>

namespace migraphx {

template <index_int... Ps>
constexpr bool has_nonzero(index_ints<Ps...>)
{
    return ((Ps != 0) or ...);
}

// Tiles the spatial dims of an (N, C, spatial...) output across workgroups. Each block
// covers ChannelTile channels of one batch; lanes and the shared-memory halo follow the
// memory order of the tensors, so channels-last layouts get contiguous accesses.
template <index_int NTiles,
          class TileLens,
          class OutputShape,
          class Padding         = index_ints<0>,
          index_int ChannelTile = 1>
struct spatial_tiler
{
    static_assert(OutputShape{}.lens[1] % ChannelTile == 0,
                  "Channel tile must divide the output channels");

    static constexpr auto keep_spatial()
    {
        return [](auto, auto i, auto) { return i >= 2; };
    }

    // Tile owned by the lanes of a block: (1, ChannelTile, tile spatial...)
    static constexpr auto lane_lens() { return join(index_ints<1, ChannelTile>{}, TileLens{}); }

    // Output region per block: lane tile with last dim scaled by NTiles
    static constexpr auto output_lens()
    {
        return return_array_c([] {
            auto result       = lane_lens();
            constexpr auto nd = result.size();
            result[nd - 1] *= NTiles;
            return result;
        });
    }

    static constexpr auto out_spatial_lens()
    {
        return make_slice(OutputShape{}, keep_spatial()).lens;
    }

    // Extent of the block's channel group over the whole output: (1, ChannelTile, spatial...)
    static constexpr auto region_lens()
    {
        return return_array_c([] {
            auto result = out_spatial_lens();
            result[1]   = ChannelTile;
            return result;
        });
    }

    static constexpr auto tiles_per_dim()
    {
        return transform(
            out_spatial_lens(), output_lens(), [](auto o, auto t) { return (o + t - 1) / t; });
    }

    static constexpr index_int tiles_total() { return tiles_per_dim().product(); }
    static constexpr index_int channel_tiles() { return OutputShape{}.lens[1] / ChannelTile; }
    static constexpr auto ndim() { return out_spatial_lens().size(); }

    // Memory order of the output; lanes are assigned in this order
    static constexpr auto permutation() { return find_permutation(OutputShape{}); }

    static constexpr auto get_padding()
    {
        if constexpr(Padding{}.size() < 2)
        {
            auto pre = transform(TileLens{}, [](auto) { return index_c<0>; });
            return join(pre, pre);
        }
        else
        {
            return Padding{};
        }
    }

    // Left (begin) padding per dim: (0, 0, left_h, left_w)
    static constexpr auto left_padding()
    {
        constexpr auto p  = get_padding();
        constexpr auto ns = p.size() / 2;
        return generate_const_array<index_int>(_c<ns + 2>, [&](auto i) {
            if constexpr(i < 2)
                return index_c<0>;
            else
                return index_c<p[i - 2]>;
        });
    }

    // Total (left+right) padding per dim: (0, 0, left_h+right_h, left_w+right_w)
    static constexpr auto total_padding()
    {
        constexpr auto p  = get_padding();
        constexpr auto ns = p.size() / 2;
        return generate_const_array<index_int>(_c<ns + 2>, [&](auto i) {
            if constexpr(i < 2)
                return index_c<0>;
            else
                return index_c<p[i - 2] + p[i - 2 + ns]>;
        });
    }

    static constexpr bool is_padded()
    {
        return (region_lens() != (tiles_per_dim() * output_lens() + total_padding()));
    }

    // Output channels per input channel (channel multiplier)
    template <class InputShape>
    static constexpr index_int channel_multiplier()
    {
        return OutputShape{}.lens[1] / InputShape{}.lens[1];
    }

    // Input channels covered by the halo: the whole tile, or a single input channel
    // shared by every output channel of the tile.
    template <class InputShape>
    static constexpr index_int input_channel_tile()
    {
        constexpr auto multiplier = channel_multiplier<InputShape>();
        static_assert(multiplier == 1 or multiplier % ChannelTile == 0,
                      "Channel tile must stay within one input channel");
        return multiplier == 1 ? ChannelTile : 1;
    }

    // Halo lens for a given input shape: output_lens + (input_spatial - output_spatial)
    // With padding, the output is larger so the raw difference is too small; add total padding.
    template <class InputShape>
    static constexpr auto halo_lens_for()
    {
        return return_array_c([] {
            auto input_spatial = make_slice(InputShape{}, keep_spatial()).lens;
            auto result = output_lens() + (input_spatial - out_spatial_lens() + total_padding());
            result[1]   = input_channel_tile<InputShape>();
            return result;
        });
    }

    // Shared-memory halo layout: packed in the input's memory order
    template <class InputShape>
    static constexpr auto halo_shape_for()
    {
        return make_shape_from_permutation(halo_lens_for<InputShape>(),
                                           find_permutation(InputShape{}));
    }

    // Halo as seen by the output channels of the tile; a shared input channel is broadcast
    template <class InputShape>
    static constexpr auto halo_view_shape_for()
    {
        constexpr auto lens    = return_array_c([] {
            auto result = halo_lens_for<InputShape>();
            result[1]   = ChannelTile;
            return result;
        });
        constexpr auto strides = return_array_c([] {
            auto result = halo_shape_for<InputShape>().strides;
            if(halo_lens_for<InputShape>()[1] != ChannelTile)
                result[1] = 0;
            return result;
        });
        return make_shape(lens, strides);
    }

    // A halo dim is fully covered when every block reads its whole input extent
    template <class InputShape>
    static constexpr auto halo_covered_for()
    {
        return generate_const_array<index_int>(ndim(), [](auto d) {
            constexpr auto hl        = halo_lens_for<InputShape>();
            constexpr bool whole     = hl[d] == InputShape{}.lens[d];
            constexpr bool spatial   = d >= 2;
            constexpr bool untiled   = tiles_per_dim()[d] == 1 and total_padding()[d] == 0;
            constexpr index_int flag = (whole and (not spatial or untiled)) ? 1 : 0;
            return index_c<flag>;
        });
    }

    // Contiguous input elements per halo row: the fastest memory-order dims merge while
    // dense in the input and fully covered, plus the first partially covered one.
    template <class InputShape>
    static constexpr index_int halo_span_for()
    {
        constexpr auto perm    = find_permutation(InputShape{});
        constexpr auto lens    = reorder_dims(halo_lens_for<InputShape>(), perm);
        constexpr auto strides = reorder_dims(InputShape{}.strides, perm);
        constexpr auto covered = reorder_dims(halo_covered_for<InputShape>(), perm);
        index_int span         = 1;
        index_int expected     = 1;
        for(diff_int d = lens.size() - 1; d >= 0; d--)
        {
            if(strides[d] != expected)
                break;
            span *= lens[d];
            expected = strides[d] * lens[d];
            if(covered[d] == 0)
                break;
        }
        return span;
    }

    // Halo lens outside the contiguous span (span dims are 1)
    template <class InputShape>
    static constexpr auto halo_row_lens_for()
    {
        return return_array_c([] {
            constexpr auto perm = find_permutation(InputShape{});
            auto lens           = reorder_dims(halo_lens_for<InputShape>(), perm).base();
            index_int span      = halo_span_for<InputShape>();
            for(diff_int d = lens.size() - 1; d >= 0 and span > 1; d--)
            {
                span /= lens[d];
                lens[d] = 1;
            }
            return reorder_dims(lens, invert_permutation(perm));
        });
    }

    // Logical dim that bounds the span: the only span dim that can be out of bounds
    template <class InputShape>
    static constexpr index_int halo_span_dim_for()
    {
        constexpr auto perm     = find_permutation(InputShape{});
        constexpr auto lens     = reorder_dims(halo_lens_for<InputShape>(), perm);
        constexpr auto row_lens = reorder_dims(halo_row_lens_for<InputShape>(), perm);
        diff_int d              = lens.size() - 1;
        while(d > 0 and row_lens[d - 1] != lens[d - 1])
            d--;
        return perm[d];
    }

    // Members are whole arrays so vectorized array ops never straddle two members,
    // which would keep the struct in memory instead of registers.
    index idx;
    array<index_int, ndim()> origin;      // (batch, first channel, 0...) of the block's slice
    array<index_int, ndim()> tile_origin; // (0, 0, spatial origin...) of the block's tile
    array<index_int, ndim()> lane;        // this lane's position within the tile

    // Type for shared memory allocation
    template <class Input>
    __device__ auto shared_allocate() const
    {
        using type        = typename Input::type;
        constexpr auto hl = halo_lens_for<get_shape_c<Input>>();
        return uninitialized_buffer<type, hl.product()>{};
    }

    // View of an (N, C, spatial...) tensor over this block's batch and channel group
    template <index_int TC, class Tensor>
    __device__ auto slice_channels(Tensor t) const
    {
        constexpr auto s    = get_shape_c<Tensor>{};
        constexpr auto lens = return_array_c([] {
            auto result = get_shape_c<Tensor>{}.lens;
            result[0]   = 1;
            result[1]   = TC;
            return result;
        });
        // Channel origin scaled for tensors with fewer channels (channel multiplier)
        constexpr auto scale = return_array_c([] {
            auto result = transform(get_shape_c<Tensor>{}.lens, [](auto) { return index_c<1>; });
            result[1]   = channel_multiplier<get_shape_c<Tensor>>();
            return result;
        });
        return make_tensor_view(t.data() + s.index(origin / scale), make_shape(lens, s.strides));
    }

    // View of an output-shaped tensor over this block: (1, ChannelTile, spatial...)
    template <class Tensor>
    __device__ auto slice(Tensor t) const
    {
        return slice_channels<ChannelTile>(t);
    }

    // View of the (C, 1, k...) weights for this lane's channel: (1, 1, k...)
    template <class Tensor>
    __device__ auto slice_weights(Tensor w) const
    {
        constexpr auto s    = get_shape_c<Tensor>{};
        constexpr auto lens = make_slice(s, keep_spatial()).lens;
        auto w_origin       = generate_array<index_int>(ndim(), [&](auto d) -> index_int {
            if constexpr(d == 0)
                return origin[1] + lane[1];
            else
                return 0;
        });
        return make_tensor_view(w.data() + s.index(w_origin), make_shape(lens, s.strides));
    }

    // Copy input halo tile into shared memory, return tensor_view over smem
    template <class Input, class Smem>
    __device__ auto copy(Input input, Smem& smem) const
    {
        using type                = typename Input::type;
        using input_shape         = get_shape_c<Input>;
        constexpr auto hl         = halo_lens_for<input_shape>();
        constexpr auto perm       = find_permutation(input_shape{});
        constexpr index_int span  = halo_span_for<input_shape>();
        constexpr auto row_lens   = halo_row_lens_for<input_shape>();
        constexpr auto row_shape  = make_shape(index_ints<hl.product() / span, span>{});
        constexpr index_int outer = halo_span_dim_for<input_shape>();
        constexpr index_int inner = span / hl[outer];
        constexpr auto row_mask   = transform(row_lens, hl, [](auto r, auto h) { return r == h; });

        auto in_ch       = slice_channels<input_channel_tile<input_shape>()>(input);
        auto halo_origin = tile_origin - left_padding();
        // Only the outer span dim can leave the input; clamp it to a column range
        index_int pad       = left_padding()[outer];
        index_int col_begin = pad > tile_origin[outer] ? (pad - tile_origin[outer]) * inner : 0;
        index_int limit     = in_ch.get_shape().lens[outer] + pad;
        index_int col_end   = 0;
        if(tile_origin[outer] < limit)
        {
            index_int avail = limit - tile_origin[outer];
            col_end         = (avail < hl[outer] ? avail : hl[outer]) * inner;
        }

        // The halo is packed in the order rows are decomposed, so the linear index is
        // already the shared-memory offset.
        idx.local_stride(_c<hl.product()>, [&](auto i) {
            auto rc        = row_shape.multi(i);
            auto row_multi = multi_from_permutation(row_lens, perm, rc[0]);
            auto pos       = halo_origin + row_multi;
            auto offset    = in_ch.get_shape().index(pos) + rc[1];
            if constexpr(is_padded())
            {
                bool valid = rc[1] >= col_begin and rc[1] < col_end and
                             in_bounds(pos * row_mask, in_ch.get_shape().lens);
                smem[i]    = valid ? type{in_ch.data()[offset]} : type{0};
            }
            else
            {
                smem[i] = in_ch.data()[offset];
            }
        });

        return make_tensor_view(smem.data(), halo_view_shape_for<input_shape>());
    }

    // Iterate over this lane's output positions with bounds checking. Outputs step by a
    // constant along the last spatial dim so their offsets fold into immediates.
    template <class F>
    __device__ void for_each(F f) const
    {
        repeat(_c<NTiles>, [&](auto k) {
            auto out_multi = lane;
            out_multi[ndim() - 1] += k * TileLens{}.back();
            auto out_pos = tile_origin + out_multi;
            if constexpr(is_padded())
            {
                if(not in_bounds(out_pos, region_lens()))
                    return;
            }
            f(out_pos, out_multi);
        });
    }
};

template <index_int NTiles,
          index_int ChannelTile = 1,
          class TileLens,
          class OutputShape,
          class Padding = index_ints<0>>
__device__ auto make_spatial_tiler(index idx, TileLens, OutputShape, Padding = {})
{
    using tiler_type = spatial_tiler<NTiles, TileLens, OutputShape, Padding, ChannelTile>;

    // Blocks: (N, C / ChannelTile, tiles...)
    constexpr auto block_shape = make_shape(return_array_c([] {
        auto result = tiler_type::tiles_per_dim().base();
        auto olens  = OutputShape{}.lens;
        result[0]   = olens[0];
        result[1]   = tiler_type::channel_tiles();
        return result;
    }));
    auto block_multi           = block_shape.multi(idx.group);
    auto origin      = generate_array<index_int>(tiler_type::ndim(), [&](auto d) -> index_int {
        if constexpr(d == 0)
            return block_multi[0];
        else if constexpr(d == 1)
            return block_multi[1] * ChannelTile;
        else
            return 0;
    });
    auto tile_origin = generate_array<index_int>(tiler_type::ndim(), [&](auto d) -> index_int {
        if constexpr(d < 2)
            return 0;
        else
            return block_multi[d] * tiler_type::output_lens()[d];
    });
    MIGRAPHX_ASSERT(idx.nlocal() == tiler_type::lane_lens().product());
    // Lanes are assigned in the output's memory order so a wave stays contiguous
    auto lane =
        multi_from_permutation(tiler_type::lane_lens(), tiler_type::permutation(), idx.local);

    return tiler_type{idx, origin, tile_origin, lane};
}

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_SPATIAL_TILER_HPP
