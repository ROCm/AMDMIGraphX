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
#ifndef MIGRAPHX_GUARD_KERNELS_DIRECT_CONV_HPP
#define MIGRAPHX_GUARD_KERNELS_DIRECT_CONV_HPP

#include <migraphx/kernels/tiled_conv.hpp>

namespace migraphx {

// Input channels a lane reduces per dot product: half channel pairs map onto
// v_dot2_f32_f16
template <class T, index_int CTile>
constexpr index_int direct_conv_dot_size()
{
    return is_same<T, half>{} and CTile % 2 == 0 ? 2 : 1;
}

// Shared-memory layout of the staged (CTile, KTile, k...) weights with dim 1 fastest, so a
// lane reads its KVec output channels as one vector. With dot products of D channels, dim 1
// holds D input channels per output channel, so an operand's values are adjacent.
template <index_int CTile, index_int KTile, class Weights>
constexpr auto direct_conv_weights_shape(Weights)
{
    constexpr auto lens    = return_array_c([] {
        auto result = get_shape_c<Weights>{}.lens;
        result[0]   = CTile;
        result[1]   = KTile;
        return result;
    });
    constexpr index_int nd = lens.size();
    constexpr auto perm    = generate_const_array<index_int>(_c<nd>, [](auto i) {
        if constexpr(i == 0)
            return index_c<0>;
        else if constexpr(i == nd - 1)
            return index_c<1>;
        else
            return index_c<i + 1>;
    });
    return make_shape_from_permutation(lens, perm);
}

// Stage the block's weights for the input channels starting at `channel` in the
// direct_conv_weights_shape layout. Elements are visited in global memory order so the loads
// coalesce; output channels past the end of a partial tile get zeros.
template <index_int D, class Tiler, class Weights, class WeightsLds>
__device__ void
direct_conv_copy_weights(const Tiler& tiler, Weights w, WeightsLds w_lds, index_int channel)
{
    constexpr auto lds_lens   = get_shape_c<WeightsLds>{}.lens;
    constexpr auto chunk_lens = return_array_c([] {
        auto result = get_shape_c<WeightsLds>{}.lens;
        result[0]   = get_shape_c<WeightsLds>{}.lens[1] / D;
        result[1]   = get_shape_c<WeightsLds>{}.lens[0] * D;
        return result;
    });
    constexpr auto dot_shape  = make_shape(index_ints<lds_lens[0], D>{});
    constexpr auto perm       = find_permutation(get_shape_c<Weights>{});
    array<index_int, lds_lens.size()> start{};
    start[0] = tiler.origin[1];
    start[1] = channel;
    tiler.idx.local_stride(_c<chunk_lens.product()>, [&](auto i) {
        auto multi     = multi_from_permutation(chunk_lens, perm, i);
        auto dot_multi = dot_shape.multi(multi[1]);
        auto lds_multi = multi;
        lds_multi[0]   = dot_multi[0];
        lds_multi[1]   = multi[0] * D + dot_multi[1];
        if constexpr(Tiler::partial_channel_tile())
        {
            constexpr index_int out_channels = get_shape_c<Weights>{}.lens[0];
            using type                       = typename WeightsLds::type;
            auto pos                         = start + multi;
            w_lds[lds_multi]                 = pos[0] < out_channels ? type{w[pos]} : type{0};
        }
        else
        {
            w_lds[lds_multi] = w[start + multi];
        }
    });
}

// Halo values of one filter column for each of the lane's NTiles column tiles, keeping the
// rows that the taps of its row runs read. origin is the column's first halo position.
template <index_int NTiles,
          index_int NRows,
          index_int Taps,
          index_int TileStep,
          class Halo,
          class Pos,
          class Stride,
          class Dilation>
__device__ auto direct_conv_load_column(Halo x_ch, Pos origin, Stride, Dilation)
{
    using type                    = typename Halo::type;
    constexpr index_int ndim      = get_shape_c<Halo>{}.lens.size();
    constexpr auto stride         = Stride{};
    constexpr auto dilation       = Dilation{};
    constexpr index_int halo_rows = run_halo_rows<NRows, Taps, stride[2], dilation[2]>();
    static_assert(halo_rows <= get_shape_c<Halo>{}.lens[2], "Row run reads past the halo");
    constexpr auto col_rows = make_shape(index_ints<NTiles, halo_rows>{});
    array<type, col_rows.elements()> col;
    repeat(_c<NTiles>, [&](auto t) {
        repeat(_c<halo_rows>, [&](auto i) {
            if constexpr(halo_row_used<NRows, Taps, stride[2], dilation[2]>(decltype(i){}))
            {
                auto pos = origin;
                pos[2] += i;
                pos[ndim - 1] += t * TileStep;
                col[col_rows.index(array<index_int, 2>{t, i})] = x_ch[pos];
            }
        });
    });
    return col;
}

// The weight operands of a lane's KVec output channels at w_multi, D input channels each,
// as one vector load when they are aligned
template <index_int KVec, index_int D, class Weights, class Pos>
__device__ auto direct_conv_load_weights(Weights w_ch, Pos w_multi)
{
    using type = typename Weights::type;
    array<type, KVec * D> values;
    if constexpr(channel_vector_aligned<KVec * D>(get_shape_c<Weights>{}))
    {
        values = load_channels<KVec * D>(w_ch, w_multi);
    }
    else
    {
        repeat(_c<KVec>, [&](auto v) {
            auto pos = w_multi;
            pos[1] += v * D;
            auto operand = load_channels<D>(w_ch, pos);
            repeat(_c<D>, [&](auto d) { values[v * D + d] = operand[d]; });
        });
    }
    if constexpr(D == 1)
    {
        return values;
    }
    else
    {
        array<vec<type, D>, KVec> result;
        repeat(_c<KVec>, [&](auto v) {
            result[v] = vec_generate<D>([&](auto d) { return values[v * D + d]; });
        });
        return result;
    }
}

// Multiply the KVec weight operands of filter row r with the column values each of the
// lane's outputs reads for that row
template <index_int NTiles,
          index_int NRows,
          index_int KVec,
          index_int Taps,
          class Acc,
          class Col,
          class Values,
          class R,
          class Stride,
          class Dilation>
__device__ Acc direct_conv_fma(Acc acc, const Col& col, const Values& wv, R r, Stride, Dilation)
{
    constexpr auto stride    = Stride{};
    constexpr auto dilation  = Dilation{};
    constexpr auto acc_shape = make_shape(index_ints<NTiles, NRows, KVec>{});
    constexpr auto col_rows  = make_shape(index_ints<NTiles, decltype(col.size()){} / NTiles>{});
    repeat(_c<KVec>, [&](auto v) {
        repeat(_c<NTiles>, [&](auto t) {
            repeat(_c<NRows>, [&](auto q) {
                constexpr index_int row = q * stride[2] + r * dilation[2];
                static_assert(row < col_rows.lens[1] and
                                  halo_row_used<NRows, Taps, stride[2], dilation[2]>(row),
                              "Tap reads an unloaded halo row");
                auto i = acc_shape.index(array<index_int, 3>{t, q, v});
                acc[i] = vec_dot(col[col_rows.index(array<index_int, 2>{t, row})], wv[v], acc[i]);
            });
        });
    });
    return acc;
}

// Accumulate one staged channel group (D input channels) over every filter tap into the
// lane's NTiles x NRows x KVec outputs. in_multi is the lane's first halo position with the
// group in dim 1. A column's halo rows load once for all its taps, and each weight operand
// serves all the lane's outputs.
template <index_int NTiles,
          index_int NRows,
          index_int KVec,
          index_int TileW,
          index_int D,
          class Acc,
          class Halo,
          class Weights,
          class Pos,
          class Stride,
          class Dilation>
__device__ Acc direct_conv_accumulate(Acc acc,
                                      Halo x_ch,
                                      Weights w_ch,
                                      Pos in_multi,
                                      index_int lane_k,
                                      Stride stride,
                                      Dilation dilation)
{
    constexpr auto k_lens         = get_shape_c<Weights>{}.lens;
    constexpr index_int kh        = k_lens[2];
    constexpr auto col_shape      = make_shape(return_array_c([] {
        auto result = get_shape_c<Weights>{}.lens;
        result[0]   = 1;
        result[1]   = 1;
        result[2]   = 1;
        return result;
    }));
    constexpr index_int tile_step = TileW * Stride{}[k_lens.size() - 1];
    repeat(col_shape.elements(), [&](auto j) {
        auto col_multi = col_shape.multi(j);
        auto col       = direct_conv_load_column<NTiles, NRows, kh, tile_step>(
            x_ch, in_multi + col_multi * dilation, stride, dilation);
        repeat(_c<kh>, [&](auto r) {
            auto w_multi = col_multi;
            w_multi[0]   = in_multi[1];
            w_multi[1]   = lane_k * D;
            w_multi[2]   = r;
            acc          = direct_conv_fma<NTiles, NRows, KVec, kh>(
                acc, col, direct_conv_load_weights<KVec, D>(w_ch, w_multi), r, stride, dilation);
        });
    });
    return acc;
}

// Apply the fused op to row r of the lane's accumulators and store its channels. A partial
// channel tile stores only the channels the tiler contains, element by element.
template <index_int KVec,
          class ConvType,
          class Tiler,
          class F,
          class Output,
          class Pack,
          class Pos,
          class Acc,
          class R>
__device__ void direct_conv_store(
    const Tiler& tiler, F f, Output out_ch, Pack xs_pack, Pos pos, const Acc& acc, R r)
{
    if constexpr(not Tiler::partial_channel_tile())
    {
        tiled_conv_store<KVec, ConvType>(f, out_ch, xs_pack, pos, acc, r);
    }
    else
    {
        constexpr auto acc_shape = make_shape(index_ints<decltype(acc.size()){} / KVec, KVec>{});
        xs_pack([&](auto... xs) {
            repeat(_c<KVec>, [&](auto v) {
                auto p = pos;
                p[1] += v;
                if(not tiler.contains(p))
                    return;
                out_ch[p] =
                    f(static_cast<ConvType>(acc[acc_shape.index(array<index_int, 2>{r, v})]),
                      xs[p]...);
            });
        });
    }
}

// Dense convolution: a block computes KTile output channels over a spatial tile and a lane
// computes KVec output channels of NRows rows by NTiles strided columns. The input channels
// are staged CTile at a time in shared memory, as halos for the tile and as the weights of
// the block's output channels, and reduced in dot products of direct_conv_dot_size.
template <class TileLens,
          index_int NTiles,
          index_int KTile = 1,
          index_int NRows = 1,
          index_int KVec  = 1,
          index_int CTile = 1,
          class Padding,
          class Stride,
          class Dilation,
          class F,
          class Output,
          class Input,
          class Weights,
          class... Inputs>
__device__ void direct_conv(
    TileLens, Padding, Stride, Dilation, F f, Output output, Input x, Weights w, Inputs... inputs)
{
    using type                       = typename Input::type;
    using input_shape                = get_shape_c<Input>;
    constexpr index_int ndim         = get_shape_c<Output>{}.lens.size();
    constexpr index_int channels     = input_shape{}.lens[1];
    constexpr index_int out_channels = get_shape_c<Output>{}.lens[1];
    static_assert(channels % CTile == 0, "Input channel tile must divide the input channels");
    static_assert(get_shape_c<Weights>{}.lens[0] == out_channels and
                      get_shape_c<Weights>{}.lens[1] == channels,
                  "Weights must be (K, C, k...) for a dense convolution");
    static_assert(is_same<typename Weights::type, type>{}, "Weights must have the input type");
    constexpr auto dilation = spatial_steps<ndim>(Dilation{});
    constexpr auto reach    = conv_reach<Weights>(dilation);
    auto idx                = make_index();
    auto tiler              = make_spatial_tiler<NTiles, KTile, NRows, KVec>(
        idx, TileLens{}, get_shape_c<Output>{}, Padding{}, Stride{}, reach);
    using tiler_type      = decltype(tiler);
    constexpr auto stride = tiler_type::get_stride();
    // Widest halo copy that fits in 16 bytes
    constexpr index_int halo_vec =
        tiler_type::template halo_channel_vector<input_shape, CTile, 16 / sizeof(type)>();
    constexpr index_int dot_size = direct_conv_dot_size<type, CTile>();
    constexpr auto w_lds_shape =
        direct_conv_weights_shape<CTile / dot_size, KTile * dot_size>(get_shape_c<Weights>{});
    // Halos are staged channel-major whatever the input layout, so lanes along the last
    // spatial dim read consecutive shared-memory words. Each element packs the dot_size
    // channels of one dot product operand.
    constexpr auto halo_shape =
        make_shape(tiler_type::template halo_lens_for<input_shape, CTile / dot_size>());

    __shared__ decltype(tiler.template shared_allocate<Input, CTile>()) smem;
    // A lane's weight operands load as one vector, which can be wider than 16 bytes
    constexpr index_int w_vec_bytes = KVec * dot_size * sizeof(type);
    constexpr index_int w_align =
        w_vec_bytes > 16 and is_power_of_2(w_vec_bytes) ? w_vec_bytes : 16;
    __shared__ uninitialized_buffer<type, w_lds_shape.element_space(), w_align> wmem;
    auto w_ch = make_tensor_view(wmem.data(), w_lds_shape);

    array<float, NTiles * NRows * KVec> acc{};
    // The lane's first input position: its spatial origin stepped by the stride
    auto lane_in = tiler.lane * stride;
    for(index_int c0 = 0; c0 < channels; c0 += CTile)
    {
        auto x_ch = tiler.template copy_channels<CTile, halo_vec>(x, smem, c0, halo_shape);
        direct_conv_copy_weights<dot_size>(tiler, w, w_ch, c0);
        __syncthreads();
        for(index_int c = 0; c < CTile / dot_size; c++)
        {
            auto in_multi = lane_in;
            in_multi[1]   = c;
            acc = direct_conv_accumulate<NTiles, NRows, KVec, TileLens{}.back(), dot_size>(
                acc, x_ch, w_ch, in_multi, tiler.lane[1], stride, dilation);
        }
        __syncthreads();
    }

    auto out_ch              = tiler.slice(output);
    auto xs_pack             = pack(tiler.slice(inputs)...);
    constexpr auto acc_shape = make_shape(index_ints<NTiles, NRows * KVec>{});
    tiler.for_each_run([&](auto out_pos, auto, auto t) {
        array<float, NRows * KVec> tile_acc;
        repeat(_c<NRows * KVec>,
               [&](auto i) { tile_acc[i] = acc[acc_shape.index(array<index_int, 2>{t, i})]; });
        repeat(_c<NRows>, [&](auto r) {
            auto pos = out_pos;
            pos[2] += r;
            if constexpr(tiler_type::needs_bounds_check())
            {
                if(not tiler.contains(pos))
                    return;
            }
            direct_conv_store<KVec, type>(tiler, f, out_ch, xs_pack, pos, tile_acc, r);
        });
    });
}

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_DIRECT_CONV_HPP
