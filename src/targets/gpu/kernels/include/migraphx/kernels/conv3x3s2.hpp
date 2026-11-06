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
 */
#ifndef MIGRAPHX_GUARD_KERNELS_CONV3X3S2_HPP
#define MIGRAPHX_GUARD_KERNELS_CONV3X3S2_HPP

#include <migraphx/kernels/array.hpp>
#include <migraphx/kernels/bit_cast.hpp>
#include <migraphx/kernels/buffer_load.hpp>
#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/tensor_view.hpp>
#include <migraphx/kernels/vec.hpp>
#include <migraphx/kernels/types.hpp>
#include <migraphx/kernels/uninitialized_buffer.hpp>
#include <migraphx/kernels/functional.hpp>
#include <migraphx/kernels/integral_constant.hpp>

namespace migraphx {

// FP32 3x3 convolution, stride 2, padding 1, group 1, as a direct convolution on gfx12:
//   out[n][k][oh][ow] = f(sum_{c,r,s} w[k][c][r][s] * x[n][c][2*oh-1+r][2*ow-1+s], inputs...)
// Output pixel (oh, ow) reads the 3x3 input window starting at row 2*oh-1, column 2*ow-1
// (2 = stride, -1 = padding); window cells outside the image count as 0.
//
// Who computes what (example: x [1,16,8,8], K=8 -> output [1,8,4,4], TK=8, TP=2, NW=4):
//  - A lane (one thread of a 32-wide wave) owns a TK x TP register tile: TK output channels
//    k0..k0+TK-1 times TP neighbouring output columns ow0..ow0+TP-1 of one output row oh.
//    Example: lane 0 computes channels 0..7 at output pixels (0,0) and (0,1).
//  - Per input channel the lane loads one window of 3 rows x (2*TP+1) columns (neighbouring
//    outputs share a column) and does 9*TK*TP FMAs with it (144 in the example), so every
//    loaded pixel is reused by all TK output channels.
//  - The 9*TK weights of one input channel do not depend on the lane, so the whole wave reads
//    the same values: scalar loads, used as the SGPR operand of v_dual_fmac_f32.
//  - The NW waves of a workgroup own the same tiles but split the input channels (example:
//    wave w sums channels 4w..4w+3); their partial sums are added through LDS at the end.
//
// Steps: (1) find the lane's tile, (2) compute the window's byte offsets once, (3) for each
// input channel load the window and do the FMAs, (4) add the NW partial sums in LDS,
// (5) apply f (fused bias, SiLU, ...) and store.
//
// Weights: host-transposed and zero padded to [C, 9, KPAD] (KPAD = K rounded up to 16, a
// multiple of TK): the TK weights of one tap are neighbours in memory, and a partial last
// block of TK channels reads zeros instead of the next tap's weights.
// Input: NCHW rows contiguous, even H and W (so the bottom/right padding is never read).
// NOLINTBEGIN(readability-function-size)
template <index_int TK,
          index_int TP,
          index_int NW,
          class PostInput,
          class F,
          class Output,
          class Input,
          class Weights,
          class... Inputs>
__device__ void conv3x3s2_fp32(F f, Output output, Input x, Weights weights, Inputs... inputs)
{
    // Tile parameters: TP columns per lane, TK output channels per lane, wave32 only.
    static_assert(TP == 1 or TP == 2 or TP == 4, "TP must be 1, 2 or 4");
    static_assert(TK >= 1 and NW >= 1, "TK and NW must be non-zero");
    static_assert(MIGRAPHX_WAVEFRONTSIZE == 32, "conv3x3s2_fp32 requires wave32");

    // Shapes: x is [N, C, H, W], output is [N, K, OH, OW], weights are [C, 9, KPAD].
    constexpr auto x_lens     = get_shape_c<Input>{}.lens;
    constexpr auto x_strides  = get_shape_c<Input>{}.strides;
    constexpr auto out_lens   = get_shape_c<Output>{}.lens;
    constexpr auto w_lens     = get_shape_c<Weights>{}.lens;
    constexpr index_int in_c  = x_lens[1];   // input channels C
    constexpr index_int in_w  = x_lens[3];   // input width W
    constexpr index_int out_c = out_lens[1]; // output channels K
    constexpr index_int out_h = out_lens[2]; // output height OH
    constexpr index_int out_w = out_lens[3]; // output width OW
    constexpr index_int kpad  = w_lens[2];   // K rounded up to a multiple of TK
    // The loads below assume dense NCHW rows, even H and W, and the transposed weights.
    static_assert(x_strides[3] == 1 and x_strides[2] == in_w, "input rows must be contiguous");
    static_assert(x_lens[2] == 2 * out_h and in_w == 2 * out_w, "needs even H and W");
    static_assert(w_lens[0] == in_c and w_lens[1] == 9 and kpad % TK == 0,
                  "weights must be [C, 9, KPAD]");

    // Work split: one lane per (output row, group of TP columns), TK channels per lane.
    constexpr index_int groups_w   = (out_w + TP - 1) / TP; // column groups per output row
    constexpr index_int groups     = out_h * groups_w;      // (row, column group) pairs
    constexpr index_int p_blocks   = (groups + 31) / 32;    // waves needed to cover all groups
    constexpr index_int k_blocks   = (out_c + TK - 1) / TK; // blocks of TK output channels
    constexpr index_int c_per_wave = (in_c + NW - 1) / NW;  // input channels per wave
    constexpr index_int ncols      = 2 * TP + 1;            // window columns per row

    // Step 1: which tile this lane owns. The output plane is cut into groups of TP columns of
    // one row, one group per lane. Example, 4x4 output, TP=2 -> 8 groups, lanes 8..31 idle;
    // every lane computes all TK channels k0..k0+TK-1 at its TP pixels:
    //   ow        0   1 |   2   3
    //   oh=0     lane 0 |  lane 1
    //   oh=1     lane 2 |  lane 3
    //   oh=2     lane 4 |  lane 5
    //   oh=3     lane 6 |  lane 7
    // The workgroup index is a mixed-radix number with p_block fastest:
    //   idx.group = (batch * k_blocks + k_block) * p_blocks + p_block
    auto idx                = make_index();
    const index_int lane    = idx.local % 32;                                 // lane in the wave
    const index_int wave    = __builtin_amdgcn_readfirstlane(idx.local / 32); // wave in the group
    const index_int p_block = idx.group % p_blocks;                // which run of 32 groups
    const index_int k_block = (idx.group / p_blocks) % k_blocks;   // which block of TK channels
    const index_int batch   = idx.group / (p_blocks * k_blocks);   // which image
    const index_int g       = p_block * 32 + lane;                 // this lane's group number
    const bool valid        = g < groups;                          // false for idle tail lanes
    // Idle lanes still run the loop (a wave runs in lockstep); oh = ow0 = 0 keeps their loads
    // in range and they never store.
    const index_int oh      = valid ? g / groups_w : 0;            // output row
    const index_int ow0     = valid ? (g % groups_w) * TP : 0;     // first output column
    const index_int k0      = k_block * TK;                        // first output channel

    // Step 2: byte offsets of the lane's window, computed once and reused for every channel.
    // Zero padding needs no branch: the top padding row and the left padding column get an
    // offset past the end of the buffer, which the buffer range check turns into zeros
    // (a negative offset would wrap instead).
    // Example, 8x8 input, lane 1 (oh=0, ow0=2): window = image rows -1..1, columns 3..7
    // (P = padding, L = b32 at left_off, AA = b64 at row_off, BB = b64 at row_off + 8):
    //   col      -1  0  1  2  3  4  5  6  7
    //   row -1    P  P  P  P  P  P  P  P  P    left_off = oob, row_off = oob
    //   row  0    P  .  .  .  L  A  A  B  B    left_off =  12, row_off =  16
    //   row  1    P  .  .  .  L  A  A  B  B    left_off =  44, row_off =  48
    //   row  2    P  .  .  .  .  .  .  .  .
    // Lane 0 (ow0=0): its L column is the padding column -1, so left_off = [oob, oob, oob],
    // row_off = [oob, 0, 32].
    // x_rsrc: buffer descriptor over the whole input; x_bytes is its size in bytes.
    const uint32_t x_bytes = x.get_shape().element_space() * sizeof(float);
    auto x_rsrc            = make_oob_buffer_rsrc(x.data(), x_bytes);
    // oob: an offset one byte past the end of the buffer; a load there returns 0.
    const int32_t oob = static_cast<int32_t>(x_bytes);
    array<int32_t, 3> row_off{};  // byte offset of window column 2*ow0, per window row
    array<int32_t, 3> left_off{}; // byte offset of window column 2*ow0 - 1, per window row
    repeat_c<3>([&](auto r) {
        // Input row of window row r (stride 2, padding 1); -1 is the top padding row.
        const int32_t ih = static_cast<int32_t>(2 * oh + r) - 1;
        // Padding row: use oob. Otherwise the byte offset inside one channel plane.
        row_off[r] = ih >= 0 ? static_cast<int32_t>((ih * in_w + 2 * ow0) * sizeof(float)) : oob;
        // Left column is padding at the image edge (ow0 == 0) or in the padding row.
        left_off[r] = ih >= 0 and ow0 > 0 ? row_off[r] - 4 : oob;
    });
    // Scalar offsets: start of this image, and the size of one channel plane in bytes.
    const int32_t x_base      = batch * x_strides[0] * sizeof(float);
    constexpr int32_t c_bytes = x_strides[1] * sizeof(float);

    const float* w = weights.data();

    // This wave sums input channels [c_begin, c_end); the last wave may get fewer.
    const index_int c_begin = wave * c_per_wave;
    const index_int c_end   = c_begin + c_per_wave < in_c ? c_begin + c_per_wave : in_c;

    // rows: the 3 x ncols input window of one channel (3 rows, ncols columns).
    // xr[r][j] = image row 2*oh-1+r, image column 2*ow0-1+j. Top/left padding reads 0.
    // In a partial last column group, columns >= W hold the next row's values (e.g. W=22,
    // TP=2: columns 19..23 load, 22..23 are not zeros); only the skipped column uses them.
    using rows = array<array<float, ncols>, 3>;
    // load(c): read the window of input channel c with 1 b32 + TP b64 loads per row
    // (TP=2: column 2*ow0-1 | pair 2*ow0, 2*ow0+1 | pair 2*ow0+2, 2*ow0+3). The single
    // column is odd (only 4-byte aligned) and is the only one that can be left padding.
    auto load = [&](index_int c) {
        // Scalar offset that moves row_off and left_off to image `batch`, channel c.
        const int32_t soff = x_base + static_cast<int32_t>(c) * c_bytes;
        rows xr;
        repeat_c<3>([&](auto r) {
            // Window column 0 is image column 2*ow0 - 1, only 4-byte aligned: one b32.
            xr[r][0] = bit_cast<float>(
                __builtin_amdgcn_raw_buffer_load_b32(x_rsrc, left_off[r], soff, 0));
            repeat_c<TP>([&](auto j) {
                // Columns 2*ow0 + 2j and 2*ow0 + 2j + 1 are an 8-byte aligned pair: one b64.
                auto v = bit_cast<vec<float, 2>>(__builtin_amdgcn_raw_buffer_load_b64(
                    x_rsrc, row_off[r] + static_cast<int32_t>(j * 8), soff, 0));
                xr[r][1 + 2 * j] = v[0];
                xr[r][2 + 2 * j] = v[1];
            });
        });
        return xr;
    };

    // acc[k][t]: running sum for output channel k0 + k, output column ow0 + t.
    array<array<float, TP>, TK> acc{};
    // fma_channel(c, xr): add input channel c's 9 taps x TK x TP products to acc.
    auto fma_channel = [&](index_int c, const rows& xr) {
        // Weights of channel c for this lane's TK channels; same for every lane of the wave.
        const float* wc = w + c * 9 * kpad + k0;
        repeat_c<3>([&](auto r) {     // filter row
            repeat_c<3>([&](auto s) { // filter column
                repeat_c<TK>([&](auto k) {
                    // Weight of tap (r, s) for output channel k0 + k.
                    const float wk = wc[(r * 3 + s) * kpad + k];
                    repeat_c<TP>([&](auto t) {
                        // Output column t reads window column 2*t + s of row r.
                        acc[k][t] = __builtin_fmaf(wk, xr[r][2 * t + s], acc[k][t]);
                    });
                });
            });
        });
    };

    // Step 3: for each input channel of this wave, load its window and add its products.
    // Output column t uses window columns 2t..2t+2 (filter columns s = 0..2):
    //   window column j   0   1   2   3   4
    //   t=0 (ow0)        s0  s1  s2
    //   t=1 (ow0+1)              s0  s1  s2      (column 2 is shared)
    // Example (lane 0, k=0, C=16, NW=4): what each input channel c adds to output pixels
    // (0,0) and (0,1); wave w loops over channels 4w..4w+3 only:
    // c        0    1    2    3 |    4    5    6    7 |    8    9   10   11 |   12   13   14   15
    // (0,0)  0.5  2.5 -4.5 -3.0 |  2.0 -0.5  4.0 -2.5 | -2.0  2.0  2.0  5.5 | -4.0 -4.5  2.0  1.0
    // (0,1)  1.0  0.5 -3.5  1.0 |  1.0  3.5  0.5 -2.5 | -4.0  6.0 -1.0 -3.0 | -5.0 -2.0  4.0  1.5
    //         ---- wave 0 ----      ---- wave 1 ----      ---- wave 2 ----      ---- wave 3 ----
    // (Loading the next channel ahead was slower on gfx1201: more VGPRs, fewer waves.)
    for(index_int c = c_begin; c < c_end; ++c)
        fma_channel(c, load(c));

    // Step 5 helper. store(k, v): apply the post-op f and write the lane's TP outputs of
    // channel k. The TP values are neighbours in one output row, so they go to consecutive
    // addresses; the last group of a row may run past out_w, and that column is skipped.
    // Each output element is written by exactly one lane. Example, out_w=7, TP=2, row oh=0:
    //   ow      0  1 |  2  3 |  4  5 |  6  x      (x = column 7 >= out_w, skipped)
    //   lane    0    |  1    |  2    |  3
    using out_type = typename Output::type;
    auto store     = [&](index_int k, const array<float, TP>& v) {
        repeat_c<TP>([&](auto t) {
            const index_int ow = ow0 + t;
            if(ow >= out_w) // last column group may be partial
                return;
            const array<index_int, 4> oid{batch, k, oh, ow};
            // f combines the conv result with the fused inputs at the same output position.
            output[oid] = static_cast<out_type>(f(static_cast<PostInput>(v[t]), inputs[oid]...));
        });
    };

    if constexpr(NW > 1)
    {
        // Step 4: each wave holds a partial sum over its own channels; add them through LDS.
        // Example (lane 0, k=0): each wave's acc[0][t] is the sum of its 4 columns in step 3:
        //   pixel   wave 0  wave 1  wave 2  wave 3       sum
        //   (0,0)     -4.5     3.0     7.5    -5.5  ->   0.5
        //   (0,1)     -1.0     2.5    -2.0    -1.5  ->  -2.0
        constexpr auto partial_shape = make_shape(index_ints<NW, TK * TP, 32>{});
        __shared__ uninitialized_buffer<float, NW * TK * TP * 32> partial_storage;
        auto partials = make_tensor_view(partial_storage.data(), partial_shape);
        // Write this wave's acc to partials[wave][k * TP + t][lane].
        repeat_c<TK>([&](auto k) {
            repeat_c<TP>([&](auto t) {
                partials[array<index_int, 3>{wave, k * TP + t, lane}] = acc[k][t];
            });
        });
        __syncthreads(); // all waves must have written before any wave reads
        // The TK output channels are shared out over the waves. With a full channel tile and
        // NW dividing TK (all tuning configs) every wave stores; in a partial last tile
        // (e.g. K=20, TK=8, NW=4: channels 16..19) waves 2 and 3 have nothing. Example, TK=8, NW=4:
        //   k            0   1 |   2   3 |   4   5 |   6   7
        //   summed by   wave 0 |  wave 1 |  wave 2 |  wave 3
        constexpr index_int rows_per_wave = (TK + NW - 1) / NW;
        if(not valid)
            return;
        repeat_c<rows_per_wave>([&](auto kk) {
            const index_int k = wave * rows_per_wave + kk;
            if(k >= TK or k0 + k >= out_c) // past TK, or a padded channel (KPAD > K)
                return;
            // Sum channel k over the NW waves, then store it.
            array<float, TP> sum{};
            repeat_c<TP>([&](auto t) {
                repeat_c<NW>([&](auto s) {
                    sum[t] += partials[array<index_int, 3>{s, k * TP + t, lane}];
                });
            });
            store(k0 + k, sum);
        });
    }
    else
    {
        // One wave: acc already holds the full sum, store it directly.
        if(not valid)
            return;
        repeat_c<TK>([&](auto k) {
            if(k0 + k >= out_c) // padded channel (KPAD > K)
                return;
            store(k0 + k, acc[k]);
        });
    }
}
// NOLINTEND(readability-function-size)

} // namespace migraphx

#endif // MIGRAPHX_GUARD_KERNELS_CONV3X3S2_HPP
