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
#ifndef MIGRAPHX_GUARD_KERNELS_RNN_VARIABLE_SEQ_LENS_HPP
#define MIGRAPHX_GUARD_KERNELS_RNN_VARIABLE_SEQ_LENS_HPP

#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/shape.hpp>
#include <migraphx/kernels/tensor_view.hpp>

namespace migraphx {

template <class Input, class SeqLens, class Output>
__device__ void rnn_var_sl_shift_sequence(Input input, SeqLens seq_lens, Output output)
{
    auto ind     = make_index();
    auto max_len = output.get_shape().lens[0];

    ind.global_stride(output.get_shape().elements(), [&](auto i) {
        auto idx = output.get_shape().multi(i);
        auto t   = idx[0];
        auto b   = idx[1];
        auto l   = seq_lens[b];
        if(t >= max_len - l)
        {
            auto in_idx = idx;
            in_idx[0] -= (max_len - l);
            output[i] = input[in_idx];
        }
        else
        {
            output[i] = 0;
        }
    });
}

template <bool IsReverse, class Input, class SeqLens, class Output>
__device__ void rnn_var_sl_shift_output(Input input, SeqLens seq_lens, Output output)
{
    auto ind     = make_index();
    auto max_len = output.get_shape().lens[0];

    ind.global_stride(output.get_shape().elements(), [&](auto i) {
        auto idx = output.get_shape().multi(i);
        auto t   = idx[0];
        auto d   = idx[1];
        auto b   = idx[2];
        auto l   = seq_lens[b];
        if(t < l)
        {
            // d==1 is reverse direction in bidirectional
            auto offset = (d == 1 or IsReverse) ? 1 : 0;
            auto in_idx = idx;
            in_idx[0] += offset * (max_len - l);
            output[i] = input[in_idx];
        }
        else
        {
            output[i] = 0;
        }
    });
}

template <bool IsReverse, class Input, class SeqLens, class Output>
__device__ void rnn_var_sl_last_output(Input input, SeqLens seq_lens, Output output)
{
    auto ind = make_index();

    constexpr auto in_shape      = get_shape_c<Input>{};
    // output is 3D [directions, batch, hidden] but input is 4D [seq_len, directions, batch, hidden]
    // create a 4D computational shape with seq_len=1 to map output indices to input indices
    constexpr auto out_comp_lens = [&] {
        auto lens = in_shape.lens;
        lens[0]   = 1;
        return lens;
    }();
    constexpr auto out_comp_shape = make_shape(out_comp_lens, in_shape.strides);

    ind.global_stride(output.get_shape().elements(), [&](auto i) {
        auto idx = out_comp_shape.multi(i);
        auto d   = idx[1];
        auto b   = idx[2];
        auto l   = seq_lens[b];

        idx[0]    = (IsReverse or d == 1) ? 0 : (l - 1);
        output[i] = input[idx];
    });
}

} // namespace migraphx
#endif
