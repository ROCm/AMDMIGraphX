/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef MIGRAPHX_GUARD_KERNELS_SPARSE_ATTN_SOFTMAX_HPP
#define MIGRAPHX_GUARD_KERNELS_SPARSE_ATTN_SOFTMAX_HPP

#include "migraphx/kernels/float8.hpp"
#include "migraphx/kernels/gqa_softmax.hpp"
#include <migraphx/kernels/group_query_attention.hpp>
#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/tensor_view.hpp>

namespace migraphx {

template <size_t SparseBlockSize, class AttnProbs, class SeqLensK, class Params>
__device__ void sparse_attn_calculate_softmax(AttnProbs attention_probs,
                                              SeqLensK seqlens_k,
                                              Params params,
                                              index_int idx)
{
    (void)attention_probs;
    (void)seqlens_k;
    (void)params;
    (void)idx;
}

template <size_t SparseBlockSize,
          class Output,
          class Input,
          class PresentKey,
          class Probs,
          class SeqLensK,
          class Mask,
          class Params>
__device__ void sparse_attn_softmax(
    Output output, Input, PresentKey, Probs, SeqLensK seqlens_k, Mask mask, Params params)
{
    constexpr index_int elements    = output.get_shape().elements();
    constexpr index_int num_layouts = mask.get_shape().lens[0];
    make_index().global_stride(elements, [&](auto idx) {
        const auto multi_idx = output.get_shape().multi(idx);
        const auto batch_idx = multi_idx[0];
        const auto head_idx  = multi_idx[1];
        const auto row_idx   = multi_idx[2];
        const auto col_idx   = multi_idx[3];

        const index_int key_total_seq_len = seqlens_k[batch_idx];
        if(col_idx >= key_total_seq_len)
            return;

        const index_int past_seq_len = key_total_seq_len - params.sequence_length;
        const auto row_abs_idx       = row_idx + past_seq_len;
        const auto causal_length     = row_abs_idx + 1;

        if(col_idx < causal_length)
        {
            const index_int layout_idx  = head_idx % num_layouts;
            const index_int mask_row    = row_abs_idx / SparseBlockSize;
            const index_int mask_column = col_idx / SparseBlockSize;
            if(not mask[make_array(layout_idx, mask_row, mask_column)])
            {
                output[multi_idx] = numeric_lowest<typename Output::type>();
            }
        }
        else
        {
            output[multi_idx] = 0;
        }

        // Not nice, failing to utilise threads, but this is just a basic working implementation
        // which will (hopefully) be improved upon
        if(col_idx == 0)
        {
            auto it = output.begin() +
                      output.get_shape().index(make_array(batch_idx, head_idx, row_idx, 0));
            softmax_inplace(it, 1, causal_length);
        }
    });
}

} // namespace migraphx
#endif
