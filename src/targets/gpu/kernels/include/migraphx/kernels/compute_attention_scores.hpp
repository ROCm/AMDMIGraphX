/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef MIGRAPHX_GUARD_KERNELS_COMPUTE_ATTENTION_SCORES_HPP
#define MIGRAPHX_GUARD_KERNELS_COMPUTE_ATTENTION_SCORES_HPP

#include <migraphx/kernels/group_query_attention.hpp>
#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/algorithm.hpp>
#include <migraphx/kernels/integral_constant.hpp>
#include <migraphx/kernels/tensor_view.hpp>
#include <migraphx/kernels/ck.hpp>
#include <migraphx/kernels/gemm_batcher.hpp>
#include <limits>
#include <migraphx/kernels/type_traits.hpp>

namespace migraphx {

template <class Output,
          class Attn_Probs,
          class SeqLens,
          class PresentValue,
          class Params>
__device__ void
CalculateVxAttentionScore(Output output, // buffer for the result with size BxSxNxH
                          const Attn_Probs attention_probs,   // Attention probs with size BxNxSxT
                          const SeqLens seqlens_k,            // past sequence lengths tensor
                          int batch_size,                     // batch size
                          int sequence_length,                // sequence length
                          int present_buffer_sequence_length, // sequence length in past state
                          int head_size,                      // head size of Q, K, V
                          int hidden_size,                    // hidden size of Output
                          PresentValue present_value,         // present value only
                          Params params,
                          index_int idx)
{
    const int num_heads           = params.num_heads;
    const int kv_num_heads        = params.kv_num_heads;
    const int kv_num_heads_factor = num_heads / kv_num_heads;
    const size_t present_buff_chunk_length =
        static_cast<size_t>(present_buffer_sequence_length) * head_size; // T x H

    auto loop_len           = batch_size * num_heads;
    const index_int i       = idx / (sequence_length * head_size);
    const index_int inner_i = idx % (sequence_length * head_size);
    if(i < loop_len)
    {
        const index_int batch_index  = i / num_heads;
        const index_int head_index   = i % num_heads;
        const index_int total_seqlen = seqlens_k[batch_index] + 1;

        auto pv = present_value + ((i / kv_num_heads_factor) * present_buff_chunk_length);
        Output output_current =
            output + (batch_index * sequence_length * num_heads + head_index) * head_size;
        ptrdiff_t attention_probs_offset = sequence_length * present_buffer_sequence_length * i;

        gemm(sequence_length,
             head_size,
             total_seqlen,
             present_buffer_sequence_length, // 4096
             head_size,
             hidden_size,
             output_current,
             attention_probs + attention_probs_offset,
             pv,
             1.0f,
             0.0f,
             inner_i);
    }
}

template <class Output,
          class Query,
          class Key,
          class Value,
          class Seqlens_K,
          class Attn_Probs,
          class Params>
__device__ void compute_attention_scores(
    Output output, Query, Key, Value, Seqlens_K seqlens_k, Attn_Probs attn_probs, Params params)
{
    const int batch_size      = params.batch_size;
    const int num_heads       = params.num_heads;
    const int sequence_length = params.sequence_length;
    const int head_size       = params.head_size;
    const int elements        = batch_size * num_heads * sequence_length * head_size;
    auto ind                  = make_index();
    ind.global_stride(elements, [&](auto idx) {
        int seqlen_present_kv_cache = params.seqlen_present_kv_cache;
        output([&](auto output0, auto, auto v_cache) {
            const int hidden_size = params.hidden_size;

            CalculateVxAttentionScore(output0.begin(),
                                      attn_probs.begin(),
                                      seqlens_k.begin(),
                                      batch_size,
                                      sequence_length,
                                      seqlen_present_kv_cache,
                                      head_size,
                                      hidden_size,
                                      v_cache.begin(),
                                      params,
                                      idx);
        });
    });
}

} // namespace migraphx
#endif
