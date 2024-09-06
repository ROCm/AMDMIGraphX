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
#ifndef MIGRAPHX_GUARD_KERNELS_COMPUTE_ATTENTION_PROBABILITIES_HPP
#define MIGRAPHX_GUARD_KERNELS_COMPUTE_ATTENTION_PROBABILITIES_HPP

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


template <class Attn_Probs,
          class Q_,
          class SeqLens,
          class PresentKey,
          class Params>
__device__ void
CalculateAttentionProbs(Attn_Probs attention_probs,         // output buffer with size BxNxSxT
                        Q_ Q,                               // Q data. Its size is BxNxSxH
                        SeqLens seqlens_k,                  // past sequence lengths tensor
                        int batch_size,                     // batch size of self-attention
                        int sequence_length,                // sequence length of self-attention (S)
                        int present_buffer_sequence_length, // sequence length of present state
                        int head_size,                      // head size of self-attention
                        PresentKey present_key,             // present key only
                        bool packed_qkv,                    // whether Q, K, V are packed
                        Params params,
                        index_int idx)
{
    const int num_heads    = params.num_heads;
    const int kv_num_heads = params.kv_num_heads;
    const int packed_batch_stride =
        packed_qkv ? (num_heads + 2 * kv_num_heads) * sequence_length * head_size : 0;
    const int kv_num_heads_factor     = num_heads / kv_num_heads;
    const size_t q_input_chunk_length = static_cast<size_t>(sequence_length) * head_size; // S x H
    const size_t present_buff_chunk_length =
        static_cast<size_t>(present_buffer_sequence_length) * head_size; // T x H

    const index_int loop_len = batch_size * num_heads;
    const float alpha =
        params.scale == 0.0f ? 1.0f / sqrt(static_cast<float>(head_size)) : params.scale;
    const int max_sequence_length = 4096;

    const index_int i       = idx / (sequence_length * max_sequence_length);
    const index_int inner_i = idx % (sequence_length * max_sequence_length);
    if(i < loop_len)
    {
        const auto batch_index  = i / num_heads;
        const auto head_index   = i % num_heads;
        const int total_seqlen = seqlens_k[batch_index] + 1;
        const index_int output_offset =
            i * sequence_length * present_buffer_sequence_length;
        auto output = attention_probs + output_offset;
        auto pk     = present_key + ((i / kv_num_heads_factor) * present_buff_chunk_length);
        Q_ q;
        if(packed_qkv)
        {
            q = Q + packed_batch_stride * batch_index + q_input_chunk_length * head_index;
        }
        else
        {
            q = Q + q_input_chunk_length * i;
        }
        gemm(sequence_length,
             total_seqlen,
             head_size,
             head_size,
             head_size,
             present_buffer_sequence_length, // 4096
             output,
             q,
             pk,
             alpha,
             0.0f,
             inner_i,
             true);
    }
}

template <class Output, class Query, class Key, class Value, class Seqlens_K, class Params>
__device__ void compute_attention_probabilities(
    Output output, Query query, Key, Value, Seqlens_K seqlens_k, Params params)
{
    auto ind = make_index();
    ind.global_stride(params.batch_size * params.num_heads * params.sequence_length * params.seqlen_present_kv_cache, [&](auto idx) {
        auto q                    = query.begin();
        const int batch_size      = params.batch_size;
        const int sequence_length = params.sequence_length;
        const int head_size       = params.head_size;
        const bool packed_qkv     = true;

        int seqlen_present_kv_cache = params.seqlen_present_kv_cache;
        output([&](auto output0, auto k_cache, auto) {
            CalculateAttentionProbs(output0.begin(),
                                    q,
                                    seqlens_k.begin(),
                                    batch_size,
                                    sequence_length,
                                    seqlen_present_kv_cache,
                                    head_size,
                                    k_cache.begin(),
                                    packed_qkv,
                                    params,
                                    idx);
        });
    });
}

} // namespace migraphx
#endif
