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
#ifndef MIGRAPHX_GUARD_KERNELS_GROUP_QUERY_ATTENTION_HPP
#define MIGRAPHX_GUARD_KERNELS_GROUP_QUERY_ATTENTION_HPP

#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/algorithm.hpp>
#include <migraphx/kernels/integral_constant.hpp>
#include <migraphx/kernels/tensor_view.hpp>
#include <migraphx/kernels/ck.hpp>
#include <migraphx/kernels/gemm_batcher.hpp>
#include <limits>
#include <migraphx/kernels/type_traits.hpp>

namespace migraphx {

struct RotaryParameters
{
    float scale;
    int batch_size;           // Batch size used by input
    int sequence_length;      // Sequence length used by input
    int hidden_size;          // Hidden size used by input
    int head_size;            // Head size
    int rotary_embedding_dim; // Rotary embedding dimension.
    int num_heads;            // num_heads = hidden_size / head_size
    int max_sequence_length;  // Sequence length used by cos/sin cache
    int head_stride;          // Head stride
    int seq_stride;           // Sequence stride
    int batch_stride;         // Batch stride
    int position_ids_format;  // Format of position ids - 0 is (1), 1 is (batch_size,
                              // sequence_length)
    int transposed; // Whether the input tensor has been transposed into (batch, num_heads,
                    // seq_len, hidden)
    int seqlen_present_kv_cache;

    int do_rotary;
    int kv_num_heads;
    int local_window_size;
    int rotary_interleaved;

    __host__ __device__ void print() const
    {
        printf("scale: %f\n", scale);
        printf("batch_size: %d\n", batch_size);
        printf("sequence_length: %d\n", sequence_length);
        printf("hidden_size: %d\n", hidden_size);
        printf("head_size: %d\n", head_size);
        printf("rotary_embedding_dim: %d\n", rotary_embedding_dim);
        printf("num_heads: %d\n", num_heads);
        printf("max_sequence_length: %d\n", max_sequence_length);
        printf("head_stride: %d\n", head_stride);
        printf("seq_stride: %d\n", seq_stride);
        printf("batch_stride: %d\n", batch_stride);
        printf("position_ids_format: %d\n", position_ids_format);
        printf("transposed: %d\n", transposed);
        printf("seqlen_present_kv_cache: %d\n", seqlen_present_kv_cache);
        printf("do_rotary: %d\n", do_rotary);
        printf("kv_num_heads: %d\n", kv_num_heads);
        printf("local_window_size: %d\n", local_window_size);
        printf("rotary_interleaved: %d\n", rotary_interleaved);
    }
};

template <class S, class... Ts>
__device__ RotaryParameters make_rotary_params(S s, Ts... ts)
{
    return {static_cast<float>(s), ts...};
}

template <class Dest, class Src>
__device__ void copy_data(Dest destination, const Src source, std::size_t n, std::size_t idx)
{
    if(idx < n)
    {
        destination[idx] = source[idx];
    }
}

template <class Past, class Chunk, class Present>
__device__ Present ConcatStateChunkGQA(const Past past,
                                       Chunk chunk,
                                       Present present,
                                       size_t present_buff_chunk_length,
                                       size_t past_buff_chunk_length,
                                       size_t past_chunk_length,
                                       size_t new_chunk_length,
                                       bool is_prompt,
                                       bool past_present_share_buffer,
                                       std::ptrdiff_t i,
                                       size_t idx)
{
    auto start = present + i * present_buff_chunk_length;

    auto p = start;
    if(!is_prompt)
    {
        if(!past_present_share_buffer)
        {
            const auto src_past = past + i * past_buff_chunk_length;
            copy_data(p, src_past, past_chunk_length, idx);
        }
        p += past_chunk_length;
    }
    copy_data(p, chunk, new_chunk_length, idx);
    return start;
}

template <class C, class A, class B, class F>
__device__ void gemm(std::size_t M,
                     std::size_t N,
                     std::size_t K,
                     std::size_t lda,
                     std::size_t ldb,
                     std::size_t ldc,
                     C cmat,
                     A amat,
                     B bmat,
                     F alpha,
                     F beta,
                     std::size_t idx,
                     const bool b_transpose = false)
{
    auto m      = idx / N;
    auto n      = idx % N;
    auto a_idx  = [&](auto ii, auto kk) { return kk + (ii * lda); };
    auto b_idx  = [&](auto kk, auto jj) { return jj + (kk * ldb); };
    auto bt_idx = [&](auto kk, auto jj) { return jj + (kk * ldb); };
    auto c_idx  = [&](auto ii, auto jj) { return jj + (ii * ldc); };

    if(m < M)
    {
        if(n < N)
        {
            double s = 0.0;
            for(int k = 0; k < K; ++k)
            {
                auto a_i = a_idx(m, k);
                auto b_i = b_transpose ? bt_idx(n, k) : b_idx(k, n);
                s += static_cast<double>(amat[a_i]) * static_cast<double>(bmat[b_i]);
            }
            auto c_i  = c_idx(m, n);
            cmat[c_i] = static_cast<double>(alpha) * s + cmat[c_i] * static_cast<double>(beta);
        }
    }
}

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
                        int idx)
{
    const int num_heads    = params.num_heads;
    const int kv_num_heads = params.kv_num_heads;
    const int packed_batch_stride =
        packed_qkv ? (num_heads + 2 * kv_num_heads) * sequence_length * head_size : 0;
    const int kv_num_heads_factor     = num_heads / kv_num_heads;
    const size_t q_input_chunk_length = static_cast<size_t>(sequence_length) * head_size; // S x H
    const size_t present_buff_chunk_length =
        static_cast<size_t>(present_buffer_sequence_length) * head_size; // T x H

    const int loop_len = batch_size * num_heads;
    const float alpha =
        params.scale == 0.0f ? 1.0f / sqrt(static_cast<float>(head_size)) : params.scale;
    const int max_sequence_length = 4096;

    auto i       = idx / (sequence_length * max_sequence_length);
    auto inner_i = idx % (sequence_length * max_sequence_length);
    if(i < loop_len)
    {
        const int batch_index  = static_cast<int>(i) / num_heads;
        const int head_index   = static_cast<int>(i) % num_heads;
        const int total_seqlen = seqlens_k[batch_index] + 1;
        const int output_offset =
            static_cast<int>(i) * sequence_length * present_buffer_sequence_length;
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
    ind.global_stride((query.get_shape().elements() / params.head_size) * 4096, [&](auto idx) {
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
