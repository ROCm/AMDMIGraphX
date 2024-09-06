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
#ifndef MIGRAPHX_GUARD_KERNELS_CONCAT_PAST_PRESENT_HPP
#define MIGRAPHX_GUARD_KERNELS_CONCAT_PAST_PRESENT_HPP

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

template <class Dest, class Src>
__device__ void copy_data(Dest destination, const Src source, index_int n, index_int idx)
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
                                       index_int present_buff_chunk_length,
                                       index_int past_buff_chunk_length,
                                       index_int past_chunk_length,
                                       index_int new_chunk_length,
                                       bool is_prompt,
                                       bool past_present_share_buffer,
                                       std::ptrdiff_t i,
                                       index_int idx)
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

template <class Present, class Seqlens_K, class Cache, class Params>
__device__ void UpdateCache(Present present,
                            Seqlens_K seqlens_k,
                            int batch_size,
                            int sequence_length,
                            int past_buffer_sequence_length,
                            int present_buffer_sequence_length,
                            int head_size,
                            Cache cache,
                            bool past_present_share_buffer,
                            bool packed_qkv,
                            Params params,
                            index_int idx)
{
    const int num_heads    = params.num_heads;
    const int kv_num_heads = params.kv_num_heads;
    const bool is_prompt   = sequence_length != 1;
    const int packed_batch_stride =
        packed_qkv ? (num_heads + 2 * kv_num_heads) * sequence_length * head_size : 0;
    const int kv_num_heads_factor      = num_heads / kv_num_heads;
    const size_t kv_input_chunk_length = static_cast<size_t>(sequence_length) * head_size; // L x H
    const size_t past_buff_chunk_length =
        static_cast<size_t>(past_buffer_sequence_length) * head_size; // L x H
    const size_t present_buff_chunk_length =
        static_cast<size_t>(present_buffer_sequence_length) * head_size; // T x H

    const index_int loop_len = batch_size * num_heads;
    const index_int i        = idx / (sequence_length * head_size);
    const index_int inner_i  = idx % (sequence_length * head_size);
    if(i < loop_len)
    {
        const index_int batch_index = i / num_heads;
        const index_int head_index  = i % num_heads;
        const index_int past_seqlen = sequence_length == 1
                                          ? static_cast<int>(seqlens_k[batch_index])
                                          : past_buffer_sequence_length;
        const index_int past_chunk_length = static_cast<size_t>(past_seqlen) * head_size;

        auto current = present + packed_batch_stride * batch_index +
                       kv_input_chunk_length * (head_index / kv_num_heads_factor);
        ConcatStateChunkGQA(cache,
                            current,
                            cache,
                            present_buff_chunk_length,
                            past_buff_chunk_length,
                            past_chunk_length,
                            kv_input_chunk_length,
                            is_prompt,
                            past_present_share_buffer,
                            i / kv_num_heads_factor,
                            inner_i);
    }
}

template <class Output, class Query, class Key, class Value, class Seqlens_K, class Params>
__device__ void
concat_past_present(Output output, Query query, Key, Value, Seqlens_K seqlens_k, Params params)
{
    const int batch_size      = params.batch_size;
    const int sequence_length = params.sequence_length;
    const int head_size       = params.head_size;
    const int kv_num_heads    = params.kv_num_heads;
    auto ind                  = make_index();
    auto elements             = 2 * batch_size * kv_num_heads * sequence_length * head_size;
    ind.global_stride(elements, [&](auto idx) {
        auto q                = query.begin();
        const bool packed_qkv = true;

        int seqlen_present_kv_cache = params.seqlen_present_kv_cache;
        int seqlen_past_kv_cache    = 4096;

        bool past_present_share_buffer = true;
        auto k                         = q + params.num_heads * sequence_length * head_size;
        auto v = q + (params.num_heads + params.kv_num_heads) * sequence_length * head_size;
        output([&](auto k_cache, auto v_cache) {
            if(idx < elements / 2)
            {
                UpdateCache(k,
                            seqlens_k,
                            batch_size,
                            sequence_length,
                            seqlen_past_kv_cache,
                            seqlen_present_kv_cache,
                            head_size,
                            k_cache.begin(),
                            past_present_share_buffer,
                            packed_qkv,
                            params,
                            idx);
            }
            else if(idx < elements)
            {
                UpdateCache(v,
                            seqlens_k,
                            batch_size,
                            sequence_length,
                            seqlen_past_kv_cache,
                            seqlen_present_kv_cache,
                            head_size,
                            v_cache.begin(),
                            past_present_share_buffer,
                            packed_qkv,
                            params,
                            idx - (elements / 2));
            }
        });
    });
}

} // namespace migraphx
#endif
