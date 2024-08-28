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

    __host__ __device__ void print() const {
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


template<class S, class... Ts>
__device__ RotaryParameters make_rotary_params(S s, Ts... ts)
{
    return {static_cast<float>(s), ts...};
}


template <class Dest,
          class Src>
__device__ void copy_data(Dest destination, const Src source, std::size_t n, std::size_t idx, bool print =false)
{
    if(idx < n)
    {
        if(print)
            // printf("gpu_query%zu: %f\n", idx, static_cast<double>(source[idx]));
            printf("gpu_query%zu\n", idx);
        destination[idx] = source[idx];       
        // auto dest = destination + idx;
        // dest[0] = source[idx];      
    }
}

// template <class Dest,
//           class Src>
// __device__ void copy_data(Dest, Src, std::size_t, std::size_t)
// {
// }

template <class Past,
          class Chunk,
          class Present>
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


template <class K,
          class V,
          class Seqlens_K,
          class KCache,
          class VCache,
          class Params>
__device__ void ConcatPastPresent(
    K k,         
    V v,   
    Seqlens_K seqlens_k,                    
    int batch_size,                     
    int sequence_length,                
    int past_buffer_sequence_length,   
    int present_buffer_sequence_length,
    int head_size,                      
    KCache k_cache,                        
    VCache v_cache,                    
    bool past_present_share_buffer,    
    bool packed_qkv,                  
    Params params,
    int idx)                    
{
    const int num_heads = params.num_heads;
    const int kv_num_heads = params.kv_num_heads;
    const bool is_prompt = sequence_length != 1;
    const int packed_batch_stride =
        packed_qkv ? (num_heads + 2 * kv_num_heads) * sequence_length * head_size : 0;
    const int kv_num_heads_factor = num_heads / kv_num_heads;
    // const size_t q_input_chunk_length =
    //     static_cast<size_t>(sequence_length) * head_size; // S x H
    const size_t kv_input_chunk_length =
        static_cast<size_t>(sequence_length) * head_size; // L x H
    const size_t past_buff_chunk_length =
        static_cast<size_t>(past_buffer_sequence_length) * head_size; // L x H
    const size_t present_buff_chunk_length =
        static_cast<size_t>(present_buffer_sequence_length) * head_size; // T x H

    const int loop_len = batch_size * num_heads;
    auto i = idx / (sequence_length * head_size);
    auto inner_i = idx %  (sequence_length * head_size);
    if(i < loop_len)
    {
        const int batch_index = static_cast<int>(i) / num_heads;
        const int head_index  = static_cast<int>(i) % num_heads;
        const int past_seqlen = sequence_length == 1 ? static_cast<int>(seqlens_k[batch_index])
                                                        : past_buffer_sequence_length;
        const size_t past_chunk_length = static_cast<size_t>(past_seqlen) * head_size;
        // const int output_offset =
        //     static_cast<int>(i) * sequence_length * present_buffer_sequence_length;
        // auto output = attention_probs + output_offset;

        auto current_k = k + packed_batch_stride * batch_index +
                    kv_input_chunk_length * (head_index / kv_num_heads_factor);
        // if(i == 0 and inner_i == 255)
        // {
        //     // auto pk = k_cache + ((i / kv_num_heads_factor) * present_buff_chunk_length);
        //     // printf("gpu_vals%zu, %zu, %zu, %zu\n", present_buff_chunk_length,
        //     //                     past_buff_chunk_length,
        //     //                     past_chunk_length,
        //     //                     kv_input_chunk_length);
        //     for(int j = 0; j < 256; ++j)
        //         printf("gpu_query%d: %f\n", j, static_cast<double>(current_k[j]));
        // }
        // auto k_offset = packed_batch_stride * batch_index +
        //             kv_input_chunk_length * (head_index / kv_num_heads_factor);;
        // if(inner_i == 0)
        // {
        //     printf("gpu_query%d: %zu\n", i, k_offset);
        // }
        auto current_v = v + packed_batch_stride * batch_index +
                kv_input_chunk_length * (head_index / kv_num_heads_factor);
        /* auto pk = */ ConcatStateChunkGQA(k_cache,
                                current_k,
                                k_cache,
                                present_buff_chunk_length,
                                past_buff_chunk_length,
                                past_chunk_length,
                                kv_input_chunk_length,
                                is_prompt,
                                past_present_share_buffer,
                                i / kv_num_heads_factor,
                                inner_i);
        
       /*  auto pv = */ ConcatStateChunkGQA(v_cache,
                                current_v,
                                v_cache,
                                present_buff_chunk_length,
                                past_buff_chunk_length,
                                past_chunk_length,
                                kv_input_chunk_length,
                                is_prompt,
                                past_present_share_buffer,
                                i / kv_num_heads_factor,
                                inner_i);
        // for(int a = 0; a < 1000000; ++a)
        // {

        // }
        // if(i == 0 and inner_i == 0)
        // {
        //     // auto pk = k_cache + ((i / kv_num_heads_factor) * present_buff_chunk_length);
        //     for(int j = 0; j < 256; ++j)
        //         printf("gpu_query%d: %f\n", j, static_cast<double>(pv[j]));
        // }
    }
}

template <class Present,
          class Seqlens_K,
          class Cache,
          class Params>
__device__ void UpdateCache(
    Present present,
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
    int idx)                    
{
    const int num_heads = params.num_heads;
    const int kv_num_heads = params.kv_num_heads;
    const bool is_prompt = sequence_length != 1;
    const int packed_batch_stride =
        packed_qkv ? (num_heads + 2 * kv_num_heads) * sequence_length * head_size : 0;
    const int kv_num_heads_factor = num_heads / kv_num_heads;
    const size_t kv_input_chunk_length =
        static_cast<size_t>(sequence_length) * head_size; // L x H
    const size_t past_buff_chunk_length =
        static_cast<size_t>(past_buffer_sequence_length) * head_size; // L x H
    const size_t present_buff_chunk_length =
        static_cast<size_t>(present_buffer_sequence_length) * head_size; // T x H

    const int loop_len = batch_size * num_heads;
    auto i = idx / (sequence_length * head_size);
    auto inner_i = idx %  (sequence_length * head_size);
    if(i < loop_len)
    {
        const int batch_index = static_cast<int>(i) / num_heads;
        const int head_index  = static_cast<int>(i) % num_heads;
        const int past_seqlen = sequence_length == 1 ? static_cast<int>(seqlens_k[batch_index])
                                                        : past_buffer_sequence_length;
        const size_t past_chunk_length = static_cast<size_t>(past_seqlen) * head_size;

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


template <class Query,
          class Key,
          class Value,
          class Seqlens_K,
          class Params>
__device__ void noop(Query,
                    Key,
                    Value,
                    Seqlens_K,
                    Params) {}


template <class Output,
          class Query,
          class Key,
          class Value,
          class Seqlens_K,
          class Params>
__device__ void concat_past_present(Output output,
                                        Query query,
                                        Key,
                                        Value,
                                        Seqlens_K seqlens_k,
                                        Params params)
{
    const int batch_size      = params.batch_size;
    const int sequence_length = params.sequence_length;
    const int head_size       = params.head_size;
    const int kv_num_heads    = params.kv_num_heads;
    auto ind = make_index();
    auto elements = 2 * batch_size * kv_num_heads * sequence_length * head_size;
    ind.global_stride(elements, [&](auto idx) {
        auto q = query.begin();
        const bool packed_qkv     = true;

        int seqlen_present_kv_cache = params.seqlen_present_kv_cache;
        int seqlen_past_kv_cache    = 4096;

        bool past_present_share_buffer = true;
        auto k = q + params.num_heads * sequence_length * head_size;
        auto v = q + (params.num_heads + params.kv_num_heads) * sequence_length * head_size;
        output([&](auto k_cache, auto v_cache) {
            // noop(query, k_cache, v_cache, seqlens_k, params);
            // if(idx == 0)
            // {
            //     params.print();
            //     // for(int i = 0; i < query.get_shape().elements(); ++i)
            //     // {
            //     //     printf("gpu_query%d: %f\n", i, static_cast<double>(query[i]));
            //     // }
            // }
            // ConcatPastPresent(k,
            //                 v,
            //                 seqlens_k,
            //                 batch_size,
            //                 sequence_length,
            //                 seqlen_past_kv_cache,
            //                 seqlen_present_kv_cache,
            //                 head_size,
            //                 k_cache.begin(),
            //                 v_cache.begin(),
            //                 past_present_share_buffer,
            //                 packed_qkv,
            //                 params,
            //                 idx);
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
            else if (idx < elements)
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
