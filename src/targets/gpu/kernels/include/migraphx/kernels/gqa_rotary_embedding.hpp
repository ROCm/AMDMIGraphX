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

template <class Output,
          class Query,
          class Key,
          class Value,
          class Seqlens_K,
          class Cos_Cache,
          class Sin_Cache,
          class Rotary_QKV,
          class Attn_Probs,
          class Params>
__device__ void no_op(Output,
                        Query,
                        Key,
                        Value,
                        Seqlens_K,
                        Cos_Cache,
                        Sin_Cache,
                        Rotary_QKV,
                        Attn_Probs,
                        Params)
{}

template <class Input,
          class Cos_Cache,
          class Sin_Cache,
          class Output,
          class Pos_IDs,
          class Params>
__device__ void run_rotary_embedding(Input input,
                          Cos_Cache cos_cache,
                          Sin_Cache sin_cache,
                          Output output,
                          bool interleaved,
                          Pos_IDs pos_ids,
                          Params parameters,
                          int idx)
{
    const int batch_size          = parameters.batch_size;
    const int sequence_length     = parameters.sequence_length;
    const int n_heads             = parameters.num_heads;
    const int head_size           = parameters.head_size;
    const int head_stride         = parameters.head_stride;
    const int seq_stride          = parameters.seq_stride;
    const int batch_stride        = parameters.batch_stride;
    const int position_ids_format = parameters.position_ids_format;
    const int rotary_emb_dim      = parameters.rotary_embedding_dim;
    const int half_rotary_emb_dim = rotary_emb_dim / 2;


    const int loop_len = batch_size * sequence_length * n_heads;
    if (idx < loop_len)
    {
        // printf("%d < %d\n", static_cast<int>(idx), loop_len);
        const int b            = static_cast<int>((idx / n_heads) / sequence_length);
        const int s            = static_cast<int>((idx / n_heads) % sequence_length);
        const int n            = static_cast<int>(idx % n_heads);
        const int block_offset = b * batch_stride + s * seq_stride + n * head_stride;
        // printf("block offset: %d\n", block_offset);
        auto input_data        = input + block_offset;
        auto output_data       = output + block_offset;

        // Cache is (M, H/2) or (M, rotary_embedding_dim/2)
        int position_id = 0;
        if (sequence_length == 1)
        {
          position_id = (position_ids_format == 0)
                                      ? static_cast<int>(pos_ids[0]) + s
                                      : static_cast<int>(pos_ids[b * sequence_length + s]);
        }

        const int cache_offset = position_id * half_rotary_emb_dim;
        auto cos_data          = cos_cache + cache_offset;
        auto sin_data          = sin_cache + cache_offset;

        int cache_idx = 0;
        float sign    = 0.0;
        int j         = 0;
        for(int i = 0; i < rotary_emb_dim; i++)
        {
            if(interleaved)
            {
                cache_idx = (i / 2) % half_rotary_emb_dim;
                sign      = (i % 2 == 0) ? -1.0 : 1.0;
                j         = (i % 2 == 0) ? i + 1 : i - 1; // i - sign
            }
            else
            {
                cache_idx = i % half_rotary_emb_dim;
                sign      = (i < half_rotary_emb_dim) ? -1.0 : 1.0;
                j         = (i + half_rotary_emb_dim) % rotary_emb_dim;
            }
            output_data[i] = input_data[i] * cos_data[cache_idx] +
                              sign * input_data[j] * sin_data[cache_idx];
        }
        for(int i = rotary_emb_dim; i < head_size; i++)
        {
            output_data[i] = input_data[i];
        }
    }
    
}

template <class Params, class Input, class Output>
__device__ void pack_v_into_rotary_QKV(Params parameters, const Input input, Output output, int idx)
{
    const int loop_len = parameters.batch_size * parameters.sequence_length * parameters.kv_num_heads;

    if (idx < loop_len)
    {
        const int b = static_cast<int>((idx / parameters.kv_num_heads) / parameters.sequence_length);
        const int s = static_cast<int>((idx / parameters.kv_num_heads) % parameters.sequence_length);
        const int n = static_cast<int>(idx % parameters.kv_num_heads);
        const int block_offset = b * parameters.batch_stride + s * parameters.seq_stride +
                                    n * parameters.head_stride;
        const Input input_data = input + block_offset;
        Output output_data      = output + block_offset;
        for(int i = 0; i < parameters.head_size; i++)
        {
            output_data[i] = input_data[i];
        }
    }
}



template<class... T, class... U>
__device__ void no_op(T..., U...) {}

template <class T, class U>
__device__ void no_op(T,
                        T,
                        T,
                        T,
                        // T present_key,
                        // T present_value,
                        U,
                        T,
                        RotaryParameters,
                        int) {}

__device__ void sync()
{
    // __syncthreads();
    __builtin_amdgcn_s_waitcnt(0xc07f);
    __builtin_amdgcn_s_barrier();
}

// kernel inputs = query, past_present_key, past_present_value, cos_cache, sin_cache, rotary_qkv, attn_probs, seqlens_k
template <class Output,
          class Query,
          class Seqlens_K,
          class Cos_Cache,
          class Sin_Cache,
          class Params>
__device__ void gqa_rotary_embedding(Output output,
                                        Query query,
                                        Seqlens_K seqlens_k,
                                        Cos_Cache cos_cache,
                                        Sin_Cache sin_cache,
                                        Params params)
{

    auto q_input  = query.begin();
    auto q_rotary = output.begin();
    auto k_input  = q_input + params.num_heads * params.sequence_length * params.head_size;
    auto k_rotary = q_rotary + params.num_heads * params.sequence_length * params.head_size;
    auto ind = make_index();
    ind.global_stride(query.get_shape().elements(), [&](auto idx) {
        run_rotary_embedding(q_input,
                              cos_cache.begin(),
                              sin_cache.begin(),
                              q_rotary,
                              params.rotary_interleaved,
                              seqlens_k.begin(),
                              params,
                              idx);
        
        run_rotary_embedding(k_input,
                                cos_cache.begin(),
                                sin_cache.begin(),
                                k_rotary,
                                params.rotary_interleaved,
                                seqlens_k.begin(),
                                params,
                                idx);

        auto v_input  = k_input + params.kv_num_heads * params.sequence_length * params.head_size;
        auto v_rotary = k_rotary + params.kv_num_heads * params.sequence_length * params.head_size;

    
        pack_v_into_rotary_QKV(params, v_input, v_rotary, idx);
    });
}

} // namespace migraphx
#endif
