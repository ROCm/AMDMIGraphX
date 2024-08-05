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

// template <class... Ts>
// using common_type = typename std::common_type<Ts...>::type;

template <class T>
__device__ bool float_equal(T x, T y)
{
    return isfinite(x) and isfinite(y) and
           nextafterf(x, numeric_lowest<T>()) <= y and
           nextafterf(x, numeric_max<T>()) >= y;
}

// template <class T, class U>
// __device__ bool float_equal(T x, U y)
// {
//     return float_equal_device<common_type<T, U>>(x, y);
// }

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
    // if (idx < loop_len)
    //     printf("%d < %d\n", idx, loop_len);
    // else 
    //     printf("%d >= %d\n", idx, loop_len);
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

template <class Dest,
          class Src>
__device__ void copy_data(Dest destination, const Src source, std::size_t n, std::size_t idx)
{
    if(idx < n)
    {
        destination[idx] = source[idx];
    }
}

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

template <class T, class A, class B, class F>
__device__ void gemm(std::size_t M, std::size_t N, std::size_t K, std::size_t lda, std::size_t ldb, std::size_t ldc, T cmat, A amat, B bmat, F alpha, F beta, std::size_t idx, const bool b_transpose = false)
{
    auto m = idx / N;
    auto n = idx % N;
    auto a_idx = [&](auto ii, auto kk){ return kk + (ii * lda); };
    auto b_idx = [&](auto kk, auto jj){ return jj + (kk * ldb); };
    auto bt_idx = [&](auto kk, auto jj){ return jj + (kk * ldb); };
    auto c_idx = [&](auto ii, auto jj){ return jj + (ii * ldc); };

    if (m < M)
    {
        if (n < N)
        {
            double s = 0.0;
            for (int k = 0; k < K; ++k)
            {
                auto a_i = a_idx(m, k);
                auto b_i = b_transpose ? bt_idx(n, k) : b_idx(k, n);
                s += static_cast<double>(amat[a_i]) *
                     static_cast<double>(bmat[b_i]);
            }
            auto c_i = c_idx(m, n);
            cmat[c_i] = static_cast<double>(alpha) * s + cmat[c_i] * static_cast<double>(beta);
        }
    }
}

template <class T>
__device__ void CalculateAttentionSoftmaxInplace(T score, int N, int D)
{
    // par_for(N, [&](const auto j) {
    for (int j = 0; j < N; ++j)
    {
        auto x = score + j * D;
        auto y = x;

        // e^x is represented as infinity if x is large enough, like 100.f.
        // Infinity divided by Infinity is a NAN. Thus, softmax gets a NAN if
        // one or more item are large enough. a math transform as below is
        // leveraged to get a stable softmax: e^xi/(e^x1 + ...e^xn) = e^(xi -
        // max) / (e^(x1 - max) + ... + e^(xn - max))
        float max = -numeric_max<float>();
        for(int i = 0; i < D; i++)
        {
            if(max < x[i])
                max = x[i];
        }
        for(int i = 0; i < D; i++)
        {
            y[i] = expf(x[i] - max);
        }

        float sum = 0.0;
        const float zero = 0.0;
        for(int i = 0; i < D; i++)
        {
            sum += x[i];
        }

        if(float_equal(sum, zero))
        {
            for(int i = 0; i < D; i++)
            {
                y[i] = 1.0f / static_cast<float>(D);
            }
        }
        else
        {
            for(int i = 0; i < D; i++)
            {
                y[i] = x[i] / static_cast<float>(sum);
            }
        }
    }
}

template <class Attn_Probs,
          class Q_,
          class K_,
          class SeqLens,
          class PastKey,
          class Params>
__device__ void CalculateAttentionProbs(
    Attn_Probs attention_probs,                  // output buffer with size BxNxSxT
    Q_ Q,                                // Q data. Its size is BxNxSxH
    K_ K,                                // k data. Its size is BxNxLxH
    SeqLens seqlens_k,                        // past sequence lengths tensor
    int batch_size,                     // batch size of self-attention
    int sequence_length,                // sequence length of self-attention (S)
    int past_buffer_sequence_length,    // sequence length of past state
    int present_buffer_sequence_length, // sequence length of present state
    int head_size,                      // head size of self-attention
    PastKey past_key,                         // past key only
    PastKey present_key,                      // present key only
    bool past_present_share_buffer,     // whether present key and value share the same buffer
    bool packed_qkv,                    // whether Q, K, V are packed
    Params params,
    int idx)                    
{
    const int num_heads = params.num_heads;
    const int kv_num_heads = params.kv_num_heads;
    const bool is_prompt = sequence_length != 1;
    const int packed_batch_stride =
        packed_qkv ? (num_heads + 2 * kv_num_heads) * sequence_length * head_size : 0;
    const int kv_num_heads_factor = num_heads / kv_num_heads;
    const size_t q_input_chunk_length =
        static_cast<size_t>(sequence_length) * head_size; // S x H
    const size_t kv_input_chunk_length =
        static_cast<size_t>(sequence_length) * head_size; // L x H
    const size_t past_buff_chunk_length =
        static_cast<size_t>(past_buffer_sequence_length) * head_size; // L x H
    const size_t present_buff_chunk_length =
        static_cast<size_t>(present_buffer_sequence_length) * head_size; // T x H

    const int loop_len = batch_size * num_heads;
    const float alpha  = params.scale == 0.0f ? 1.0f / sqrt(static_cast<float>(head_size)) : params.scale;
    const int max_sequence_length = 4096;
    // par_for(loop_len, [&](const auto i) {
    // auto ind = make_index();
    // auto idx = ind.global;
    auto i = idx / (sequence_length * head_size * max_sequence_length);
    auto inner_i = idx %  (sequence_length * max_sequence_length * head_size);
    if(i < loop_len)
    {
        // printf("%d, %d of %d\n", i, inner_i, loop_len);
        const int batch_index = static_cast<int>(i) / num_heads;
        const int head_index  = static_cast<int>(i) % num_heads;
        const int past_seqlen = sequence_length == 1 ? static_cast<int>(seqlens_k[batch_index])
                                                        : past_buffer_sequence_length;
        const size_t past_chunk_length = static_cast<size_t>(past_seqlen) * head_size;
        const int total_seqlen         = seqlens_k[batch_index] + 1;

        const int output_offset =
            static_cast<int>(i) * sequence_length * present_buffer_sequence_length;
        auto output = attention_probs + output_offset;

        auto k = K + packed_batch_stride * batch_index +
                    kv_input_chunk_length * (head_index / kv_num_heads_factor);

        PastKey pk;
        if (inner_i < (head_size * max_sequence_length))
        {
            pk = ConcatStateChunkGQA(past_key,
                                k,
                                present_key,
                                present_buff_chunk_length,
                                past_buff_chunk_length,
                                past_chunk_length,
                                kv_input_chunk_length,
                                is_prompt,
                                past_present_share_buffer,
                                i / kv_num_heads_factor,
                                inner_i);
        }
        
        // sync();
        // Calculate Q*K' + AttentionMask
        //                     original                 transposed             each iteration
        // A: Q                (B x N x) S x H          (B x N x) S x H        S x H
        // B: K'               (B x N x) T x H          (B x N x) H x T        H x T
        // C: attention_probs  (B x N x) S x T          (B x N x) S x T        S x T
        Q_ q;
        if(packed_qkv)
        {
            q = Q + packed_batch_stride * batch_index + q_input_chunk_length * head_index;
        }
        else
        {
            q = Q + q_input_chunk_length * i;
        }

        if (inner_i < (sequence_length * total_seqlen))
        {
            gemm(sequence_length,
                total_seqlen,
                head_size,
                head_size,
                head_size,
                present_buffer_sequence_length,
                output,
                q,
                pk, ////
                alpha,
                0.0f,
                inner_i,
                true);
        }
        // sync();

        const int local_window_size = params.local_window_size;
        auto output_softmax = output;
        // for(int seq = 0; seq < sequence_length; seq++)
        int seq = inner_i;
        if (inner_i < sequence_length)
        {
            int seq_causal_length = sequence_length == 1 ? total_seqlen : seq + 1;
            if(local_window_size > 0 && seq_causal_length > local_window_size + 1)
            {
                for(int total_seq_id = 0;
                    total_seq_id < seq_causal_length - local_window_size - 1;
                    total_seq_id++)
                {
                    output_softmax[total_seq_id] = 0.f;
                }
                CalculateAttentionSoftmaxInplace(output_softmax + seq_causal_length -
                                                        local_window_size - 1,
                                                    1,
                                                    local_window_size + 1);
            }
            else
            {
                CalculateAttentionSoftmaxInplace(output_softmax, 1, seq_causal_length);
            }
            // set causal [seq_causal_length, total_seqlen) to 0.f
            for(int total_seq_id = seq_causal_length; total_seq_id < total_seqlen;
                total_seq_id++)
            {
                output_softmax[total_seq_id] = 0.f;
            }

            output_softmax += present_buffer_sequence_length;
        }
        // sync();
    }
}

template <class Output, 
          class Attn_Probs,
          class V_,
          class SeqLens,
          class PastValue,
          class Params>
__device__ void CalculateVxAttentionScore(
        Output output,                           // buffer for the result with size BxSxNxH
        const Attn_Probs attention_probs,            // Attention probs with size BxNxSxT
        const V_ V,                          // V value with size BxN_kvxSxH
        const SeqLens seqlens_k,                  // past sequence lengths tensor
        int batch_size,                     // batch size
        int sequence_length,                // sequence length
        int past_buffer_sequence_length,    // sequence length in past state
        int present_buffer_sequence_length, // sequence length in past state
        int head_size,                      // head size of Q, K, V
        int hidden_size,                    // hidden size of Output
        PastValue past_value,                 // past value only
        PastValue present_value,                    // present value only
        bool past_present_share_buffer,     // whether present key and value share the same buffer
        bool packed_qkv,                    // whether Q, K, V are packed
        Params params,
        int idx)  
{
    const int num_heads = params.num_heads;
    const int kv_num_heads = params.kv_num_heads;
    const bool is_prompt = sequence_length != 1;
    const int packed_batch_stride =
        packed_qkv ? (num_heads + 2 * kv_num_heads) * sequence_length * head_size : 0;
    const int kv_num_heads_factor   = num_heads / kv_num_heads;
    const int kv_input_chunk_length = sequence_length * head_size; // L x H
    const size_t past_buff_chunk_length =
        static_cast<size_t>(past_buffer_sequence_length) * head_size; // L x H
    const size_t present_buff_chunk_length =
        static_cast<size_t>(present_buffer_sequence_length) * head_size; // T x H
    const int max_sequence_length = 4096;

    auto loop_len = batch_size * num_heads;
    auto i = idx / (sequence_length * head_size * max_sequence_length);
    auto inner_i = idx %  (sequence_length * max_sequence_length * head_size);
    if (i < loop_len)
    {
        const int batch_index = static_cast<int>(i / num_heads);
        const int head_index  = static_cast<int>(i % num_heads);
        const int past_seqlen = sequence_length == 1 ? static_cast<int>(seqlens_k[batch_index])
                                                        : past_buffer_sequence_length;
        const size_t past_chunk_length = static_cast<size_t>(past_seqlen) * head_size;
        const int total_seqlen         = seqlens_k[batch_index] + 1;
        // printf("%d: ts = %d\n", i, total_seqlen);
        V_ v;
        if(packed_qkv)
        {
            v = V + packed_batch_stride * batch_index +
                kv_input_chunk_length * (head_index / kv_num_heads_factor);
        }
        else
        {
            v = V + kv_input_chunk_length * (i / kv_num_heads_factor);
        }
        
        PastValue pv;
        if (inner_i < (head_size * max_sequence_length))
        {
            pv = ConcatStateChunkGQA(past_value,
                                v,
                                present_value,
                                present_buff_chunk_length,
                                past_buff_chunk_length,
                                past_chunk_length,
                                kv_input_chunk_length,
                                is_prompt,
                                past_present_share_buffer,
                                i / kv_num_heads_factor,
                                inner_i);
        }

        Output output_current =
            output + (batch_index * sequence_length * num_heads + head_index) * head_size;
        ptrdiff_t attention_probs_offset = sequence_length * present_buffer_sequence_length * i;

        if (inner_i < (sequence_length * head_size))
        {
            gemm(sequence_length,
                head_size,
                total_seqlen,
                present_buffer_sequence_length,
                head_size,
                hidden_size,
                output_current,
                attention_probs + attention_probs_offset,
                pv, /////
                1.0f,
                0.0f,
                inner_i);
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
          class Key,
          class Value,
          class Seqlens_K,
        //   class Cos_Cache,
        //   class Sin_Cache,
        //   class Rotary_QKV,
          class Attn_Probs,
          class Params>
__device__ void compute_attention_scores(Output output,
                                        Query query,
                                        Key key,
                                        Value value,
                                        Seqlens_K seqlens_k,
                                        // Cos_Cache cos_cache,
                                        // Sin_Cache sin_cache,
                                        // Rotary_QKV rotary_qkv,
                                        Attn_Probs attn_probs,
                                        Params params)
{
    // no_op(output, query, key, value, seqlens_k, cos_cache, sin_cache, /* rotary_qkv, */ attn_probs, params);
    // MIGRAPHX_ASSERT(query.begin() != query.end());
    // MIGRAPHX_ASSERT(query.size() != 0);
    // MIGRAPHX_ASSERT(output.begin() != output.end());
    // MIGRAPHX_ASSERT(cos_cache.begin() != cos_cache.end());
    // MIGRAPHX_ASSERT(sin_cache.begin() != sin_cache.end());
    // MIGRAPHX_ASSERT(rotary_qkv.begin() != rotary_qkv.end());
    // MIGRAPHX_ASSERT(rotary_qkv.size() != 0);
    // MIGRAPHX_ASSERT(seqlens_k.begin() != seqlens_k.end());

    // auto q_input  = query.begin();
    // auto q_rotary = rotary_qkv.begin();
    // auto k_input  = q_input + params.num_heads * params.sequence_length * params.head_size;
    // auto k_rotary = q_rotary + params.num_heads * params.sequence_length * params.head_size;
    auto ind = make_index();
    ind.global_stride(query.get_shape().elements() * 4096, [&](auto idx) {
        auto q = query.begin();
        const int batch_size      = params.batch_size;
        const int sequence_length = params.sequence_length;
        const int head_size       = params.head_size;
        const bool packed_qkv     = true;

        int seqlen_present_kv_cache = params.seqlen_present_kv_cache;
        int seqlen_past_kv_cache    = 4096;

        // // Calculate the attention score.
        bool past_present_share_buffer = false;//true;
        // auto k                      = q + params.num_heads * sequence_length * head_size;
        // sync();
        output([&](auto output0, auto output1, auto output2) {
        //     CalculateAttentionProbs(attn_probs.begin(),
        //                             q,
        //                             k,
        //                             seqlens_k.begin(),
        //                             batch_size,
        //                             sequence_length,
        //                             seqlen_past_kv_cache,
        //                             seqlen_present_kv_cache,
        //                             head_size,
        //                             key.begin(),
        //                             // key.begin(),
        //                             output1.begin(),
        //                             past_present_share_buffer,
        //                             packed_qkv,
        //                             params,
        //                             idx);

            // sync();
            auto v = q + (params.num_heads + params.kv_num_heads) * sequence_length * head_size;
            const int hidden_size     = params.hidden_size;
        
            CalculateVxAttentionScore(output0.begin(),
                                    attn_probs.begin(),
                                    v,
                                    seqlens_k.begin(),
                                    batch_size,
                                    sequence_length,
                                    seqlen_past_kv_cache,
                                    seqlen_present_kv_cache,
                                    head_size,
                                    hidden_size,
                                    value.begin(),
                                    // value.begin(),
                                    output2.begin(),
                                    past_present_share_buffer,
                                    packed_qkv,
                                    params,
                                    idx);
            // sync();
            // if (idx < key.get_shape().elements())
            // {
            //     output1[idx] = key[idx];
            //     output2[idx] = value[idx];
            // }
            // sync();
            if (idx == 0)
            {
                output1 = key;
                // output2 = value;
            }
        });
    });
}

} // namespace migraphx
#endif
