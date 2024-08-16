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


template <class T>
__device__ bool float_equal(T x, T y)
{
    return isfinite(x) and isfinite(y) and
           nextafterf(x, numeric_lowest<T>()) <= y and
           nextafterf(x, numeric_max<T>()) >= y;
}


template<class S, class... Ts>
__device__ RotaryParameters make_rotary_params(S s, Ts... ts)
{
    return {static_cast<float>(s), ts...};
}

template <class T>
__device__ void CalculateAttentionSoftmaxInplace(T score, int N, int D)
{
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
          class SeqLens,
          class Params>
__device__ void CalculateSoftmax(
    Attn_Probs attention_probs,                  // output buffer with size BxNxSxT
    SeqLens seqlens_k,                        // past sequence lengths tensor
    int batch_size,                     // batch size of self-attention
    int sequence_length,                // sequence length of self-attention (S)
    int present_buffer_sequence_length, // sequence length of present state
    Params params,
    int idx)                    
{
    const int num_heads = params.num_heads;

    const int loop_len = batch_size * num_heads;
    auto i = idx / sequence_length;
    auto inner_i = idx % sequence_length;
    if(i < loop_len)
    {
        const int batch_index = static_cast<int>(i) / num_heads;
        const int total_seqlen         = seqlens_k[batch_index] + 1;
        const int output_offset =
            static_cast<int>(i) * sequence_length * present_buffer_sequence_length;
        auto output = attention_probs + output_offset;
        
        const int local_window_size = params.local_window_size;
        auto output_softmax = output;
        int seq = inner_i;
        if (seq < sequence_length)
        {
            output_softmax += seq * present_buffer_sequence_length;
            auto consume = total_seqlen + local_window_size;
            seq += consume;
            seq -= consume;
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
        }
    }
}

template <class Output,
          class Pass,
          class Input,
          class Seqlens_K,
          class Params>
__device__ void gqa_softmax(Output output,
                                        Pass,
                                        Input,
                                        Seqlens_K seqlens_k,
                                        Params params)
{
    auto ind = make_index();
    ind.global_stride(output.get_shape().elements() / 4096, [&](auto idx) {
        const int batch_size      = params.batch_size;
        const int sequence_length = params.sequence_length;
        int seqlen_present_kv_cache = params.seqlen_present_kv_cache;
        CalculateSoftmax(output.begin(),
                                seqlens_k.begin(),
                                batch_size,
                                sequence_length,
                                seqlen_present_kv_cache,
                                params,
                                idx);
        
    });
}

} // namespace migraphx
#endif
