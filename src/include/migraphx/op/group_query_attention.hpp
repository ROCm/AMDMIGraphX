#ifndef MIGRAPHX_GUARD_OPERATORS_GROUP_QUERY_ATTENTION_HPP
#define MIGRAPHX_GUARD_OPERATORS_GROUP_QUERY_ATTENTION_HPP

#include <migraphx/check_shapes.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/par_for.hpp>
#include <migraphx/gemm.hpp>
#include <migraphx/argument.hpp>
#include <fstream>
#include <iostream>
#include <iomanip>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct RotaryParameters
{
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
    bool transposed; // Whether the input tensor has been transposed into (batch, num_heads,
                     // seq_len, hidden)
    int seqlen_present_kv_cache;

    void print() const {
        // printf("scale: %f\n", scale);
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
        // printf("do_rotary: %d\n", do_rotary);
        // printf("kv_num_heads: %d\n", kv_num_heads);
        // printf("local_window_size: %d\n", local_window_size);
        // printf("rotary_interleaved: %d\n", rotary_interleaved);
    }
};

struct group_query_attention
{
    int do_rotary          = 0;
    int kv_num_heads       = 0;
    int local_window_size  = -1;
    std::size_t num_heads  = 1;
    int rotary_interleaved = 0;
    float scale            = 1.0;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.do_rotary, "do_rotary"),
                    f(self.kv_num_heads, "kv_num_heads"),
                    f(self.local_window_size, "local_window_size"),
                    f(self.num_heads, "num_heads"),
                    f(self.rotary_interleaved, "rotary_interleaved"),
                    f(self.scale, "scale"));
    }

    std::string name() const { return "group_query_attention"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        if (inputs.size() == 8)
        {
            auto query_lens = inputs.front().lens();
            std::size_t q_hidden_size = (query_lens[1] * query_lens[3] * num_heads) / (num_heads + 2 * kv_num_heads);
            std::vector<std::size_t> output_lens{query_lens.at(0), query_lens.at(2), q_hidden_size};
            shape output_shape{inputs.front().type(), output_lens};
            return shape({output_shape, inputs[1], inputs[2]});
        }
        auto query_lens           = inputs.front().lens();
        std::size_t q_hidden_size = (query_lens[2] * num_heads) / (num_heads + 2 * kv_num_heads);
        std::vector<std::size_t> output_lens{query_lens.at(0), query_lens.at(1), q_hidden_size};
        shape output_shape{inputs.front().type(), output_lens};
        return shape({output_shape, inputs[3], inputs[4]});
    }

    template <class T>
    void run_rotary_embedding(T input,
                              T cos_cache,
                              T sin_cache,
                              T output,
                              bool interleaved,
                              const int64_t* pos_ids,
                              RotaryParameters parameters) const
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
        par_for(loop_len, [&](const auto idx) {
            const int b            = static_cast<int>((idx / n_heads) / sequence_length);
            const int s            = static_cast<int>((idx / n_heads) % sequence_length);
            const int n            = static_cast<int>(idx % n_heads);
            const int block_offset = b * batch_stride + s * seq_stride + n * head_stride;
            auto input_data        = input + block_offset;
            auto output_data       = output + block_offset;

            // Cache is (M, H/2) or (M, rotary_embedding_dim/2)
            const int position_id = (position_ids_format == 0)
                                        ? static_cast<int>(pos_ids[0]) + s
                                        : static_cast<int>(pos_ids[b * sequence_length + s]);
            // if(print)
            //     printf("ref_pos_id%lu: %d | %d + %d\n", idx, position_id, static_cast<int>(pos_ids[0]), s);
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
        });
    }

    template <class T>
    void pack_v_into_rotary_QKV(RotaryParameters parameters, const T input, T output) const
    {
        const int loop_len = parameters.batch_size * parameters.sequence_length * kv_num_heads;
        par_for(loop_len, [&](const auto idx) {
            const int b = static_cast<int>((idx / kv_num_heads) / parameters.sequence_length);
            const int s = static_cast<int>((idx / kv_num_heads) % parameters.sequence_length);
            const int n = static_cast<int>(idx % kv_num_heads);
            const int block_offset = b * parameters.batch_stride + s * parameters.seq_stride +
                                     n * parameters.head_stride;
            const T input_data = input + block_offset;
            T output_data      = output + block_offset;
            for(int i = 0; i < parameters.head_size; i++)
            {
                output_data[i] = input_data[i];
            }
        });
    }

    template <class T>
    void copy_data(T destination, const T source, std::size_t n, bool print=false) const
    {
        par_for(n, [&](auto i) { 
            if(print)
                // printf("ref_query%zu: %f\n", i, static_cast<double>(source[i]));
                printf("ref_query%zu\n", i);

            destination[i] = source[i]; });
    }

    template <typename T>
    T ConcatStateChunkGQA(const T past,
                          const T chunk,
                          T present,
                          size_t present_buff_chunk_length,
                          size_t past_buff_chunk_length,
                          size_t past_chunk_length,
                          size_t new_chunk_length,
                          bool is_prompt,
                          bool past_present_share_buffer,
                          std::ptrdiff_t i,
                          bool print = false) const
    {
        T start = present + i * present_buff_chunk_length;

        T p = start;
        if(!is_prompt)
        {
            if(!past_present_share_buffer)
            {
                const T src_past = past + i * past_buff_chunk_length;
                copy_data(p, src_past, past_chunk_length);
            }
            p += past_chunk_length;
        }
        copy_data(p, chunk, new_chunk_length, print);
        return start;
    }

    template <class T>
    void CalculateAttentionSoftmaxInplace(T score, int N, int D) const
    {
        par_for(N, [&](const auto j) {
            auto x = score + j * D;
            auto y = x;

            // e^x is represented as infinity if x is large enough, like 100.f.
            // Infinity divided by Infinity is a NAN. Thus, softmax gets a NAN if
            // one or more item are large enough. a math transform as below is
            // leveraged to get a stable softmax: e^xi/(e^x1 + ...e^xn) = e^(xi -
            // max) / (e^(x1 - max) + ... + e^(xn - max))
            float max = -std::numeric_limits<float>::infinity();
            for(int i = 0; i < D; i++)
            {
                if(max < x[i])
                    max = x[i];
            }
            for(int i = 0; i < D; i++)
            {
                y[i] = expf(x[i] - max);
            }

            double sum = 0.0;

            for(int i = 0; i < D; i++)
            {
                sum += x[i];
            }

            if(float_equal(sum, 0.0))
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
        });
    }

    // Helper function to compute the attention probs. It does 2 things:
    //  attention_probs(B, N, S, T) = 1/sqrt(H) x Q(B, N, S, H) x K'(B, N, T, H -> B, N, H, T)
    //  attention_probs(B, N, S, T) = Softmax(attention_probs)
    template <class T, class U>
    void CalculateAttentionProbs(
        T attention_probs,                  // output buffer with size BxNxSxT
        T Q,                                // Q data. Its size is BxNxSxH
        T K,                                // k data. Its size is BxNxLxH
        U seqlens_k,                        // past sequence lengths tensor
        int batch_size,                     // batch size of self-attention
        int sequence_length,                // sequence length of self-attention (S)
        int past_buffer_sequence_length,    // sequence length of past state
        int present_buffer_sequence_length, // sequence length of present state
        int head_size,                      // head size of self-attention
        T past_key,                         // past key only
        T present_key,                      // present key only
        bool past_present_share_buffer,     // whether present key and value share the same buffer
        bool packed_qkv,                    // whether Q, K, V are packed
        shape::type_t dtype) const
    {
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
        const float alpha  = scale == 0.0f ? 1.0f / sqrt(static_cast<float>(head_size)) : scale;

        par_for(loop_len /* - loop_len + 1 */, [&](const auto i) {
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
            // if(i == 0)
            // {
            //     for(int j = 0; j < kv_input_chunk_length; ++j)
            //     {
            //         printf("ref_query%d: %f\n", j, static_cast<double>(k[j]));
            //     }
            // }
            // if (i == 0)
            // {
            //     auto k_offset = packed_batch_stride * batch_index +
            //         kv_input_chunk_length * (head_index / kv_num_heads_factor);;
            //     printf("ref_query%lu: %zu\n", i, k_offset);
            // }
            
            k = ConcatStateChunkGQA(past_key,
                                    k,
                                    present_key,
                                    present_buff_chunk_length,
                                    past_buff_chunk_length,
                                    past_chunk_length,
                                    kv_input_chunk_length,
                                    is_prompt,
                                    past_present_share_buffer,
                                    i / kv_num_heads_factor);
            // if(i == 0)
            // {
            //     // printf("ref_vals%zu, %zu, %zu, %zu\n", present_buff_chunk_length,
            //     //                 past_buff_chunk_length,
            //     //                 past_chunk_length,
            //     //                 kv_input_chunk_length);
            //     for(int j = 0; j < kv_input_chunk_length; ++j)
            //     {
            //         printf("ref_query%d: %f\n", j, static_cast<double>(k[j]));
            //     }
            // }
            // Calculate Q*K' + AttentionMask
            //                     original                 transposed             each iteration
            // A: Q                (B x N x) S x H          (B x N x) S x H        S x H
            // B: K'               (B x N x) T x H          (B x N x) H x T        H x T
            // C: attention_probs  (B x N x) S x T          (B x N x) S x T        S x T
            T q;
            if(packed_qkv)
            {
                q = Q + packed_batch_stride * batch_index + q_input_chunk_length * head_index;
            }
            else
            {
                q = Q + q_input_chunk_length * i;
            }

            gemm(sequence_length, //m
                 total_seqlen, //n 
                 head_size, //k
                 head_size,
                 head_size,
                 present_buffer_sequence_length,
                 output,
                 q,
                 k,
                 alpha,
                 0.0f,
                 dtype,
                 true);
            // if(i == 0)
            // {
            //     for(int j = 0; j < sequence_length * total_seqlen; ++j)
            //     {
            //         printf("ref_query%d: %f\n", j, static_cast<double>(output[j]));
            //     }
            // }
            T output_softmax = output;
            for(int seq = 0; seq < sequence_length; seq++)
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
        });
    }

    template <class T, class U, class W>
    void CalculateVxAttentionScore(
        T output,                           // buffer for the result with size BxSxNxH
        const W attention_probs,            // Attention probs with size BxNxSxT
        const T V,                          // V value with size BxN_kvxSxH
        const U seqlens_k,                  // past sequence lengths tensor
        int batch_size,                     // batch size
        int sequence_length,                // sequence length
        int past_buffer_sequence_length,    // sequence length in past state
        int present_buffer_sequence_length, // sequence length in past state
        int head_size,                      // head size of Q, K, V
        int hidden_size,                    // hidden size of Output
        const T past_value,                 // past value only
        T present_value,                    // present value only
        bool past_present_share_buffer,     // whether present key and value share the same buffer
        bool packed_qkv,
        shape::type_t dtype) const // whether Q, K, V are packed
    {
        const bool is_prompt = sequence_length != 1;
        const int packed_batch_stride =
            packed_qkv ? (num_heads + 2 * kv_num_heads) * sequence_length * head_size : 0;
        const int kv_num_heads_factor   = num_heads / kv_num_heads;
        const int kv_input_chunk_length = sequence_length * head_size; // L x H
        const size_t past_buff_chunk_length =
            static_cast<size_t>(past_buffer_sequence_length) * head_size; // L x H
        const size_t present_buff_chunk_length =
            static_cast<size_t>(present_buffer_sequence_length) * head_size; // T x H

        auto loop_len = batch_size * num_heads;
        par_for(loop_len /* - loop_len + 1 */, [&](const auto i) {
            const int batch_index = static_cast<int>(i / num_heads);
            const int head_index  = static_cast<int>(i % num_heads);
            const int past_seqlen = sequence_length == 1 ? static_cast<int>(seqlens_k[batch_index])
                                                         : past_buffer_sequence_length;
            const size_t past_chunk_length = static_cast<size_t>(past_seqlen) * head_size;
            const int total_seqlen         = seqlens_k[batch_index] + 1;

            T v;
            if(packed_qkv)
            {
                v = V + packed_batch_stride * batch_index +
                    kv_input_chunk_length * (head_index / kv_num_heads_factor);
            }
            else
            {
                v = V + kv_input_chunk_length * (i / kv_num_heads_factor);
            }

            v = ConcatStateChunkGQA(past_value,
                                    v,
                                    present_value,
                                    present_buff_chunk_length,
                                    past_buff_chunk_length,
                                    past_chunk_length,
                                    kv_input_chunk_length,
                                    is_prompt,
                                    past_present_share_buffer,
                                    i / kv_num_heads_factor);
            // if(i == 0)
            // {
            //     for(int j = 0; j < kv_input_chunk_length; ++j)
            //     {
            //         printf("ref_query%d: %f\n", j, static_cast<double>(v[j]));
            //     }
            // }

            T output_current =
                output + (batch_index * sequence_length * num_heads + head_index) * head_size;
            ptrdiff_t attention_probs_offset = sequence_length * present_buffer_sequence_length * i;
            // auto out_off = (batch_index * sequence_length * num_heads + head_index) * head_size;
            // printf("ref%d: %d\n", i, hidden_size);

            gemm(sequence_length,
                 head_size,
                 total_seqlen,
                 present_buffer_sequence_length,
                 head_size,
                 hidden_size,
                 output_current,
                 attention_probs + attention_probs_offset,
                 v,
                 1.0f,
                 0.0f,
                 dtype);
        });
    }

    template <class T, class U>
    void apply_attention(T Q,
                         T past_key,
                         T past_value,
                         T output,
                         T present_key,
                         T present_value,
                         U seqlens_k,
                         T attention_probs,
                         RotaryParameters parameters,
                         shape::type_t dtype) const
    {
        const int batch_size      = parameters.batch_size;
        const int sequence_length = parameters.sequence_length;
        const int head_size       = parameters.head_size;
        const int hidden_size     = parameters.hidden_size;
        const bool packed_qkv     = true;

        int seqlen_present_kv_cache = parameters.seqlen_present_kv_cache;
        int seqlen_past_kv_cache    = 4096;

        // Calculate the attention score.
        bool past_present_share_buffer = false;
        const T k                      = Q + num_heads * sequence_length * head_size;
        
        CalculateAttentionProbs(attention_probs,
                                Q,
                                k,
                                seqlens_k,
                                batch_size,
                                sequence_length,
                                seqlen_past_kv_cache,
                                seqlen_present_kv_cache,
                                head_size,
                                past_key,
                                present_key,
                                past_present_share_buffer,
                                packed_qkv,
                                dtype);
        // for(auto i = 0; i < 4096; ++i)
        // {
        //     output[i] = attention_probs[i];
        // }
        
        // for(int j = 0; j < 262144; ++j)
        // {
        //     printf("ref_query%d: %f\n", j, static_cast<double>(attention_probs[j]));
        // }
        const T v = Q + (num_heads + kv_num_heads) * sequence_length * head_size;
        // for(int j = 0; j < parameters.num_heads * sequence_length * head_size; ++j)
        //     printf("ref_query%d: %f\n", j, static_cast<double>(v[j]));
        CalculateVxAttentionScore(output,
                                  attention_probs,
                                  v,
                                  seqlens_k,
                                  batch_size,
                                  sequence_length,
                                  seqlen_past_kv_cache,
                                  seqlen_present_kv_cache,
                                  head_size,
                                  hidden_size,
                                  past_value,
                                  present_value,
                                  past_present_share_buffer,
                                  packed_qkv,
                                  dtype);
    }

    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        auto q_shape                      = args[0].get_shape();
        auto q_lens                       = q_shape.lens();
        const std::size_t batch_size      = q_lens[0];
        const std::size_t sequence_length = q_lens[1];
        auto past_key_shape               = args[3].get_shape();
        auto past_key_lens                = past_key_shape.lens();
        auto past_sequence_length         = past_key_lens[2];
        std::size_t q_hidden_size         = q_lens[2];
        std::size_t head_size             = q_hidden_size / (num_heads + 2 * kv_num_heads);
        q_hidden_size                     = head_size * num_heads;
        const bool packed_qkv             = true;
        std::size_t rotary_dim            = args[7].get_shape().lens()[1] * 2;
        std::size_t present_kv_seqlen;

        args[6].visit([&](auto total_sequence_length) {
            std::size_t max_total_sequence_length_val = 0;
            for(auto i = 0; i < batch_size; ++i)
            {
                if(total_sequence_length[i] > max_total_sequence_length_val)
                {
                    max_total_sequence_length_val = total_sequence_length[i];
                }
            }
            present_kv_seqlen = std::max(static_cast<std::size_t>(max_total_sequence_length_val),
                                         past_sequence_length);
        });

        auto output_shape_0 = output_shape.sub_shapes().front();
        argument result{output_shape_0};
        argument qkv_rotary{shape{output_shape_0.type(),
                                  {batch_size,
                                   static_cast<std::size_t>(num_heads + 2 * kv_num_heads),
                                   sequence_length,
                                   head_size}}};

        shape kv_shape{
            output_shape_0.type(),
            {batch_size, static_cast<std::size_t>(kv_num_heads), past_sequence_length, head_size}};
        argument present_k_out{kv_shape};
        argument present_v_out{kv_shape};
        argument attention_probs{shape{
            output_shape_0.type(), {batch_size, num_heads, sequence_length, present_kv_seqlen}}};

        args[0] = args[0].reshape(shape{output_shape_0.type(),
                                        {batch_size,
                                         sequence_length,
                                         static_cast<std::size_t>(num_heads + 2 * kv_num_heads),
                                         head_size}});
        argument qkv{qkv_rotary.get_shape()};
        visit_all(qkv, args[0])([&](auto a, auto b) {
            auto in_shape  = args[0].get_shape();
            auto out_shape = qkv.get_shape();
            shape_for_each(in_shape, [&](const auto& idx) {
                std::vector<std::size_t> out_idx{idx[0], idx[2], idx[1], idx[3]};
                a(out_idx.begin(), out_idx.end()) = b(idx.begin(), idx.end());
            });
        });

        visit_all(result,
                qkv,
                  args[3],
                  args[4],
                  args[7],
                  args[8],
                  qkv_rotary,
                  present_k_out,
                  present_v_out,
                  attention_probs)([&](auto output,
                                       auto query,
                                       auto past_key,
                                       auto past_value,
                                       auto cos_cache,
                                       auto sin_cache,
                                       auto RotaryQKV,
                                       auto present_k,
                                       auto present_v,
                                       auto attn_probs) {
            visit_all(args[5])([&](auto seqlens_k) {
                if(do_rotary)
                {
                    par_for(kv_shape.elements(), [&](auto i){
                        present_k[i] = past_key[i];
                        present_v[i] = past_value[i];
                    }); 
                    auto seq_stride  = head_size;
                    auto head_stride = sequence_length * seq_stride;
                    auto batch_stride =
                        (packed_qkv ? (num_heads + 2 * kv_num_heads) : num_heads) * head_stride;
                    auto position_ids_format = sequence_length == 1 ? 1 : 0;
                    bool transposed          = true;
                    std::vector<int64_t> pos_ids(sequence_length == 1 ? batch_size : 1);
                    if(sequence_length == 1)
                    {
                        for(int b = 0; b < batch_size; b++)
                        {
                            pos_ids[b] = static_cast<int64_t>(seqlens_k[b]);
                        }
                    }
                    else
                    {
                        pos_ids[0] = static_cast<int64_t>(0);
                    }
                    auto q_input  = query.begin();
                    auto k_input  = q_input + num_heads * sequence_length * head_size;
                    auto q_rotary = RotaryQKV.begin();
                    auto k_rotary = q_rotary + num_heads * sequence_length * head_size;

                    RotaryParameters rotary_params     = {};
                    rotary_params.batch_size           = batch_size;
                    rotary_params.sequence_length      = sequence_length;
                    rotary_params.hidden_size          = q_hidden_size;
                    rotary_params.head_size            = head_size;
                    rotary_params.rotary_embedding_dim = rotary_dim;
                    rotary_params.num_heads            = num_heads;
                    rotary_params.max_sequence_length  = sequence_length;
                    rotary_params.seq_stride           = head_size;
                    rotary_params.head_stride          = sequence_length * rotary_params.seq_stride;
                    rotary_params.batch_stride         = batch_stride;
                    rotary_params.position_ids_format  = position_ids_format;
                    rotary_params.transposed           = transposed;
                    rotary_params.seqlen_present_kv_cache = present_kv_seqlen;
                    // for(int i = 0; i < query.get_shape().elements(); ++i)
                    // {
                    //     printf("ref_query%d: %f\n", i, static_cast<double>(query[i]));
                    // }
                    for(int i = 0; i < query.get_shape().elements(); ++i)
                    {
                        RotaryQKV[i] = 0.0;
                    }
                    // for(int i = 0; i < query.get_shape().elements(); ++i)
                    // {
                    //     printf("ref_query%d: %f\n", i, static_cast<double>(query[i]));
                    // }
                    run_rotary_embedding(q_input,
                                         cos_cache.begin(),
                                         sin_cache.begin(),
                                         q_rotary,
                                         rotary_interleaved,
                                         pos_ids.data(),
                                         rotary_params);
                    // for(int i = 0; i < query.get_shape().elements(); ++i)
                    // {
                    //     printf("ref_query%d: %f\n", i, static_cast<double>(RotaryQKV[i]));
                    // }
                    std::size_t kv_hidden_size = head_size * kv_num_heads;
                    rotary_params.num_heads    = kv_num_heads;
                    rotary_params.hidden_size  = kv_hidden_size;

                    run_rotary_embedding(k_input,
                                         cos_cache.begin(),
                                         sin_cache.begin(),
                                         k_rotary,
                                         rotary_interleaved,
                                         pos_ids.data(),
                                         rotary_params);
                    // for(int i = 0; i < query.get_shape().elements(); ++i)
                    // {
                    //     printf("ref_query%d: %f\n", i, static_cast<double>(RotaryQKV[i]));
                    // }
                    auto v_input            = k_input + kv_num_heads * sequence_length * head_size;
                    auto v_rotary           = k_rotary + kv_num_heads * sequence_length * head_size;
                    rotary_params.num_heads = num_heads;

                    pack_v_into_rotary_QKV(rotary_params, v_input, v_rotary);
                    // for(int i = 0; i < query.get_shape().elements(); ++i)
                    // {
                    //     printf("ref_query%d: %f\n", i, static_cast<double>(RotaryQKV[i]));
                    // }
                    auto Q = RotaryQKV;
                    // rotary_params.print();
                    // for(int i = 0; i < query.get_shape().elements(); ++i)
                    // {
                    //     printf("ref_query%d: %f\n", i, static_cast<double>(Q[i]));
                    // }
                    // for(int i = 0; i < past_key.get_shape().elements(); ++i)
                    // {
                    //     printf("ref_query%d: %f\n", i, static_cast<double>(past_value[i]));
                    // }
                    apply_attention(Q.begin(),
                                    past_key.begin(),
                                    past_value.begin(),
                                    output.begin(),
                                    present_k.begin(),
                                    present_v.begin(),
                                    seqlens_k.begin(),
                                    attn_probs.begin(),
                                    rotary_params,
                                    output_shape_0.type());
                    // for(int i = 0; i < (kv_shape.elements() / 32); ++i)
                    // {
                    //     printf("ref_query%d: %f\n", i, static_cast<double>(present_k[i]));
                    // }
                    // for(int j = 0; j < output_shape_0.elements(); ++j)
                    // {
                    //     printf("ref_query%d: %f\n", j, static_cast<double>(output[j]));
                    // }
                    // for(int j = 0; j < kv_shape.elements() - 2/* 524288 */; ++j)
                    // {
                    //     if (float_equal(static_cast<double>(present_k[j]), 0.750000))
                    //     {
                    //         if(float_equal(static_cast<double>(present_k[j + 1]), 0.125000))
                    //         {
                    //             if(float_equal(static_cast<double>(present_k[j + 2]), -0.187500))
                    //             {
                    //                 // if(float_equal(static_cast<double>(present_k[j + 3]), 0.562500))
                    //                 // {
                    //                     printf("pattern found at %d\n", j);
                    //                 // }
                    //             }
                    //         }
                    //     }
                    //     // float zero = 0.0;
                    //     // if(not float_equal(static_cast<float>(present_k[j]), zero))
                    //         // printf("ref_query%d: %f\n", j, static_cast<double>(present_k[j]));
                    // }
                    // present_k.begin() = past_key.begin();
                    // present_v.begin() = past_value.begin();
                    // for(auto i = 0; i < present_k_out.get_shape().elements(); ++i)
                    // {
                    //     present_k.data()[i] = past_key.data()[i];
                    //     present_v.data()[i] = past_value.data()[i];
                    // }
                    // for(auto i = 0; i < result.get_shape().elements(); ++i)
                    // {
                    //     output.data()[i] = Q.data()[i];
                    // }
                }
            });
        });

        return {{result, present_k_out, present_v_out}};
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
