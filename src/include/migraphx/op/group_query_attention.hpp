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

    void print(std::string title, bool init=false) 
    {
        std::ofstream out;
        auto mode = init ? std::ios::trunc : std::ios::app;
        out.open("gqa_log.txt", mode);
        out << title << std::endl;
        out << "batch_size: " << batch_size << std::endl;
        out << "sequence_length: " << sequence_length << std::endl;
        out << "hidden_size: " << hidden_size << std::endl;
        out << "head_size: " << head_size << std::endl;
        out << "rotary_embedding_dim: " << rotary_embedding_dim << std::endl;
        out << "num_heads: " << num_heads << std::endl;
        out << "max_sequence_length: " << max_sequence_length << std::endl;
        out << "head_stride: " << head_stride << std::endl;
        out << "seq_stride: " << seq_stride << std::endl;
        out << "batch_stride: " << batch_stride << std::endl;
        out << "position_ids_format: " << position_ids_format << std::endl;
        out << "transposed: " << (transposed ? "true" : "false") << std::endl;
        out << "seqlen_present_kv_cache: " << seqlen_present_kv_cache << std::endl << std::endl;
        out.close();
    }
    
};

struct group_query_attention
{
    int do_rotary          = 0;
    int kv_num_heads       = 0;
    int local_window_size  = -1;
    std::size_t num_heads          = 1;
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
        //check_shapes{inputs, *this};
        auto query_lens = inputs.front().lens();
        std::size_t q_hidden_size = (query_lens[2] * num_heads) / (num_heads + 2 * kv_num_heads);
        std::vector<std::size_t> output_lens{query_lens.at(0), query_lens.at(1), q_hidden_size};
        shape output_shape{inputs.front().type(), output_lens};
        int kv_offset = inputs.size() == 7 ? 0 : 2;
        return shape({output_shape, inputs[1 + kv_offset], inputs[2 + kv_offset]});
    }

    template<class T>
    void print_tensor(std::string title, const T t, const int elem, const bool init=false) const
    {
        std::cout << "writing " << elem << " elements of " << title << " to log" << std::endl;
        std::ofstream out;
        auto mode = init ? std::ios::trunc : std::ios::app;
        out.open("gqa_tensor_log.txt", mode);
        out.precision(3);
        out << title << std::endl;
        for (auto i = 0; i < elem; ++i)
        {
            out << std::setw(6) << static_cast<float>(t[i]) << std::endl;
        }
        out << std::endl;
        out.close();
    }
    template<class T>
    void print_to_py(const T a, const T b, const int elem_a, const int elem_b) const
    {
        std::ofstream out;
        auto mode = std::ios::trunc;
        out.open("gemm_params_log.txt", mode);
        out.precision(4);
        out << "a_data = [";
        for (auto i = 0; i < elem_a; ++i)
        {
            out << static_cast<float>(a[i]) << ",";
        }
        out << "]" << std::endl;
        out << "b_data = [";
        for (auto i = 0; i < elem_b; ++i)
        {
            out << static_cast<float>(b[i]) << ",";
        }
        out << "]" << std::endl;
        out.close();
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
    void copy_data(T destination, const T source, std::size_t n) const
    {
        par_for(n, [&](auto i) { destination[i] = source[i]; });
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
                          bool do_print=false) const
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
        if (do_print)
        {
            std::cout << "copy data1" << std::endl;
            print_tensor("p", p, new_chunk_length, true);
            std::cout << "printed p" << std::endl;
            print_tensor("chunk", chunk, new_chunk_length);
            std::cout << "printed chunk" << std::endl;
        }
        
        copy_data(p, chunk, new_chunk_length);
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
        T Q,                          // Q data. Its size is BxNxSxH
        T K,                          // k data. Its size is BxNxLxH
        U seqlens_k,                  // past sequence lengths tensor
        int batch_size,                     // batch size of self-attention
        int sequence_length,                // sequence length of self-attention (S)
        int past_buffer_sequence_length,    // sequence length of past state
        int present_buffer_sequence_length, // sequence length of present state
        int head_size,                      // head size of self-attention
        T past_key,                   // past key only
        T present_key,                      // present key only
        bool past_present_share_buffer,     // whether present key and value share the same buffer
        bool packed_qkv,                     // whether Q, K, V are packed
        shape::type_t dtype
    ) const
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

        // if (!past_present_share_buffer) {
        // memset(present_key, 0, batch_size * kv_num_heads_ * present_buffer_sequence_length *
        // head_size * sizeof(T));
        // }

        const int loop_len = batch_size * num_heads;
        const float alpha  = scale == 0.0f ? 1.0f / sqrt(static_cast<float>(head_size)) : scale;
        // // std::cout << "scale: " << scale << ", head_size: " << head_size << std::endl;

        // TensorOpCost unit_cost;
        // const size_t probs_matrix_bytes = SafeInt<size_t>(sequence_length) *
        // present_buffer_sequence_length * sizeof(T); unit_cost.compute_cycles =
        // static_cast<double>(2 * sequence_length * head_size * present_buffer_sequence_length);
        // unit_cost.bytes_loaded = static_cast<double>((sequence_length +
        // present_buffer_sequence_length) * head_size * sizeof(T)); unit_cost.bytes_stored =
        // static_cast<double>(probs_matrix_bytes);

        // unit_cost.bytes_loaded += static_cast<double>(probs_matrix_bytes);
        // unit_cost.bytes_stored += static_cast<double>(probs_matrix_bytes);

        // if (present_key) {
        // double bytes_to_copy_key = static_cast<double>(sizeof(T) * present_buff_chunk_length);
        // unit_cost.bytes_loaded += bytes_to_copy_key;
        // unit_cost.bytes_stored += bytes_to_copy_key;
        // }
        // char* mt = nullptr;
        // print_tensor("CAP", mt, 0, true);
        // print_tensor("Q", Q, batch_size * num_heads * sequence_length * head_size);
        // print_tensor("attention_probs", attention_probs, batch_size * num_heads * 4096 * sequence_length);
        // std::vector<int> idxs(loop_len);
        // std::iota(idxs.begin(), idxs.end(), 0);
        // std::for_each(idxs.begin(), idxs.end(), [&](const auto& i) {
        par_for(loop_len, [&](const auto i) {
            const int batch_index = static_cast<int>(i) / num_heads;
            const int head_index  = static_cast<int>(i) % num_heads;
            const int past_seqlen = sequence_length == 1 ? static_cast<int>(seqlens_k[batch_index])
                                                         : past_buffer_sequence_length;
            const size_t past_chunk_length = static_cast<size_t>(past_seqlen) * head_size;
            const int total_seqlen         = seqlens_k[batch_index] + 1;

            const int output_offset =
                static_cast<int>(i) * sequence_length * present_buffer_sequence_length;
            auto output = attention_probs + output_offset;

            // const T* k;
            // if (packed_qkv) {
            auto k = K + packed_batch_stride * batch_index +
                     kv_input_chunk_length * (head_index / kv_num_heads_factor);
            // } else {
            // k = K + kv_input_chunk_length * (i / kv_num_heads_factor);
            // }
            // print_tensor("k", k, total_seqlen * head_size);
            // if (i == 0)
            // {
                // std::cout << present_buff_chunk_length << std::endl
                //           <<  past_buff_chunk_length << std::endl
                //           <<  past_chunk_length << std::endl
                //           <<  kv_input_chunk_length << std::endl
                //           <<  is_prompt << std::endl
                //           <<  past_present_share_buffer << std::endl
                //           <<  i / kv_num_heads_factor << std::endl;
            // }
            // std::cout << "Concat " << i << " of " << loop_len << std::endl;
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
            // math::GemmEx<T, ThreadPool>(CblasNoTrans, CblasTrans,
            //                             sequence_length, total_seqlen, head_size, alpha,
            //                             q, head_size, k, head_size,
            //                             0.0f /*bata*/,
            //                             output, present_buffer_sequence_length, nullptr);
            // // std::cout << "m, n, k: " << sequence_length << ", " << total_seqlen << ", " << head_size << std::endl;
            // print_tensor("q", q, sequence_length * head_size);
            // print_tensor("k", k, total_seqlen * head_size);
            // print_tensor("out", output, sequence_length * total_seqlen);
            // std::cout << "alpha: " << alpha << std::endl;
            // print_to_py(q, k, 64 * 128, 128);
            gemm(sequence_length, total_seqlen, head_size, head_size, head_size, present_buffer_sequence_length, output, q, k, alpha, 0.0f, dtype, true);
            // std::vector<float> a(64 * 128, 1.0);
            // std::vector<float> b(128, 1.0);
            // std::vector<float> c(64, 1.0);
            
            // gemm(sequence_length, total_seqlen, head_size, c, a, b, alpha, 0.0f, dtype, false, true);
            // print_tensor("output", c.data(), 64);
            // print_tensor("output_" + std::to_string(i), output, (batch_size * sequence_length * num_heads * 4096) - output_offset);
            // print_tensor("output_" + std::to_string(i), output, sequence_length * total_seqlen);
            // compute Softmax
            // std::cout << "Softmax" << std::endl;
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
        const U seqlens_k,           // past sequence lengths tensor
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
        shape::type_t dtype) const              // whether Q, K, V are packed
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

        // if(!past_present_share_buffer)
        // {
        //     memset(present_value,
        //            0,
        //            batch_size * kv_num_heads * present_buffer_sequence_length * head_size *
        //                sizeof(T));
        // }

        // The cost of Gemm
        // TensorOpCost unit_cost;
        // unit_cost.compute_cycles =
        //     static_cast<double>(2 * sequence_length * head_size *
        //     present_buffer_sequence_length);
        // unit_cost.bytes_loaded = static_cast<double>((sequence_length + head_size) *
        //                                              present_buffer_sequence_length * sizeof(T));
        // unit_cost.bytes_stored = static_cast<double>(sequence_length * head_size * sizeof(T));

        // if(present_value)
        // {
        //     double bytes_to_copy_value = static_cast<double>(present_buff_chunk_length *
        //     sizeof(T)); unit_cost.bytes_loaded += bytes_to_copy_value; unit_cost.bytes_stored +=
        //     bytes_to_copy_value;
        // }
        auto v_len = (batch_size * (num_heads + 2 * kv_num_heads) * sequence_length * head_size) - (2 * num_heads * sequence_length * head_size);
        char* mt = nullptr;
        //print_tensor("VXscore", mt, 0, true);
        // print_tensor("present_value", present_value, (batch_size) * num_heads * 4096 * 128);
        // const size_t bytes_to_copy_trans = SafeInt<size_t>(head_size) * sizeof(T);
        // double bytes_to_copy_trans_all = static_cast<double>(sequence_length *
        // bytes_to_copy_trans); unit_cost.bytes_loaded += bytes_to_copy_trans_all;
        // unit_cost.bytes_stored += bytes_to_copy_trans_all;
        auto loop_len = batch_size * num_heads;
        // std::vector<int> idxs(loop_len);
        // std::iota(idxs.begin(), idxs.end(), 0);
        // std::for_each(idxs.begin(), idxs.end(), [&](const auto& i) {
        par_for(loop_len, [&](const auto i) {
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
            // print_tensor("v_" + std::to_string(i), v, v_len - (packed_batch_stride * batch_index + kv_input_chunk_length * (head_index / kv_num_heads_factor)));
            //print_tensor("past_value_" + std::to_string(i), past_value, 4096 * 128);
            // std::cout << "pc" << i << std::endl;
            bool do_print = false;
            // if (i == 0)
            // {
            //     std::cout << present_buff_chunk_length << std::endl
            //               <<  past_buff_chunk_length << std::endl
            //               <<  past_chunk_length << std::endl
            //               <<  kv_input_chunk_length << std::endl
            //               <<  is_prompt << std::endl
            //               <<  past_present_share_buffer << std::endl
            //               <<  i / kv_num_heads_factor << std::endl;
            //     do_print = true;
            // }
            v = ConcatStateChunkGQA(past_value,
                                    v,
                                    present_value,
                                    present_buff_chunk_length,
                                    past_buff_chunk_length,
                                    past_chunk_length,
                                    kv_input_chunk_length,
                                    is_prompt,
                                    past_present_share_buffer,
                                    i / kv_num_heads_factor,
                                    do_print);
            // std::cout << "pc" << i << std::endl;
            //print_tensor("v_" + std::to_string(i), v, v_len);

            T output_current =
                output + (batch_index * sequence_length * num_heads + head_index) * head_size;
            ptrdiff_t attention_probs_offset =
                sequence_length * present_buffer_sequence_length * i;

            // math::GemmEx<T, ThreadPool>(CblasNoTrans,
            //                             CblasNoTrans,
            //                             sequence_length,
            //                             head_size,
            //                             total_seqlen,
            //                             1.f, /*alpha*/
            //                             attention_probs + attention_probs_offset,
            //                             present_buffer_sequence_length,
            //                             v,
            //                             head_size,
            //                             0.0f /*beta*/,
            //                             output_current,
            //                             hidden_size,
            //                             nullptr);
            // print_tensor("attention_probs_" + std::to_string(i), attention_probs + attention_probs_offset, (batch_size * num_heads * sequence_length * 4096) - attention_probs_offset);
            // print_tensor("attention_probs_" + std::to_string(i), attention_probs + attention_probs_offset, sequence_length * total_seqlen);
            // print_tensor("v_" + std::to_string(i), v, total_seqlen * head_size);
            gemm(sequence_length, head_size, total_seqlen, present_buffer_sequence_length, head_size, hidden_size, output_current, attention_probs + attention_probs_offset, v, 1.0f, 0.0f, dtype);
            // std::cout << "pg" << i << std::endl;
            //print_tensor("output_" + std::to_string(i), output_current, sequence_length * head_size);
        });
    }

    template <class T, class U>
    void apply_attention(T Q,
                        //  T K,
                        //  T V,
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

        int seqlen_past_kv_cache = 4096; // only use with max_seq_len
        // if (past_key != nullptr && past_value != nullptr) {
        //     seqlen_past_kv_cache = static_cast<int>(past_key->Shape().GetDims()[2]);
        // }
        int seqlen_present_kv_cache = parameters.seqlen_present_kv_cache;

        // Calculate the attention score.

        bool past_present_share_buffer = false;

        const T k = Q + num_heads * sequence_length * head_size;
        
        // std::cout << "Apply_attn 1" << std::endl;
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
        // print_tensor("attention_probs:", attention_probs, parameters.batch_size * num_heads * parameters.sequence_length * 4096, true);
        // std::cout << "Apply_attn 2" << std::endl;
        const T v = Q + (num_heads + kv_num_heads) * sequence_length * head_size;   
        
        CalculateVxAttentionScore(output, attention_probs,
                            v, seqlens_k, batch_size, sequence_length, seqlen_past_kv_cache,
                            seqlen_present_kv_cache, head_size, hidden_size, past_value, present_value,
                            past_present_share_buffer, packed_qkv, dtype);      
        // std::cout << "Apply_attn 3" << std::endl;
        // print_tensor("present_value", present_value, batch_size * kv_num_heads * seqlen_present_kv_cache * head_size, true);
    }

    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        int kv_offset = args.size() == 7 ? 0 : 2;
        // std::cout << "Chkpt 1" << std::endl;
        auto q_shape              = args[0].get_shape();
        auto q_lens               = q_shape.lens();
        const std::size_t batch_size      = q_lens[0];
        const std::size_t sequence_length = q_lens[1];
        // std::cout << "Chkpt 2" << std::endl;
        auto past_key_shape       = args[1 + kv_offset].get_shape();
        auto past_key_lens        = past_key_shape.lens();
        auto past_sequence_length = past_key_lens[2];
        // std::cout << "Chkpt 3" << std::endl;
        // auto total_sequence_length_val = total_sequence_length[0];
        // const int present_kv_seqlen    = std::max(
        //     static_cast<std::size_t>(total_sequence_length_val), past_sequence_length);
        std::size_t q_hidden_size = q_lens[2];
        std::size_t head_size     = q_hidden_size / (num_heads + 2 * kv_num_heads);
        q_hidden_size = head_size * num_heads;
        const bool packed_qkv =
            true; // assume true for now, update if using key/value inputs tensors
        // std::cout << "Chkpt 4" << std::endl;
        std::size_t rotary_dim = args[5 + kv_offset].get_shape().lens()[1] * 2;
        std::size_t present_kv_seqlen; 
        std::size_t total_sequence_length_val;
        
        args[4 + kv_offset].visit([&](auto total_sequence_length){
            std::size_t max_total_sequence_length_val = 0;
            for(auto i = 0; i < batch_size; ++i)
            {
                std::cout << total_sequence_length[i] << std::endl;
                if(total_sequence_length[i] > max_total_sequence_length_val)
                {
                    max_total_sequence_length_val = total_sequence_length[i];
                }
            }
            present_kv_seqlen =     max_total_sequence_length_val;
            total_sequence_length_val = std::max(
                static_cast<std::size_t>(max_total_sequence_length_val), past_sequence_length);
        });

        // std::cout << "batch_size: " << batch_size << std::endl
        //             << "sequence_length: " << sequence_length << std::endl
        //             << "past_sequence_length: " << past_sequence_length << std::endl
        //             //<< "total_sequence_length_val: " << total_sequence_length_val << std::endl
        //             << "present_kv_seqlen: " << present_kv_seqlen << std::endl
        //             << "q_hidden_size: " << q_hidden_size << std::endl
        //             << "head_size: " << head_size << std::endl
        //             << "rotary_dim: " << rotary_dim << std::endl;
        auto output_shape_0 = output_shape.sub_shapes().front();
        argument result{output_shape_0};
        argument qkv_rotary{shape{output_shape_0.type(),
                                  {batch_size,
                                   static_cast<std::size_t>(num_heads + 2 * kv_num_heads),
                                   sequence_length,
                                   head_size}}}; // args[0].get_shape().lens()[2] / num_heads?
        
        shape kv_shape{output_shape_0.type(), 
                            {batch_size, //batch_size
                                static_cast<std::size_t>(kv_num_heads),
                                past_sequence_length,
                                head_size}};
        // std::cout << "pk_data len: " << args[1 + kv_offset].get_shape().elements() << std::endl;
        // std::cout << "pv_data len: " << args[2 + kv_offset].get_shape().elements() << std::endl;
        // argument present_k_out{kv_shape, args[1 + kv_offset].data()};
        // argument present_v_out{kv_shape, args[2 + kv_offset].data()};
        argument present_k_out{kv_shape};
        argument present_v_out{kv_shape};
        // auto present_k_out = args[1 + kv_offset];
        // auto present_v_out = args[2 + kv_offset];
        // std::cout << "Shared" << std::endl;
        argument attention_probs{
            shape{output_shape_0.type(),
                  {batch_size, num_heads, sequence_length, present_kv_seqlen}}};

        // // std::cout << "Num args: " << args.size() << std::endl;
        // for(auto i = 0; i < args.size(); ++i)
        // {
        //     // std::cout << i << ": " << args[i].get_shape() << std::endl;
        // }
        // std::cout << "Chkpt 5" << std::endl;
        //args[0].reshape()
        args[0] = args[0].reshape(shape{output_shape_0.type(),
                                  {batch_size,
                                   sequence_length,
                                   static_cast<std::size_t>(num_heads + 2 * kv_num_heads),
                                   head_size}});
        argument qkv{qkv_rotary.get_shape()};
        visit_all(qkv, args[0])([&](auto a, auto b){
            auto in_shape = args[0].get_shape();
            auto out_shape = qkv.get_shape();
            shape_for_each(in_shape, [&](const auto& idx) {
                std::vector<std::size_t> out_idx{idx[0], idx[2], idx[1], idx[3]};
                a(out_idx.begin(), out_idx.end()) = b(idx.begin(), idx.end());
            });
        });
        // std::cout << "transposed" << std::endl;
        //qkv = qkv.reshape(qkv_rotary.get_shape());

        visit_all(
            result, qkv/* args[0] */, args[1 + kv_offset], args[2 + kv_offset], args[5 + kv_offset], args[6 + kv_offset], qkv_rotary, present_k_out, present_v_out, attention_probs)([&](auto output,
                                                                                 auto query,
                                                                                 auto past_key,
                                                                                 auto past_value,
                                                                                 auto cos_cache,
                                                                                 auto sin_cache,
                                                                                 auto RotaryQKV,
                                                                                 auto present_k,
                                                                                 auto present_v,
                                                                                 auto attn_probs) {
            visit_all(args[3 + kv_offset])([&](auto seqlens_k) {
            
                // std::cout << "Chkpt 6" << std::endl;
                // print_tensor("query", query, batch_size * (num_heads + 2 * kv_num_heads) * sequence_length * head_size, true);
                if(do_rotary)
                {
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
                    // if(packed_qkv)
                    // {
                    auto q_input  = query.begin();
                    auto k_input  = q_input + num_heads * sequence_length * head_size;
                    auto q_rotary = RotaryQKV.begin();
                    auto k_rotary = q_rotary + num_heads * sequence_length * head_size;
                    // }
                    // else
                    // {
                    //     //to-do
                    // }
                    RotaryParameters rotary_params     = {};
                    rotary_params.batch_size           = batch_size;
                    rotary_params.sequence_length      = sequence_length;
                    rotary_params.hidden_size          = q_hidden_size;
                    rotary_params.head_size            = head_size;
                    rotary_params.rotary_embedding_dim = rotary_dim;
                    rotary_params.num_heads            = num_heads;
                    rotary_params.max_sequence_length  = sequence_length; // unused
                    rotary_params.seq_stride           = head_size;
                    rotary_params.head_stride          = sequence_length * rotary_params.seq_stride;
                    rotary_params.batch_stride         = batch_stride;
                    rotary_params.position_ids_format = position_ids_format;
                    rotary_params.transposed          = transposed;
                    rotary_params.seqlen_present_kv_cache = present_kv_seqlen;
                    // std::cout << "Chkpt 7" << std::endl;
                    // rotary_params.print("run_rotary_embedding 1", true);
                    // print_tensor("q_rotary: ", q_rotary, batch_size * (num_heads + 2 * kv_num_heads) * sequence_length * head_size, true);
                    run_rotary_embedding(q_input,
                                         cos_cache.begin(),
                                         sin_cache.begin(),
                                         q_rotary,
                                         rotary_interleaved,
                                         pos_ids.data(),
                                         rotary_params);
                    // print_tensor("q_rotary: ", q_rotary, batch_size * (num_heads + 2 * kv_num_heads) * sequence_length * head_size, true);

                    std::size_t kv_hidden_size        = head_size * kv_num_heads;
                    rotary_params.num_heads   = kv_num_heads;
                    rotary_params.hidden_size = kv_hidden_size;

                    // rotary_params.print("run_rotary_embedding 2");
                    run_rotary_embedding(k_input,
                                         cos_cache.begin(),
                                         sin_cache.begin(),
                                         k_rotary,
                                         rotary_interleaved,
                                         pos_ids.data(),
                                         rotary_params);
                    // print_tensor("k_rotary:", k_rotary, (batch_size * (num_heads + 2 * kv_num_heads) * sequence_length * head_size) - (num_heads * sequence_length * head_size), true);
                    // std::cout << "Chkpt 8" << std::endl;
                    auto v_input            = k_input + kv_num_heads * sequence_length * head_size;
                    auto v_rotary           = k_rotary + kv_num_heads * sequence_length * head_size;
                    rotary_params.num_heads = num_heads;

                    // rotary_params.print("pack_v_into_rotary_QKV");
                    pack_v_into_rotary_QKV(rotary_params, v_input, v_rotary);
                    // print_tensor("v_rotary:", v_rotary, (batch_size * (num_heads + 2 * kv_num_heads) * sequence_length * head_size) - (2 * num_heads * sequence_length * head_size), true);

                    // std::cout << "Chkpt 9" << std::endl;

                    // rotary_params.print("apply_attention");
                    auto Q = RotaryQKV;
                    apply_attention(Q.begin(), past_key.begin(), past_value.begin(), output.begin(), present_k.begin(), present_v.begin(),
                        seqlens_k.begin(), attn_probs.begin(), rotary_params, output_shape_0.type());
                    // print_tensor("output:", output.begin(), output_shape_0.elements(), true);
                    // print_tensor("present_k:", present_k.begin(), kv_shape.elements(), true);
                    // print_tensor("present_v:", present_v.begin(), kv_shape.elements(), true);
                }
            });
        });
        
        // result.visit([&](auto res){
        //     shape_for_each(res.get_shape(), [&](auto idx){
        //         if (res.get_shape().index(idx) < 100)
        //             std::cout << res(idx.begin(), idx.end()) << std::endl;
        //     });
        // });
        // std::cout << "returning" << std::endl;
        return {{result, present_k_out, present_v_out}};
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
