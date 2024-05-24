#ifndef MIGRAPHX_GUARD_OPERATORS_GROUP_QUERY_ATTENTION_HPP
#define MIGRAPHX_GUARD_OPERATORS_GROUP_QUERY_ATTENTION_HPP

#include <migraphx/check_shapes.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/par_for.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct RotaryParameters {
    int batch_size;            // Batch size used by input
    int sequence_length;       // Sequence length used by input
    int hidden_size;           // Hidden size used by input
    int head_size;             // Head size
    int rotary_embedding_dim;  // Rotary embedding dimension.
    int num_heads;             // num_heads = hidden_size / head_size
    int max_sequence_length;   // Sequence length used by cos/sin cache
    int head_stride;           // Head stride
    int seq_stride;            // Sequence stride
    int batch_stride;          // Batch stride
    int position_ids_format;   // Format of position ids - 0 is (1), 1 is (batch_size, sequence_length)
    bool transposed;           // Whether the input tensor has been transposed into (batch, num_heads, seq_len, hidden)
};


struct group_query_attention
{
    int do_rotary = 0;
    int kv_num_heads = 0;
    int local_window_size = -1;
    int num_heads = 1;
    int rotary_interleaved = 0;
    float scale = 1.0;

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
        check_shapes{inputs, *this};
        auto query_lens = inputs.front().lens();
        std::vector<std::size_t> output_lens{query_lens.at(0), query_lens.at(1), 4096};
        shape output_shape{inputs.front().type(), output_lens};
        return output_shape;
    }

    template<class T>
    void run_rotary_embedding(T input, T cos_cache, T sin_cache, T output, bool interleaved, const int64_t* pos_ids, RotaryParameters parameters) const
    {
        const int batch_size = parameters.batch_size;
        const int sequence_length = parameters.sequence_length;
        const int n_heads = parameters.num_heads;
        const int head_size = parameters.head_size;
        const int head_stride = parameters.head_stride;
        const int seq_stride = parameters.seq_stride;
        const int batch_stride = parameters.batch_stride;
        const int position_ids_format = parameters.position_ids_format;
        const int rotary_emb_dim = parameters.rotary_embedding_dim;
        const int half_rotary_emb_dim = rotary_emb_dim / 2;

        const int loop_len = batch_size * sequence_length * n_heads;
        par_for(loop_len, [&](const auto idx) {
            const int b = static_cast<int>((idx / n_heads) / sequence_length);
            const int s = static_cast<int>((idx / n_heads) % sequence_length);
            const int n = static_cast<int>(idx % n_heads);
            const int block_offset = b * batch_stride + s * seq_stride + n * head_stride;
            auto input_data = input + block_offset;
            auto output_data = output + block_offset;

            // Cache is (M, H/2) or (M, rotary_embedding_dim/2)
            const int position_id = (position_ids_format == 0)
                                        ? static_cast<int>(pos_ids[0]) + s
                                        : static_cast<int>(pos_ids[b * sequence_length + s]);
            const int cache_offset = position_id * half_rotary_emb_dim;
            auto cos_data = cos_cache + cache_offset;
            auto sin_data = sin_cache + cache_offset;

            int cache_idx = 0;
            float sign = 0.0; 
            int j = 0;
            for (int i = 0; i < rotary_emb_dim; i++) {
                if (interleaved) { 
                cache_idx = (i / 2) % half_rotary_emb_dim;
                sign = (i % 2 == 0) ? -1.0 : 1.0;
                j = (i % 2 == 0) ? i + 1 : i - 1;  // i - sign
                } else {
                cache_idx = i % half_rotary_emb_dim;
                sign = (i < half_rotary_emb_dim) ? -1.0 : 1.0;
                j = (i + half_rotary_emb_dim) % rotary_emb_dim;
                }
                output_data[i] = input_data[i] * cos_data[cache_idx] + sign * input_data[j] * sin_data[cache_idx];
            }
            for (int i = rotary_emb_dim; i < head_size; i++) {
                output_data[i] = input_data[i];
            }
        });
    }

    template <class T>
    void pack_v_into_rotary_QKV(RotaryParameters parameters, const T* input, T* output) const
    {
        const int loop_len = parameters.batch_size * parameters.sequence_length * kv_num_heads;
        const double cost = static_cast<double>(parameters.head_size);
        par_for(loop_len, [&](const auto idx) {
            const int b = static_cast<int>((idx / kv_num_heads) / parameters.sequence_length);
            const int s = static_cast<int>((idx / kv_num_heads) % parameters.sequence_length);
            const int n = static_cast<int>(idx % kv_num_heads);
            const int block_offset = b * parameters.batch_stride + s * parameters.seq_stride + n * parameters.head_stride;
            const T* input_data = input + block_offset;
            T* output_data = output + block_offset;
            for (int i = 0; i < parameters.head_size; i++) {
                output_data[i] = input_data[i];
            }
        });
    }

    template<class T, class U>
    void apply_attention(T Q, T K, T V, T past_key, T past_value, T output, T present_key, T present_value, U seqlens_k, RotaryParameters parameters) const
    {
        ////// Resume here
    }   


    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};
        argument qkv_rotary{shape{output_shape.type(), {output_shape.lens()[0], static_cast<std::size_t>(kv_num_heads), output_shape.lens()[1], args[0].get_shape().lens()[2] / num_heads}}};
        // std::cout << "Num args: " << args.size() << std::endl;
        // for(auto i = 0; i < args.size(); ++i)
        // {
        //     std::cout << i << ": " << args[i].get_shape() << std::endl;
        // }
        visit_all(result, args[0], args[3], args[4], args[7], args[8], qkv_rotary)([&](auto output, auto query, auto past_key, auto past_value, auto cos_cache, auto sin_cache, auto RotaryQKV) {
            visit_all(args[5], args[6])([&](auto seqlens_k, auto total_sequence_length){
                std::cout << "Chkpt 1" << std::endl;
                auto q_shape = query.get_shape();
                auto q_lens = q_shape.lens();
                const int batch_size = q_lens[0];
                const int sequence_length = q_lens[1];
                std::cout << "Chkpt 2" << std::endl;
                auto past_key_shape = past_key.get_shape();
                auto past_key_lens = past_key_shape.lens();
                auto past_sequence_length = past_key_lens[2];
                std::cout << "Chkpt 3" << std::endl;
                auto total_sequence_length_val = total_sequence_length[0];
                const int present_kv_seqlen = std::max(static_cast<std::size_t>(total_sequence_length_val), past_sequence_length);
                int q_hidden_size = q_lens[2];
                int head_size = q_hidden_size / num_heads;
                const bool packed_qkv = true; //assume true for now, update if using key/value inputs tensors 
                int rotary_dim = cos_cache.get_shape().lens()[1] * 2;

                std::cout << "batch_size: " << batch_size << std::endl
                          << "sequence_length: " << sequence_length << std::endl
                          << "past_sequence_length: " << past_sequence_length << std::endl
                          << "total_sequence_length_val: " << total_sequence_length_val << std::endl
                          << "present_kv_seqlen: " << present_kv_seqlen << std::endl
                          << "q_hidden_size: " << q_hidden_size << std::endl
                          << "head_size: " << head_size << std::endl
                          << "rotary_dim: " << rotary_dim << std::endl;
                
                if(do_rotary)
                {
                    auto seq_stride = head_size;
                    auto head_stride = sequence_length * seq_stride;
                    auto batch_stride = (packed_qkv ? (num_heads + 2 * kv_num_heads) : num_heads) * head_stride;
                    auto position_ids_format = sequence_length == 1 ? 1 : 0;
                    bool transposed = true;
                    std::vector<int64_t> pos_ids(sequence_length == 1 ? batch_size : 1);
                    if (sequence_length == 1) 
                    {
                        for (int b = 0; b < batch_size; b++) {
                            pos_ids[b] = static_cast<int64_t>(seqlens_k[b]);
                        }
                    } 
                    else 
                    {
                        pos_ids[0] = static_cast<int64_t>(0);
                    }
                    // if(packed_qkv)
                    // {
                        auto q_input = query.begin();
                        auto k_input = q_input + num_heads * sequence_length * head_size;
                        auto q_rotary = RotaryQKV.begin();
                        auto k_rotary = q_rotary + num_heads * sequence_length * head_size;
                    // }
                    // else
                    // {
                    //     //to-do
                    // }
                    RotaryParameters rotary_params = {};
                    rotary_params.batch_size = batch_size;
                    rotary_params.sequence_length = sequence_length;
                    rotary_params.hidden_size = q_hidden_size;
                    rotary_params.head_size = head_size;
                    rotary_params.rotary_embedding_dim = rotary_dim;
                    rotary_params.num_heads = num_heads;
                    rotary_params.max_sequence_length = sequence_length;  // unused
                    rotary_params.seq_stride = head_size;
                    rotary_params.head_stride = sequence_length * rotary_params.seq_stride;
                    rotary_params.batch_stride = (packed_qkv ? (num_heads + 2 * kv_num_heads) : num_heads) * rotary_params.head_stride;
                    rotary_params.position_ids_format = sequence_length == 1 ? 1 : 0;
                    rotary_params.transposed = true;

                    run_rotary_embedding(q_input, cos_cache.begin(), sin_cache.begin(), q_rotary, rotary_interleaved, pos_ids.data(), rotary_params);

                    int kv_hidden_size = head_size * kv_num_heads;
                    rotary_params.num_heads = kv_num_heads;
                    rotary_params.hidden_size = kv_hidden_size;

                    run_rotary_embedding(k_input, cos_cache.begin(), sin_cache.begin(), k_rotary, rotary_interleaved, pos_ids.data(), rotary_params);

                    auto v_input = k_input + kv_num_heads * sequence_length * head_size;
                    auto v_rotary = k_rotary + kv_num_heads * sequence_length * head_size;
                    rotary_params.num_heads = num_heads;

                    pack_v_into_rotary_QKV(rotary_params, v_input, v_rotary);


                }
            });
        });

        return result;
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
