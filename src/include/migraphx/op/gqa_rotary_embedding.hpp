/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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
 *
 */
#ifndef MIGRAPHX_GUARD_OPERATORS_GQA_ROTARY_EMBEDDING_HPP
#define MIGRAPHX_GUARD_OPERATORS_GQA_ROTARY_EMBEDDING_HPP

#include <migraphx/check_shapes.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/par_for.hpp>
#include <migraphx/argument.hpp>
#include <fstream>
#include <iostream>
#include <iomanip>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct rotary_parameters
{
    std::size_t batch_size           = 0;     // Batch size used by input
    std::size_t sequence_length      = 0;     // Sequence length used by input
    std::size_t hidden_size          = 0;     // Hidden size used by input
    std::size_t head_size            = 0;     // Head size
    std::size_t rotary_embedding_dim = 0;     // Rotary embedding dimension.
    std::size_t num_heads            = 0;     // num_heads = hidden_size / head_size
    std::size_t max_sequence_length  = 0;     // Sequence length used by cos/sin cache
    std::size_t head_stride          = 0;     // Head stride
    std::size_t seq_stride           = 0;     // Sequence stride
    std::size_t batch_stride         = 0;     // Batch stride
    bool position_ids_use_batch      = false; // Format of position ids - false is (1), true is
                                              // (batch_size, sequence_length)
};

struct gqa_rotary_embedding
{
    // std::size_t kv_num_heads      = 0;
    // std::size_t num_heads         = 1;
    // bool rotary_interleaved       = false;

    // template <class Self, class F>
    // static auto reflect(Self& self, F f)
    // {
    //     return pack(f(self.kv_num_heads, "kv_num_heads"),
    //                 f(self.num_heads, "num_heads"),
    //                 f(self.rotary_interleaved, "rotary_interleaved"));
    // }
    bool do_rotary                = false;
    std::size_t kv_num_heads      = 0;
    int local_window_size         = -1;
    std::size_t num_heads         = 1;
    bool rotary_interleaved       = false;
    float scale                   = 1.0;

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

    std::string name() const { return "gqa_rotary_embedding"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        return inputs.front();
    }

    template <class T>
    void run_rotary_embedding(T input,
                              T cos_cache,
                              T sin_cache,
                              T output,
                              bool interleaved,
                              const std::size_t* pos_ids,
                              rotary_parameters parameters) const
    {
        const std::size_t batch_size             = parameters.batch_size;
        const std::size_t sequence_length        = parameters.sequence_length;
        const std::size_t n_heads                = parameters.num_heads;
        const std::size_t head_size              = parameters.head_size;
        const std::size_t head_stride            = parameters.head_stride;
        const std::size_t seq_stride             = parameters.seq_stride;
        const std::size_t batch_stride           = parameters.batch_stride;
        const std::size_t position_ids_use_batch = parameters.position_ids_use_batch;
        const std::size_t rotary_emb_dim         = parameters.rotary_embedding_dim;
        const std::size_t half_rotary_emb_dim    = rotary_emb_dim / 2;

        const std::size_t loop_len = batch_size * sequence_length * n_heads;
        par_for(loop_len, [&](const auto idx) {
            const std::size_t b            = (idx / n_heads) / sequence_length;
            const std::size_t s            = (idx / n_heads) % sequence_length;
            const std::size_t n            = idx % n_heads;
            const std::size_t block_offset = b * batch_stride + s * seq_stride + n * head_stride;
            auto input_data                = input + block_offset;
            auto output_data               = output + block_offset;

            // Cache is (M, H/2) or (M, rotary_embedding_dim/2)
            const std::size_t position_id =
                position_ids_use_batch ? pos_ids[b * sequence_length + s] : pos_ids[0] + s;
            const std::size_t cache_offset = position_id * half_rotary_emb_dim;
            auto cos_data                  = cos_cache + cache_offset;
            auto sin_data                  = sin_cache + cache_offset;

            std::size_t cache_idx = 0;
            float sign            = 0.0;
            std::size_t j         = 0;
            for(std::size_t i = 0; i < rotary_emb_dim; i++)
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
            std::copy(
                input_data + rotary_emb_dim, input_data + head_size, output_data + rotary_emb_dim);
        });
    }

    template <class T>
    void pack_v_into_rotary_qkv(rotary_parameters parameters, const T input, T output) const
    {
        const std::size_t loop_len =
            parameters.batch_size * parameters.sequence_length * kv_num_heads;
        par_for(loop_len, [&](const auto idx) {
            const std::size_t b            = (idx / kv_num_heads) / parameters.sequence_length;
            const std::size_t s            = (idx / kv_num_heads) % parameters.sequence_length;
            const std::size_t n            = idx % kv_num_heads;
            const std::size_t block_offset = b * parameters.batch_stride +
                                             s * parameters.seq_stride + n * parameters.head_stride;
            const T input_data = input + block_offset;
            T output_data      = output + block_offset;
            for(std::size_t i = 0; i < parameters.head_size; i++)
            {
                output_data[i] = input_data[i];
            }
        });
    }

    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        auto q_shape                      = args[0].get_shape();
        const auto& q_lens                = q_shape.lens();
        const std::size_t batch_size      = q_lens[0];
        const std::size_t sequence_length = q_lens[1];
        std::size_t q_hidden_size         = q_lens[2];
        std::size_t head_size             = q_hidden_size / (num_heads + 2 * kv_num_heads);
        q_hidden_size                     = head_size * num_heads;
        std::size_t rotary_dim            = args[2].get_shape().lens()[1] * 2;

        argument result{output_shape};

        visit_all(result,
                  args[0],
                  args[2],
                  args[3])([&](auto output,
                                       auto query,
                                       auto cos_cache,
                                       auto sin_cache) {
            visit_all(args[1])([&](auto seqlens_k) {
                auto seq_stride             = head_size;
                auto head_stride            = sequence_length * seq_stride;
                auto batch_stride           = num_heads + 2 * kv_num_heads;
                auto position_ids_use_batch = sequence_length == 1;
                std::vector<std::size_t> pos_ids(sequence_length == 1 ? batch_size : 1);
                if(sequence_length == 1)
                {
                    std::copy(seqlens_k.begin(), seqlens_k.begin() + batch_size, pos_ids.begin());
                }
                else
                {
                    pos_ids[0] = 0;
                }
                auto q_input  = query.begin();
                auto k_input  = q_input + num_heads * sequence_length * head_size;
                auto q_rotary = output.begin();
                auto k_rotary = q_rotary + num_heads * sequence_length * head_size;

                rotary_parameters rotary_params            = {};
                rotary_params.batch_size                = batch_size;
                rotary_params.sequence_length           = sequence_length;
                rotary_params.hidden_size               = q_hidden_size;
                rotary_params.head_size                 = head_size;
                rotary_params.rotary_embedding_dim      = rotary_dim;
                rotary_params.num_heads                 = num_heads;
                rotary_params.max_sequence_length       = sequence_length;
                rotary_params.seq_stride                = head_size;
                rotary_params.head_stride               = head_stride;
                rotary_params.batch_stride              = batch_stride;
                rotary_params.position_ids_use_batch    = position_ids_use_batch;

                
                run_rotary_embedding(q_input,
                                        cos_cache.begin(),
                                        sin_cache.begin(),
                                        q_rotary,
                                        rotary_interleaved,
                                        pos_ids.data(),
                                        rotary_params);

                std::size_t kv_hidden_size = head_size * kv_num_heads;
                rotary_params.num_heads       = kv_num_heads;
                rotary_params.hidden_size     = kv_hidden_size;

                run_rotary_embedding(k_input,
                                        cos_cache.begin(),
                                        sin_cache.begin(),
                                        k_rotary,
                                        rotary_interleaved,
                                        pos_ids.data(),
                                        rotary_params);

                auto v_input         = k_input + kv_num_heads * sequence_length * head_size;
                auto v_rotary        = k_rotary + kv_num_heads * sequence_length * head_size;
                rotary_params.num_heads = num_heads;

                pack_v_into_rotary_qkv(rotary_params, v_input, v_rotary);
            });
        });

        return result;
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
