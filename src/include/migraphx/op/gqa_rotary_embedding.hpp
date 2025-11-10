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
#ifndef MIGRAPHX_GUARD_OPERATORS_GQA_GQA_ROTARY_EMBEDDING_HPP
#define MIGRAPHX_GUARD_OPERATORS_GQA_GQA_ROTARY_EMBEDDING_HPP

#include <migraphx/check_shapes.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/par_for.hpp>
#include <migraphx/argument.hpp>
#include <cstddef>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct gqa_rotary_embedding
{
    size_t num_heads        = 1;
    size_t kv_num_heads     = 1;
    bool interleaved = false;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.num_heads, "num_heads"),
                    f(self.kv_num_heads, "kv_num_heads"),
                    f(self.interleaved, "interleaved"));
    }

    std::string name() const { return "gqa_rotary_embedding"; }

    shape compute_shape(std::vector<shape> inputs) const { return inputs.front(); }

    struct rotary_parameters
    {
        size_t batch_size           = 0;
        size_t sequence_length      = 0;
        size_t head_size            = 0;
        size_t num_heads            = 0;
        size_t rotary_embedding_dim = 0;
        size_t max_sequence_length  = 0; // Sequence length used by cos/sin cache
        size_t head_stride          = 0;
        size_t seq_stride           = 0;
        size_t batch_stride         = 0;
        bool position_ids_use_batch = false;
    };

    template <class T>
    void run_rotary_embedding(T input,
                              T cos_cache,
                              T sin_cache,
                              T output,
                              const size_t* pos_ids,
                              rotary_parameters params) const
    {
        const size_t half_rotary_emb_dim = params.rotary_embedding_dim / 2;

        const size_t loop_len = params.batch_size * params.sequence_length * params.num_heads;
        par_for(loop_len, [&](auto idx) {
            const size_t b = (idx / params.num_heads) / params.sequence_length;
            const size_t s = (idx / params.num_heads) % params.sequence_length;
            const size_t n = idx % params.num_heads;
            const size_t block_offset =
                b * params.batch_stride + s * params.seq_stride + n * params.head_stride;
            auto input_data  = input + block_offset;
            auto output_data = output + block_offset;

            const size_t position_id = params.position_ids_use_batch
                                           ? pos_ids[b * params.sequence_length + s]
                                           : pos_ids[0] + s;

            const size_t cache_offset = position_id * half_rotary_emb_dim;
            auto cos_data             = cos_cache + cache_offset;
            auto sin_data             = sin_cache + cache_offset;

            size_t cache_idx = 0;
            float sign       = 0.0;
            size_t j         = 0;
            for(size_t i = 0; i < params.rotary_embedding_dim; i++)
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
                    j         = (i + half_rotary_emb_dim) % params.rotary_embedding_dim;
                }
                output_data[i] = input_data[i] * cos_data[cache_idx] +
                                 sign * input_data[j] * sin_data[cache_idx];
            }
            std::copy(input_data + params.rotary_embedding_dim,
                      input_data + params.head_size,
                      output_data + params.rotary_embedding_dim);
        });
    }

    template <class T>
    void pack_v_into_rotary_qkv(rotary_parameters params, const T input, T output) const
    {
        const size_t loop_len = params.batch_size * params.sequence_length * kv_num_heads;
        par_for(loop_len, [&](const auto idx) {
            const size_t b = (idx / kv_num_heads) / params.sequence_length;
            const size_t s = (idx / kv_num_heads) % params.sequence_length;
            const size_t n = idx % kv_num_heads;
            const size_t block_offset =
                b * params.batch_stride + s * params.seq_stride + n * params.head_stride;
            const T input_data = input + block_offset;
            T output_data      = output + block_offset;
            for(size_t i = 0; i < params.head_size; i++)
            {
                output_data[i] = input_data[i];
            }
        });
    }

    // Args:
    // 0 - packed QKV (batch_size, num_heads +  2 * kv_num_heads, sequence_length, head_size)
    // 1 - seqlens_k (batch_size)
    // 2 - cos cache (max_rotary_sequence_length, head_size / 2)
    // 3 - sin cache (max_rotary_sequence_length, head_size / 2)
    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        rotary_parameters params;

        const auto& qkv_lens        = args[0].get_shape().lens();
        params.batch_size           = qkv_lens[0];
        params.sequence_length      = qkv_lens[2];
        params.head_size            = qkv_lens[3];
        const auto& cache_lens      = args[2].get_shape().lens();
        params.max_sequence_length  = cache_lens[0];
        params.rotary_embedding_dim = cache_lens[1] * 2;
        params.seq_stride           = params.head_size;
        params.head_stride          = params.sequence_length * params.seq_stride;
        params.batch_stride =
            (num_heads + 2 * kv_num_heads) * params.sequence_length * params.head_size;
        params.position_ids_use_batch = params.sequence_length == 1;

        argument result{output_shape};

        visit_all(result, args[0], args[2], args[3])(
            [&](auto output, auto qkv, auto cos_cache, auto sin_cache) {
                visit_all(args[1])([&](auto seqlens_k) {
                    std::vector<size_t> pos_ids(params.position_ids_use_batch ? params.batch_size
                                                                              : 1);
                    if(params.position_ids_use_batch)
                    {
                        std::transform(seqlens_k.begin(),
                                       seqlens_k.end(),
                                       pos_ids.begin(),
                                       [](auto len) { return len; });
                    }
                    else
                    {
                        pos_ids[0] = 0;
                    }

                    auto q_input  = qkv.begin();
                    auto k_input  = q_input + num_heads * params.head_stride;
                    auto q_rotary = output.begin();
                    auto k_rotary = q_rotary + num_heads * params.head_stride;

                    params.num_heads = num_heads;
                    run_rotary_embedding(q_input,
                                         cos_cache.begin(),
                                         sin_cache.begin(),
                                         q_rotary,
                                         pos_ids.data(),
                                         params);

                    params.num_heads = kv_num_heads;
                    run_rotary_embedding(k_input,
                                         cos_cache.begin(),
                                         sin_cache.begin(),
                                         k_rotary,
                                         pos_ids.data(),
                                         params);

                    auto v_input     = k_input + kv_num_heads * params.head_stride;
                    auto v_rotary    = k_rotary + kv_num_heads * params.head_stride;
                    params.num_heads = num_heads;

                    pack_v_into_rotary_qkv(params, v_input, v_rotary);
                });
            });

        return result;
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
