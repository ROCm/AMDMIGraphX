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
#ifndef MIGRAPHX_GUARD_OPERATORS_CONCAT_PAST_PRESENT_HPP
#define MIGRAPHX_GUARD_OPERATORS_CONCAT_PAST_PRESENT_HPP

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

struct cache_parameters
{
    std::size_t batch_size           = 0;     // Batch size used by input
    std::size_t sequence_length      = 0;     // Sequence length used by input
    std::size_t head_size            = 0;     // Head size
    std::size_t num_heads            = 0;     // num_heads = hidden_size / head_size
    bool position_ids_use_batch      = false; // Format of position ids - false is (1), true is
                                              // (batch_size, sequence_length)
    std::size_t seqlen_present_kv_cache = 0;  // Sequence length of present kv-cache
                                              // (4096 when using shared buffer)
    bool past_present_share_buffer = false;   // Whether to use same buffer for KV-cache
                                              // inputs and outputs
};

struct concat_past_present
{
    // std::size_t kv_num_heads      = 1;
    // std::size_t num_heads         = 1;

    // template <class Self, class F>
    // static auto reflect(Self& self, F f)
    // {
    //     return pack(f(self.kv_num_heads, "kv_num_heads"),
    //                 f(self.num_heads, "num_heads"));
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

    std::string name() const { return "concat_past_present"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(3);
        return inputs.back();
    }

    template <class T>
    void copy_data(T destination, const T source, std::size_t n) const
    {
        par_for(n, [&](auto i) { destination[i] = source[i]; });
    }

    template <typename T>
    T concat_state_chunk(const T past,
                         const T chunk,
                         T present,
                         std::size_t present_buff_chunk_length,
                         std::size_t past_buff_chunk_length,
                         std::size_t past_chunk_length,
                         std::size_t new_chunk_length,
                         bool is_prompt,
                         bool past_present_share_buffer,
                         std::ptrdiff_t i) const
    {
        T start = present + i * present_buff_chunk_length;

        T p = start;
        if(not is_prompt)
        {
            if(not past_present_share_buffer)
            {
                const T src_past = past + i * past_buff_chunk_length;
                copy_data(p, src_past, past_chunk_length);
            }
            p += past_chunk_length;
        }
        copy_data(p, chunk, new_chunk_length);
        return start;
    }

    template <class T, class U>
    void update_cache(T past_key,  
                        U seqlens_k,     
                        T present_key,
                        cache_parameters params) const
    {
        const std::size_t batch_size                     = params.batch_size;
        const std::size_t sequence_length                = params.sequence_length;
        const std::size_t head_size                      = params.head_size;
        const std::size_t past_buffer_sequence_length    = params.seqlen_present_kv_cache;
        const std::size_t present_buffer_sequence_length = past_buffer_sequence_length;
        const bool past_present_share_buffer             = params.past_present_share_buffer;

        const bool is_prompt = sequence_length != 1;
        const std::size_t packed_batch_stride =
            (num_heads + 2 * kv_num_heads) * sequence_length * head_size;
        const std::size_t kv_num_heads_factor    = num_heads / kv_num_heads;
        const std::size_t q_input_chunk_length   = sequence_length * head_size;             // S x H
        const std::size_t kv_input_chunk_length  = sequence_length * head_size;             // L x H
        const std::size_t past_buff_chunk_length = past_buffer_sequence_length * head_size; // L x H
        const std::size_t present_buff_chunk_length =
            present_buffer_sequence_length * head_size; // T x H

        const std::size_t loop_len = batch_size * num_heads;

        par_for(loop_len, [&](const auto i) {
            const std::size_t batch_index = i / num_heads;
            const std::size_t head_index  = i % num_heads;
            const std::size_t past_seqlen =
                sequence_length == 1 ? seqlens_k[batch_index] : past_buffer_sequence_length;
            const std::size_t past_chunk_length = past_seqlen * head_size;

            auto current = present_key + packed_batch_stride * batch_index +
                     kv_input_chunk_length * (head_index / kv_num_heads_factor);
            concat_state_chunk(past_key,
                                   current,
                                   present_key,
                                   present_buff_chunk_length,
                                   past_buff_chunk_length,
                                   past_chunk_length,
                                   kv_input_chunk_length,
                                   is_prompt,
                                   past_present_share_buffer,
                                   i / kv_num_heads_factor);
        });
    }

    argument compute(const shape& /* output_shape */, std::vector<argument> args) const
    {
        auto present = args[0];
        auto seqlens = args[1];
        auto past = args[2];
        auto present_shape                   = present.get_shape();
        const auto& present_lens             = present_shape.lens();
        const std::size_t batch_size      = present_lens[0];
        const std::size_t sequence_length = present_lens[1];
        auto past_kv_shape               = past.get_shape();
        const auto& past_kv_lens         = past_kv_shape.lens();
        auto past_sequence_length         = past_kv_lens[2];
        std::size_t head_size             = present_lens[2] / (num_heads + 2 * kv_num_heads);

        cache_parameters cache_params            = {};
        cache_params.batch_size                = batch_size;
        cache_params.sequence_length           = sequence_length;
        cache_params.head_size                 = head_size;
        cache_params.num_heads                 = num_heads;
        cache_params.seqlen_present_kv_cache   = past_sequence_length;
        cache_params.past_present_share_buffer = false;

        visit_all(past, present)([&](auto past_kv, auto present_kv){
            visit_all(seqlens)([&](auto seqlens_kv) {
                update_cache(past_kv.begin(), seqlens_kv.begin(), present_kv.begin(), cache_params);
            });
        });
        

        return past;
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
