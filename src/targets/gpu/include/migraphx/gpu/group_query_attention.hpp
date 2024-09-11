/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef MIGRAPHX_GUARD_GPU_GROUP_QUERY_ATTENTION_HPP
#define MIGRAPHX_GUARD_GPU_GROUP_QUERY_ATTENTION_HPP

#include <migraphx/stringutils.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/value.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct gqa_parameters
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
    bool transposed; // Whether the input tensor has been transposed into (batch, num_heads,
                     // seq_len, hidden)
    int seqlen_present_kv_cache;

    int do_rotary;
    int kv_num_heads;
    int local_window_size;
    int rotary_interleaved;
    int past_present_share_buffer;
    int packed_qkv;

    std::string make_init_str()
    {
        return 
            std::to_string(scale) + ", " + std::to_string(batch_size) + ", " +
            std::to_string(sequence_length) + ", " + std::to_string(hidden_size) + ", " +
            std::to_string(head_size) + ", " + std::to_string(rotary_embedding_dim) + ", " +
            std::to_string(num_heads) + ", " + std::to_string(max_sequence_length) + ", " +
            std::to_string(head_stride) + ", " + std::to_string(seq_stride) + ", " +
            std::to_string(batch_stride) + ", " + std::to_string(position_ids_format) + ", " +
            std::to_string(transposed) + ", " + std::to_string(seqlen_present_kv_cache) + ", " +
            std::to_string(do_rotary) + ", " + std::to_string(kv_num_heads) + ", " +
            std::to_string(local_window_size) + ", " + std::to_string(rotary_interleaved) + ", " +
            std::to_string(past_present_share_buffer) + ", " + std::to_string(packed_qkv);
    }
};

static inline gqa_parameters init_params(const std::vector<shape>& inputs, const value& v)
{
    assert(v.contains("num_heads"));
    auto num_heads = v.at("num_heads").to<int>();
    assert(v.contains("kv_num_heads"));
    auto kv_num_heads = v.at("kv_num_heads").to<int>();
    assert(v.contains("do_rotary"));
    auto do_rotary = v.at("do_rotary").to<int>();
    assert(v.contains("local_window_size"));
    auto local_window_size = v.at("local_window_size").to<int>();
    assert(v.contains("rotary_interleaved"));
    auto rotary_interleaved = v.at("rotary_interleaved").to<int>();
    assert(v.contains("scale"));
    auto scale = v.at("scale").to<float>();
    assert(v.contains("present_kv_seqlen"));
    auto present_kv_seqlen = v.at("present_kv_seqlen").to<std::size_t>();

    auto q_shape                      = inputs[0];
    auto q_lens                       = q_shape.lens();
    const std::size_t batch_size      = q_lens[0];
    const std::size_t sequence_length = q_lens[2];
    std::size_t head_size             = q_lens[3];
    auto q_hidden_size                = kv_num_heads * head_size;

    std::size_t rotary_dim        = inputs[3].lens()[1] * 2;
    bool packed_qkv = true;
    auto seq_stride   = head_size;
    auto head_stride  = sequence_length * seq_stride;
    auto batch_stride = (packed_qkv ? (num_heads + 2 * kv_num_heads) : num_heads) * head_stride;
    auto position_ids_format = sequence_length == 1 ? 1 : 0;
    bool transposed          = true;
    bool past_present_share_buffer = true;
    gqa_parameters gqa_params;
    gqa_params.batch_size              = batch_size;
    gqa_params.sequence_length         = sequence_length;
    gqa_params.hidden_size             = q_hidden_size;
    gqa_params.head_size               = head_size;
    gqa_params.rotary_embedding_dim    = rotary_dim;
    gqa_params.num_heads               = num_heads;
    gqa_params.max_sequence_length     = sequence_length;
    gqa_params.seq_stride              = head_size;
    gqa_params.head_stride             = sequence_length * gqa_params.seq_stride;
    gqa_params.batch_stride            = batch_stride;
    gqa_params.position_ids_format     = position_ids_format;
    gqa_params.transposed              = transposed;
    gqa_params.seqlen_present_kv_cache = present_kv_seqlen;
    gqa_params.do_rotary               = do_rotary;
    gqa_params.kv_num_heads            = kv_num_heads;
    gqa_params.local_window_size       = local_window_size;
    gqa_params.rotary_interleaved      = rotary_interleaved;
    gqa_params.scale                   = scale;
    gqa_params.past_present_share_buffer = past_present_share_buffer;
    gqa_params.packed_qkv              = packed_qkv;

    return gqa_params;
}

} // namespace gpu

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_GPU_GROUP_QUERY_ATTENTION_HPP
