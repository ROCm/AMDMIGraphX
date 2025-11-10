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
 */
#ifndef MIGRAPHX_GUARD_KERNELS_GROUP_QUERY_ATTENTION_HPP
#define MIGRAPHX_GUARD_KERNELS_GROUP_QUERY_ATTENTION_HPP

#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/algorithm.hpp>
#include <migraphx/kernels/integral_constant.hpp>
#include <migraphx/kernels/tensor_view.hpp>
#include <migraphx/kernels/type_traits.hpp>

namespace migraphx {

template <class T1,
          class T2,
          class T3,
          class T4,
          class T5,
          class T6,
          class T7,
          class T8,
          class T9,
          class T10,
          class T11,
          class T12,
          class T13>
struct gqa_parameters
{
    T1 batch_size;               // Batch size used by input
    T2 sequence_length;          // Sequence length used by input
    T3 head_size;                // Head size
    T4 rotary_embedding_dim;     // Rotary embedding dimension.
    T5 num_heads;                // num_heads = hidden_size / head_size
    T6 max_sequence_length;      // Sequence length used by cos/sin cache
    T7 head_stride;              // Head stride
    T8 seq_stride;               // Sequence stride
    T9 batch_stride;             // Batch stride
    T10 position_ids_format;     // Format of position ids - 0 is (1), 1 is (batch_size,
                                 // sequence_length)
    T11 seqlen_present_kv_cache; // Sequence length of present kv-cache (4096 when using
                                 // shared buffer)
    T12 kv_num_heads;            // Number of attention heads for k and v
    T13 interleaved;      // Rotate using interleaved pattern. Default value is 0 (False).
};

template <class... Ts>
__device__ gqa_parameters<Ts...> make_gqa_parameters(Ts... ts)
{
    return {ts...};
}

} // namespace migraphx
#endif
