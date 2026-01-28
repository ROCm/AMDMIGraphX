/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef MIGRAPHX_GUARD_RTGLIB_REWRITE_KV_CACHE_HPP
#define MIGRAPHX_GUARD_RTGLIB_REWRITE_KV_CACHE_HPP

#include <string>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module;

/**
 * Rewrite standard concat operations for KV-cache (past_key/past_value concatenation)
 * into the optimized concat_past_present operator.
 *
 * This pass detects patterns where:
 * - A graph parameter named like "past.*.key" or "past.*.value" is concatenated
 *   with new key/value tensors along the sequence dimension (axis=2)
 * - The result becomes "present.*.key" or "present.*.value" output
 *
 * The concat_past_present operator enables:
 * - Pre-allocated KV-cache buffers (compile once, run many)
 * - In-place cache updates (no memory reallocation per step)
 * - Integration with fuse_attention for optimized attention kernels
 *
 * REQUIREMENTS for the ONNX model:
 * 1. Must have "seqlens_k" parameter with shape (batch, 1) of type int32
 *    indicating valid past sequence lengths per batch element
 * 2. Past KV tensors should be pre-allocated to max_sequence_length
 *    (fixed size buffer that grows in-place)
 * 3. Parameter naming must match pattern: past.{N}.key, past.{N}.value,
 *    past_key_values.{N}.key, etc.
 *
 * If seqlens_k is not present in the model, the optimization is skipped.
 * 
 * Example ONNX export modification for decoder_with_past:
 * @code
 * # Add seqlens_k as input to your decoder wrapper
 * def forward(self, input_ids, encoder_hidden_states, encoder_attention_mask,
 *             seqlens_k, *past_kv):  # seqlens_k: [batch, 1] int32
 *     # Pre-allocate past_kv to max_sequence_length before first inference
 *     ...
 * @endcode
 */
struct MIGRAPHX_EXPORT rewrite_kv_cache
{
    std::string name() const { return "rewrite_kv_cache"; }
    void apply(module& m) const;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
