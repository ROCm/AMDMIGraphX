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
#ifndef MIGRAPHX_GUARD_MIGRAPHX_FUSE_ATTENTION_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_FUSE_ATTENTION_HPP

#include <migraphx/config.hpp>
#include <string>
#include <optional>
#include <cstddef>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module_pass_manager;

/// Configuration for paged attention transformation
struct paged_attention_config
{
    std::size_t tokens_per_block = 16;  // Number of tokens per block
    std::size_t num_blocks       = 0;   // Total number of blocks in the pool (0 = auto-calculate)
    
    /// When true, uses combined KV cache format with dimension 2 for K/V separation:
    ///   - Cache shape: {2, num_blocks, tokens_per_block, num_kv_heads, head_dim}
    ///                   ^-- 0 = Key, 1 = Value
    ///   - Block table shape: {batch_size, 2, max_blocks_per_seq}
    ///                                     ^-- 0 = K pointers, 1 = V pointers
    /// When false, K and V are processed as separate tensors.
    bool use_combined_kv = true;
};

struct MIGRAPHX_EXPORT fuse_attention
{
    bool attn_enabled = false;
    std::optional<std::size_t> flash_decoding_num_splits = std::nullopt;
    // paged_attention_config paged_attn_config = {};

    std::string name() const { return "fuse_attention"; }
    void apply(module_pass_manager& mpm) const;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_MIGRAPHX_FUSE_ATTENTION_HPP
