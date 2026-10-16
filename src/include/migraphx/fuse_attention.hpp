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
#ifndef MIGRAPHX_GUARD_MIGRAPHX_FUSE_ATTENTION_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_FUSE_ATTENTION_HPP

#include <migraphx/config.hpp>
#include <string>
#include <optional>
#include <cstddef>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module_pass_manager;

struct MIGRAPHX_EXPORT fuse_attention
{
    bool attn_enabled = false;
    // Widest Q/V head dim to fuse: past this the fused kernel's per-row
    // accumulator overwhelms the register budget and loses to separate kernels
    std::size_t attn_max_head_dim = 256;
    // Fewest total q rows (batch * heads * seq) to fuse: the fused kernel only
    // parallelizes over blocks of q rows, so fewer rows leave the device idle
    // while separate kernels still parallelize over the kv length
    std::size_t attn_min_q_rows = 1;
    // Longest kv length where head dims past attn_max_head_dim still fuse: a
    // short kv loop keeps the wide accumulator profitable; 0 allows none
    std::size_t attn_wide_head_kv_len         = 0;
    bool flash_decoding_enabled               = false;
    std::size_t flash_decoding_num_splits     = 0;
    std::size_t flash_decoding_threshold      = 32;
    std::size_t flash_decoding_max_splits     = 16;
    std::size_t flash_decoding_min_chunk_size = 32;

    std::string name() const { return "fuse_attention"; }
    void apply(module_pass_manager& mpm) const;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_MIGRAPHX_FUSE_ATTENTION_HPP
