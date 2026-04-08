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
 */
#ifndef MIGRAPHX_GUARD_MIGRAPHX_ATTENTION_FLAGS_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_ATTENTION_FLAGS_HPP

#include <migraphx/config.hpp>
#include <migraphx/bit_flag.hpp>
#include <cstdint>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

enum class attention_flags : std::uint32_t
{
    none           = 0,
    kv_cache       = 1 << 0,
    causal         = 1 << 1,
    prefix_offset  = 1 << 2,
    sliding_window = 1 << 3,
    split_kv       = 1 << 4
};

template <>
struct bit_flag<attention_flags> : std::true_type
{
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
